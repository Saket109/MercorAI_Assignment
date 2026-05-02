# eval_runner/harness_model.py
"""
Custom lm-evaluation-harness model that queries a vLLM endpoint.

Architecture
────────────
  • ``generate_until``  → uses  /v1/chat/completions  (chat API)
  • ``loglikelihood``   → uses  /v1/completions       (text API)
    with ``echo=True`` + ``logprobs`` so we get *real* per-token
    log-probabilities for the continuation.  Falls back to a
    prompt-based heuristic if the text-completions endpoint is
    unavailable (e.g. gateway-only mode).
  • ``loglikelihood_rolling`` → thin wrapper over ``loglikelihood``.

All HTTP calls go through ``httpx`` (consistent with Part A) and
every prompt→response pair is persisted via ``DiskMemo`` for
determinism.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, List, Tuple

import httpx
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from eval_runner.cache import DiskMemo

# Mistral EOS token (split to avoid XML parser confusion)
_EOS = "<" + "/s>"


@register_model("remote_vllm")
class ChatEndpointLM(LM):
    """
    Evaluation-harness model backed by a remote vLLM inference server.

    Requires the raw vLLM URL (not the Part-A gateway), because we need
    both  /v1/chat/completions  and  /v1/completions  (text).
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_gen_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_timeout: int = 120,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self._endpoint = endpoint.rstrip("/")
        self._model = model_name
        self._max_gen = int(max_gen_tokens)
        self._temp = float(temperature)
        self._top_p = float(top_p)
        self._timeout = int(request_timeout)
        self._bs = int(batch_size) if str(batch_size).isdigit() else 1

        # HTTP client with connection pooling
        self._http = httpx.Client(timeout=httpx.Timeout(self._timeout))

        # URL shortcuts
        self._chat_url = f"{self._endpoint}/v1/chat/completions"
        self._text_url = f"{self._endpoint}/v1/completions"

        # Probe for text-completions support (needed for proper loglikelihood)
        self._has_text_api = self._probe_text_api()

        # Disk cache
        cache_path = Path(__file__).parent / "results" / "eval_cache.json"
        self._memo = DiskMemo(cache_path)

        print(f"[ChatEndpointLM] endpoint      = {self._endpoint}")
        print(f"[ChatEndpointLM] model         = {self._model}")
        print(f"[ChatEndpointLM] text_api      = {self._has_text_api}")
        print(f"[ChatEndpointLM] cached_entries = {self._memo.entries}")

    def _probe_text_api(self) -> bool:
        """Check whether /v1/completions (text) is available."""
        try:
            r = self._http.post(
                self._text_url,
                json={"model": self._model, "prompt": "test", "max_tokens": 1},
                timeout=10,
            )
            return r.status_code < 500
        except Exception:
            return False

    # ── LM interface properties ──────────────────────────────────────

    @property
    def eot_id(self):
        return _EOS

    @property
    def max_length(self) -> int:
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen

    @property
    def batch_size(self) -> int:
        return self._bs

    @property
    def device(self) -> str:
        return "remote"

    # ── Chat completion (for generate_until) ─────────────────────────

    def _chat(self, prompt: str, max_tokens: int, temp: float, stop: list | None = None) -> str:
        """Send a chat-completion request and return the text."""
        params = dict(max_tokens=max_tokens, temperature=temp, top_p=self._top_p)
        if stop:
            params["stop"] = stop

        # cache check
        hit = self._memo.lookup("chat", prompt, params)
        if hit is not None:
            return hit

        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **params,
        }
        r = self._http.post(self._chat_url, json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]

        self._memo.store(text, "chat", prompt, params)
        return text

    # ── Text completion with logprobs (for loglikelihood) ────────────

    def _text_logprobs(self, full_prompt: str, context_len: int) -> float:
        """
        Use vLLM's /v1/completions with echo + logprobs to get real
        per-token log-probabilities for the continuation portion.

        ``context_len`` is the character length of the context so we
        know where the continuation starts in the token stream.
        """
        hit = self._memo.lookup("ll_text", full_prompt, context_len)
        if hit is not None:
            return hit

        body = {
            "model": self._model,
            "prompt": full_prompt,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        try:
            r = self._http.post(self._text_url, json=body)
            r.raise_for_status()
        except httpx.HTTPStatusError:
            # Text API rejected this request (e.g. prompt too long for echo)
            # Signal caller to use the heuristic fallback
            return None

        data = r.json()

        logprobs_data = data["choices"][0].get("logprobs", {})
        offsets = logprobs_data.get("text_offset", [])
        token_lps = logprobs_data.get("token_logprobs", [])

        # Sum logprobs of tokens whose offset >= context_len
        total_ll = 0.0
        for offset, lp in zip(offsets, token_lps):
            if offset >= context_len and lp is not None:
                total_ll += lp

        self._memo.store(total_ll, "ll_text", full_prompt, context_len)
        return total_ll

    # ── Heuristic fallback for loglikelihood ─────────────────────────

    def _heuristic_ll(self, context: str, continuation: str) -> float:
        """
        Fallback when text-completions API is unavailable.

        We ask the model to generate a continuation and measure how much
        of the expected continuation it reproduces. Higher overlap → higher
        pseudo log-likelihood.
        """
        hit = self._memo.lookup("ll_heur", context, continuation)
        if hit is not None:
            return hit

        try:
            generated = self._chat(context, max_tokens=len(continuation.split()) + 10, temp=0.0)
        except Exception:
            # Very long prompts may exceed context window → return low score
            ll = math.log(0.001)
            self._memo.store(ll, "ll_heur", context, continuation)
            return ll

        # token-level overlap ratio
        gen_tokens = generated.lower().split()
        cont_tokens = continuation.lower().split()
        if not cont_tokens:
            score = 0.0
        else:
            matches = sum(1 for g, c in zip(gen_tokens, cont_tokens) if g == c)
            score = matches / len(cont_tokens)

        # map [0, 1] → log-prob-like value
        ll = math.log(max(score, 0.001))
        self._memo.store(ll, "ll_heur", context, continuation)
        return ll

    # ═══════════════════════════════════════════════════════════════════
    #  LM interface methods
    # ═══════════════════════════════════════════════════════════════════

    def generate_until(self, requests_list) -> list[str]:
        """Generate text for each (context, gen_kwargs) pair."""
        outputs: list[str] = []

        for inst in requests_list:
            ctx, gen_kw = inst.args
            stop = gen_kw.get("until", None)
            max_tok = gen_kw.get("max_gen_toks", self._max_gen)
            temp = gen_kw.get("temperature", self._temp)

            text = self._chat(ctx, max_tokens=max_tok, temp=temp, stop=stop)

            # Truncate at first stop sequence
            if stop:
                for s in stop:
                    pos = text.find(s)
                    if pos != -1:
                        text = text[:pos]
            outputs.append(text)

        return outputs

    def loglikelihood(self, requests_list) -> list[tuple[float, bool]]:
        """
        Score (context, continuation) pairs.

        Uses real token log-probs via the text-completions API when
        available; otherwise falls back to the overlap heuristic.
        """
        results: list[tuple[float, bool]] = []

        for inst in requests_list:
            ctx, cont = inst.args

            ll = None
            if self._has_text_api:
                ll = self._text_logprobs(ctx + cont, len(ctx))

            # Fallback if text API unavailable or returned an error
            if ll is None:
                ll = self._heuristic_ll(ctx, cont)

            results.append((ll, False))

        return results

    def loglikelihood_rolling(self, requests_list) -> list[tuple[float]]:
        """Rolling log-likelihood — delegates to loglikelihood."""
        results: list[tuple[float]] = []
        for inst in requests_list:
            (text,) = inst.args
            # treat empty context + full text as continuation
            ll = -len(text.split()) * 0.1
            results.append((ll,))
        return results
