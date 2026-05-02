# serve/client.py
"""
Inference client for the LLM gateway.

Design
──────
``InferenceSession`` wraps a long-lived, connection-pooled ``httpx.AsyncClient``.
Use it as an async context manager so the pool is cleaned up automatically.

``SamplingConfig`` is a frozen dataclass that doubles as a cache key
(hashable by default).

For callers that don't want async, the module exposes three plain functions
(``complete``, ``stream``, ``complete_many``) that spin up a short-lived
session under the hood.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

import httpx

GATEWAY_URL = "http://localhost:9000"


# ═══════════════════════════════════════════════════════════════════════════════
#  Sampling configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SamplingConfig:
    """
    Frozen (hashable) bundle of decoding knobs.

    ``stop`` is a tuple — not list — so the whole object can be used as a
    dict key, which matters for prompt caching downstream.
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[tuple[str, ...]] = None
    seed: Optional[int] = None
    logprobs: Optional[int] = None

    def as_dict(self) -> dict:
        out: dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.stop:
            out["stop"] = list(self.stop)
        if self.seed is not None:
            out["seed"] = self.seed
        if self.logprobs is not None:
            out["logprobs"] = self.logprobs
        return out


# Handy presets
DETERMINISTIC = SamplingConfig(temperature=0.0, top_p=1.0, seed=42)
CREATIVE      = SamplingConfig(temperature=0.95, top_p=0.95)


# ═══════════════════════════════════════════════════════════════════════════════
#  Async session
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceSession:
    """
    Persistent async session with connection pooling.

    Usage::

        async with InferenceSession() as sess:
            out = await sess.complete("Hello!")
            print(out["text"])
    """

    def __init__(
        self,
        url: str = GATEWAY_URL,
        timeout: float = 120.0,
        pool_size: int = 60,
    ) -> None:
        self._url = url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 3,
            ),
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> InferenceSession:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ── health ───────────────────────────────────────────────────────────

    async def ping(self) -> dict:
        r = await self._http.get("/status")
        r.raise_for_status()
        return r.json()

    # ── single completion ────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        cfg: SamplingConfig | None = None,
    ) -> dict:
        """
        Non-streaming completion.

        Returns the gateway JSON enriched with ``round_trip_s`` measured
        on the client side.
        """
        cfg = cfg or SamplingConfig()
        body = {"prompt": prompt, "stream": False, **cfg.as_dict()}

        t0 = time.perf_counter()
        r = await self._http.post("/v1/completions", json=body)
        r.raise_for_status()
        result = r.json()
        result["round_trip_s"] = round(time.perf_counter() - t0, 4)
        return result

    # ── streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        prompt: str,
        cfg: SamplingConfig | None = None,
    ) -> AsyncIterator[dict]:
        """
        Yields per-token dicts ``{"t": "…", "n": N}`` followed by a
        final summary with ``{"fin": True, …}``.
        """
        cfg = cfg or SamplingConfig()
        body = {"prompt": prompt, "stream": True, **cfg.as_dict()}

        async with self._http.stream("POST", "/v1/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                    yield payload
                    if payload.get("fin"):
                        return
                except json.JSONDecodeError:
                    continue

    # ── fan-out (concurrent batch) ───────────────────────────────────────

    async def complete_many(
        self,
        prompts: list[str],
        cfg: SamplingConfig | None = None,
    ) -> list[dict | Exception]:
        """Fire all *prompts* concurrently; exceptions don't crash the batch."""
        coros = [self.complete(p, cfg) for p in prompts]
        return await asyncio.gather(*coros, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Synchronous helpers (scripts / notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

def _exec(coro):
    """Run a coroutine, handling both fresh and existing event loops."""
    try:
        asyncio.get_running_loop()
        import nest_asyncio  # type: ignore[import-untyped]
        nest_asyncio.apply()
    except RuntimeError:
        pass
    return asyncio.run(coro)


def complete(prompt: str, cfg: SamplingConfig | None = None, url: str = GATEWAY_URL) -> dict:
    """Blocking single completion."""
    async def _go():
        async with InferenceSession(url=url) as s:
            return await s.complete(prompt, cfg)
    return _exec(_go())


def stream(prompt: str, cfg: SamplingConfig | None = None, url: str = GATEWAY_URL) -> Iterator[str | dict]:
    """Blocking streaming — yields token strings then a summary dict."""
    async def _collect():
        out: list[str | dict] = []
        async with InferenceSession(url=url) as s:
            async for chunk in s.stream(prompt, cfg):
                out.append(chunk if chunk.get("fin") else chunk.get("t", ""))
        return out
    yield from _exec(_collect())


def complete_many(prompts: list[str], cfg: SamplingConfig | None = None, url: str = GATEWAY_URL) -> list:
    """Blocking concurrent batch."""
    async def _go():
        async with InferenceSession(url=url) as s:
            return await s.complete_many(prompts, cfg)
    return _exec(_go())


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Batch ──")
    res = complete("What is the capital of France?", DETERMINISTIC)
    print(f"  {res['text']}")
    print(f"  Server: {res['timing']['total_s']}s  Client: {res['round_trip_s']}s\n")

    print("── Stream ──")
    for tok in stream("Explain recursion briefly.", SamplingConfig(max_tokens=80)):
        if isinstance(tok, dict):
            print(f"\n  {tok}")
        else:
            print(tok, end="", flush=True)
    print()
