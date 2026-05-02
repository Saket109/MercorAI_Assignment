# serve/serve.py
"""
Inference gateway — thin async layer in front of a remote vLLM backend.

Architecture
────────────
The gateway is organised around a ``BackendPool`` that encapsulates the
httpx connection pool and retry logic.  Endpoint handlers are intentionally
minimal: they validate input, delegate to the pool, and shape the response.

Endpoints
─────────
  GET  /status          Health / model list
  POST /v1/completions  Unified text completion (batch + SSE streaming)

Run
───
  python serve/serve.py                 # defaults from .env
  python serve/serve.py --port 8080     # override
  make serve
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import cfg

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gateway")


# ═══════════════════════════════════════════════════════════════════════════════
#  Backend connection pool
# ═══════════════════════════════════════════════════════════════════════════════

class BackendPool:
    """
    Manages a persistent, connection-pooled ``httpx.AsyncClient`` that
    talks to the vLLM chat-completions endpoint.

    All HTTP calls go through :meth:`request` which handles retries on
    transient server errors (5xx / timeouts).
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.request_timeout),
            limits=httpx.Limits(
                max_connections=cfg.pool_size,
                max_keepalive_connections=cfg.keepalive,
            ),
        )
        logger.info("Pool open  → %s  (connections=%d)", cfg.backend_url, cfg.pool_size)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("Pool closed")

    @property
    def http(self) -> httpx.AsyncClient:
        assert self._client is not None, "BackendPool not initialised"
        return self._client

    # ── retry wrapper ────────────────────────────────────────────────────

    async def request(
        self,
        method: str,
        url: str,
        *,
        payload: dict | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """
        Fire an HTTP request with automatic retry on 5xx / connectivity errors.
        When *stream=True* the caller owns closing the response.
        """
        last_err: Exception | None = None

        for attempt in range(1, cfg.max_retries + 1):
            try:
                if stream:
                    req = self.http.build_request(method, url, json=payload)
                    resp = await self.http.send(req, stream=True)
                else:
                    resp = await self.http.request(method, url, json=payload)

                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp

                logger.warning("Attempt %d/%d → HTTP %d", attempt, cfg.max_retries, resp.status_code)
                last_err = httpx.HTTPStatusError(
                    str(resp.status_code), request=resp.request, response=resp
                )
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                logger.warning("Attempt %d/%d → %s", attempt, cfg.max_retries, type(exc).__name__)
                last_err = exc

        raise HTTPException(status_code=502, detail=f"Backend unavailable: {last_err}")


pool = BackendPool()


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI application
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _lifespan(_: FastAPI):
    await pool.open()
    logger.info("Model: %s", cfg.model_name)
    yield
    await pool.close()


app = FastAPI(
    title="LLM Inference Gateway",
    version="0.1.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ────────────────────────────────────────────────

@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - t0:.4f}"
    return response


# ── Schemas ──────────────────────────────────────────────────────────────────

class CompletionInput(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    logprobs: Optional[int] = Field(None, ge=0, le=20)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_chat_payload(inp: CompletionInput) -> dict:
    """Map our schema → OpenAI chat-completion format expected by vLLM."""
    body: dict = {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": inp.prompt}],
        "max_tokens": inp.max_tokens,
        "temperature": inp.temperature,
        "top_p": inp.top_p,
        "stream": inp.stream,
    }
    if inp.stop:
        body["stop"] = inp.stop
    if inp.seed is not None:
        body["seed"] = inp.seed
    if inp.logprobs is not None:
        body["logprobs"] = True
        body["top_logprobs"] = inp.logprobs
    return body


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    """Liveness probe — also lists models available on the backend."""
    try:
        resp = await pool.http.get(cfg.models_endpoint, timeout=8.0)
        data = resp.json().get("data", [])
        return {"ok": True, "models": [m["id"] for m in data]}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.post("/v1/completions")
async def completions(inp: CompletionInput):
    """
    Text completion gateway.

    Returns JSON when ``stream=false`` (default), or an SSE
    ``text/event-stream`` when ``stream=true``.
    """
    body = _to_chat_payload(inp)

    if not inp.stream:
        return await _handle_batch(body)
    return StreamingResponse(_handle_stream(body), media_type="text/event-stream")


# ── Batch (non-streaming) ───────────────────────────────────────────────────

async def _handle_batch(body: dict) -> JSONResponse:
    clock = time.perf_counter()
    resp = await pool.request("POST", cfg.completions_endpoint, payload=body)
    elapsed = time.perf_counter() - clock

    data = resp.json()
    first = data["choices"][0]
    return JSONResponse({
        "text": first["message"]["content"],
        "model": data.get("model", cfg.model_name),
        "finish_reason": first.get("finish_reason", ""),
        "usage": data.get("usage", {}),
        "timing": {"total_s": round(elapsed, 4)},
    })


# ── Streaming (SSE) ─────────────────────────────────────────────────────────

async def _handle_stream(body: dict):
    """Async generator yielding SSE frames."""
    clock = time.perf_counter()
    first_token_at: float | None = None
    n_tokens = 0

    resp = await pool.request("POST", cfg.completions_endpoint, payload=body, stream=True)
    try:
        async for raw_line in resp.aiter_lines():
            if not raw_line.startswith("data:"):
                continue
            fragment = raw_line[5:].strip()
            if fragment == "[DONE]":
                break
            try:
                piece = json.loads(fragment)
                content = piece["choices"][0].get("delta", {}).get("content", "")
                if not content:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter() - clock
                n_tokens += 1
                yield f"data: {json.dumps({'t': content, 'n': n_tokens})}\n\n"
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        await resp.aclose()

    total = time.perf_counter() - clock
    tpot = (total - (first_token_at or 0)) / max(n_tokens - 1, 1)
    yield f"data: {json.dumps({'fin': True, 'tokens': n_tokens, 'total_s': round(total, 4), 'ttft_s': round(first_token_at or 0, 4), 'tpot_s': round(tpot, 4)})}\n\n"


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=cfg.gateway_host)
    ap.add_argument("--port", type=int, default=cfg.gateway_port)
    args = ap.parse_args()

    uvicorn.run("serve:app", host=args.host, port=args.port, reload=False)
