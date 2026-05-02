# perf/load_test.py
"""
Performance load generator for LLM inference gateway.

Sends concurrent requests with short vs long prompts and collects:
  • TTFT   — time-to-first-token  (streaming mode)
  • TPOT   — time per output token
  • P50 / P95 / P99 tail latency
  • GPU utilisation  (nvidia-smi, when available)

Sweeps across concurrency levels, stop-sequence configs, and
cache-hit vs cache-miss scenarios.

    python perf/load_test.py
    python perf/load_test.py --gateway http://localhost:9000 --concurrency 1,4,8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).resolve().parent
_CSV_OUT = _DIR / "metrics.csv"
_AGG_OUT = _DIR / "metrics_aggregate.csv"

# ── Prompt pools ─────────────────────────────────────────────────────────────
SHORT = [
    "What is 2+2?",
    "Name three primary colours.",
    "Define entropy in one sentence.",
    "What language is CPython written in?",
    "Explain HTTP in ten words.",
]

LONG = [
    (
        "Write a comprehensive guide to building production-grade REST APIs "
        "with Python and FastAPI, covering routing, Pydantic validation, "
        "dependency injection, middleware, error handling, SQLAlchemy "
        "integration, Alembic migrations, JWT authentication, rate limiting, "
        "and containerised deployment with Docker Compose."
    ),
    (
        "Describe the evolution of neural network architectures from "
        "perceptrons through CNNs, RNNs, LSTMs, the original Transformer, "
        "GPT, BERT, T5, LLaMA, and Mistral. For each, explain the core "
        "architectural innovation, training objective, and a notable "
        "real-world application."
    ),
    (
        "Explain the end-to-end lifecycle of a machine-learning project: "
        "problem scoping, data collection and labelling, exploratory "
        "analysis, feature engineering, model selection, cross-validation, "
        "hyperparameter tuning, evaluation metrics, deployment to "
        "production with CI/CD, monitoring, drift detection, and "
        "retraining strategies. Use a fraud-detection system as the "
        "running example."
    ),
    (
        "Write a detailed comparison of memory-management strategies in "
        "modern LLM serving systems: paged attention (vLLM), continuous "
        "batching, speculative decoding, KV-cache quantisation, and "
        "prefix caching. Discuss trade-offs between throughput, latency, "
        "and memory footprint for each technique."
    ),
    (
        "Trace the history of computing from Babbage's Analytical Engine "
        "through Turing machines, ENIAC, transistors, integrated circuits, "
        "UNIX, the internet, cloud computing, GPUs for deep learning, and "
        "modern AI accelerators like TPUs and Trainium. Highlight the key "
        "figures and breakthroughs at each stage."
    ),
]

STOP_CONFIGS = {
    "none":    None,
    "period":  ["."],
    "newline": ["\n"],
}


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Probe:
    """Metrics captured for a single request."""
    scenario: str = ""
    prompt_len: str = ""         # "short" | "long"
    concurrency: int = 1
    stop_cfg: str = "none"
    cached: bool = False
    run: int = 0
    req: int = 0

    ttft_s: Optional[float] = None
    latency_s: float = 0.0
    tokens: int = 0
    tok_per_s: float = 0.0
    finish: str = ""

    gpu_pct: Optional[float] = None
    gpu_mem_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None


@dataclass
class Digest:
    """Aggregated metrics for one scenario."""
    scenario: str
    prompt_len: str
    concurrency: int
    stop_cfg: str
    cached: bool
    n: int = 0

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    lat_min: float = 0.0
    lat_max: float = 0.0
    lat_mean: float = 0.0

    ttft_p50: Optional[float] = None
    ttft_p95: Optional[float] = None
    ttft_p99: Optional[float] = None

    avg_tok_s: float = 0.0
    sys_tok_s: float = 0.0
    req_s: float = 0.0
    wall_s: float = 0.0

    gpu_pct: Optional[float] = None


# ── GPU probe ────────────────────────────────────────────────────────────────

def gpu_snapshot() -> dict | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            timeout=5, text=True,
        )
        parts = raw.strip().split(", ")
        return dict(gpu_pct=float(parts[0]),
                    gpu_mem_mb=float(parts[1]),
                    gpu_total_mb=float(parts[2])) if len(parts) >= 3 else None
    except Exception:
        return None


# ── HTTP helpers (adapted to our gateway's /v1/completions + SSE) ────────

async def fire_stream(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> dict:
    """Send a streaming request through our gateway and measure TTFT."""
    body: dict = dict(prompt=prompt, stream=True, max_tokens=max_tokens,
                      temperature=temperature, top_p=0.9)
    if stop:
        body["stop"] = stop

    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    finish = ""

    async with client.stream("POST", f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Our gateway sends  {"fin": true, "tokens": N, ...}  at the end
            if chunk.get("fin"):
                tokens = chunk.get("tokens", tokens)
                finish = "done"
                break

            # Token chunk:  {"t": "word", "n": 1}
            if "t" in chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1

    elapsed = time.perf_counter() - t0
    return dict(ttft_s=round(ttft, 6) if ttft else None,
                latency_s=round(elapsed, 6),
                tokens=tokens,
                tok_per_s=round(tokens / elapsed, 2) if elapsed > 0 else 0,
                finish=finish)


async def fire_batch(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> dict:
    """Non-streaming request; no TTFT measurement."""
    body: dict = dict(prompt=prompt, stream=False, max_tokens=max_tokens,
                      temperature=temperature, top_p=0.9)
    if stop:
        body["stop"] = stop

    t0 = time.perf_counter()
    r = await client.post(f"{url}/v1/completions", json=body)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0

    toks = data.get("usage", {}).get("completion_tokens", 0)
    return dict(ttft_s=None,
                latency_s=round(elapsed, 6),
                tokens=toks,
                tok_per_s=round(toks / elapsed, 2) if elapsed > 0 else 0,
                finish=data.get("finish_reason", ""))


# ── Percentile helper ────────────────────────────────────────────────────────

def pct(data: list[float], p: float) -> float:
    k = (p / 100) * (len(data) - 1)
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (k - lo) * (data[hi] - data[lo]) if lo != hi else data[lo]


# ── Scenario runner ──────────────────────────────────────────────────────────

async def run_scenario(
    url: str,
    prompts: list[str],
    prompt_len: str,
    concurrency: int,
    stop_name: str,
    stop_seqs: list[str] | None,
    cached: bool,
    max_tokens: int,
    temperature: float,
    rounds: int,
    streaming: bool,
) -> tuple[list[Probe], Digest]:
    tag = f"{prompt_len}_c{concurrency}_{stop_name}_{'hit' if cached else 'miss'}"
    probes: list[Probe] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(180),
                                  limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        wall_t0 = time.perf_counter()

        for rnd in range(rounds):
            coros = []
            for rid in range(concurrency):
                p = prompts[0] if cached else prompts[(rnd * concurrency + rid) % len(prompts)]
                fn = fire_stream if streaming else fire_batch
                coros.append(fn(client, url, p, max_tokens, temperature, stop_seqs))

            results = await asyncio.gather(*coros, return_exceptions=True)
            gpu = gpu_snapshot()

            for rid, res in enumerate(results):
                pr = Probe(scenario=tag, prompt_len=prompt_len, concurrency=concurrency,
                           stop_cfg=stop_name, cached=cached, run=rnd, req=rid)
                if isinstance(res, Exception):
                    pr.latency_s = -1
                    pr.finish = f"ERR:{res}"
                else:
                    pr.ttft_s = res["ttft_s"]
                    pr.latency_s = res["latency_s"]
                    pr.tokens = res["tokens"]
                    pr.tok_per_s = res["tok_per_s"]
                    pr.finish = res["finish"]
                if gpu:
                    pr.gpu_pct = gpu["gpu_pct"]
                    pr.gpu_mem_mb = gpu["gpu_mem_mb"]
                    pr.gpu_total_mb = gpu["gpu_total_mb"]
                probes.append(pr)

        wall = time.perf_counter() - wall_t0

    # ── Aggregate ────────────────────────────────────────────────────────
    ok = [p for p in probes if p.latency_s > 0]
    if not ok:
        return probes, Digest(scenario=tag, prompt_len=prompt_len,
                              concurrency=concurrency, stop_cfg=stop_name,
                              cached=cached, wall_s=wall)

    lats = sorted(p.latency_s for p in ok)
    ttfts = sorted(p.ttft_s for p in ok if p.ttft_s is not None)
    total_tok = sum(p.tokens for p in ok)
    gpus = [p.gpu_pct for p in ok if p.gpu_pct is not None]

    d = Digest(
        scenario=tag, prompt_len=prompt_len, concurrency=concurrency,
        stop_cfg=stop_name, cached=cached, n=len(ok),
        p50=round(pct(lats, 50), 4), p95=round(pct(lats, 95), 4),
        p99=round(pct(lats, 99), 4),
        lat_min=round(lats[0], 4), lat_max=round(lats[-1], 4),
        lat_mean=round(statistics.mean(lats), 4),
        ttft_p50=round(pct(ttfts, 50), 4) if ttfts else None,
        ttft_p95=round(pct(ttfts, 95), 4) if ttfts else None,
        ttft_p99=round(pct(ttfts, 99), 4) if ttfts else None,
        avg_tok_s=round(statistics.mean(p.tok_per_s for p in ok), 2),
        sys_tok_s=round(total_tok / wall, 2) if wall else 0,
        req_s=round(len(ok) / wall, 2) if wall else 0,
        wall_s=round(wall, 4),
        gpu_pct=round(statistics.mean(gpus), 1) if gpus else None,
    )
    return probes, d


# ── CSV persistence ──────────────────────────────────────────────────────────

PROBE_COLS = [
    "scenario", "prompt_len", "concurrency", "stop_cfg", "cached",
    "run", "req", "ttft_s", "latency_s", "tokens", "tok_per_s", "finish",
    "gpu_pct", "gpu_mem_mb", "gpu_total_mb",
]
DIGEST_COLS = [
    "scenario", "prompt_len", "concurrency", "stop_cfg", "cached", "n",
    "p50", "p95", "p99", "lat_min", "lat_max", "lat_mean",
    "ttft_p50", "ttft_p95", "ttft_p99",
    "avg_tok_s", "sys_tok_s", "req_s", "wall_s", "gpu_pct",
]


def persist_csv(probes: list[Probe], digests: list[Digest]) -> None:
    with open(_CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_COLS)
        w.writeheader()
        for p in probes:
            w.writerow({k: asdict(p)[k] for k in PROBE_COLS})
    print(f"  ✓ Per-request  → {_CSV_OUT}")

    with open(_AGG_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DIGEST_COLS)
        w.writeheader()
        for d in digests:
            w.writerow({k: asdict(d)[k] for k in DIGEST_COLS})
    print(f"  ✓ Aggregates   → {_AGG_OUT}")


# ── Pretty print ─────────────────────────────────────────────────────────────

def show(d: Digest) -> None:
    ttft = f"{d.ttft_p50:.4f}" if d.ttft_p50 is not None else "  n/a "
    gpu = f"{d.gpu_pct:.0f}%" if d.gpu_pct is not None else " n/a"
    print(f"  │ {d.scenario:<42} │ {d.n:>3} │"
          f" {d.p50:>7.3f} {d.p95:>7.3f} {d.p99:>7.3f} │"
          f" {ttft:>6} │ {d.avg_tok_s:>7.1f} │ {d.req_s:>5.1f} │ {gpu:>4} │")


# ── Main ─────────────────────────────────────────────────────────────────────

async def execute(args) -> None:
    levels = [int(c) for c in args.concurrency.split(",")]

    print(f"\n{'═' * 78}")
    print(f"  LLM Load Test")
    print(f"{'═' * 78}")
    print(f"  Gateway      : {args.gateway}")
    print(f"  Concurrency  : {levels}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Tokens       : short={args.short_tokens}, long={args.long_tokens}")
    print(f"  Streaming    : {args.streaming}")
    print(f"{'═' * 78}\n")

    all_probes: list[Probe] = []
    all_digests: list[Digest] = []

    # Build test matrix
    configs = []
    for conc in levels:
        for plen, pool, mt in [("short", SHORT, args.short_tokens),
                                ("long", LONG, args.long_tokens)]:
            for sname, sseqs in STOP_CONFIGS.items():
                for cached in (False, True):
                    configs.append(dict(prompts=pool, prompt_len=plen,
                                        concurrency=conc, stop_name=sname,
                                        stop_seqs=sseqs, cached=cached,
                                        max_tokens=mt))

    total = len(configs)

    # Table header
    print(f"  ┌{'─' * 43}┬{'─' * 5}┬{'─' * 25}┬{'─' * 8}┬{'─' * 9}┬{'─' * 7}┬{'─' * 6}┐")
    print(f"  │ {'Scenario':<42}│ {'N':>3} │"
          f" {'P50':>7} {'P95':>7} {'P99':>7} │"
          f" {'TTFT':>6} │ {'Tok/s':>7} │ {'R/s':>5} │ {'GPU':>4} │")
    print(f"  ├{'─' * 43}┼{'─' * 5}┼{'─' * 25}┼{'─' * 8}┼{'─' * 9}┼{'─' * 7}┼{'─' * 6}┤")

    for i, cfg in enumerate(configs, 1):
        probes, digest = await run_scenario(
            url=args.gateway, temperature=args.temperature,
            rounds=args.rounds, streaming=args.streaming, **cfg,
        )
        all_probes.extend(probes)
        all_digests.append(digest)
        show(digest)

    print(f"  └{'─' * 43}┴{'─' * 5}┴{'─' * 25}┴{'─' * 8}┴{'─' * 9}┴{'─' * 7}┴{'─' * 6}┘")

    persist_csv(all_probes, all_digests)

    print(f"\n  ✅  {len(all_probes)} requests across {total} scenarios.")
    print(f"  📁  {_DIR}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM inference load test")
    ap.add_argument("--gateway", default="http://localhost:9000",
                    help="Gateway base URL")
    ap.add_argument("--concurrency", default="1,2,4",
                    help="Comma-separated concurrency levels")
    ap.add_argument("--rounds", type=int, default=2,
                    help="Rounds per scenario")
    ap.add_argument("--short-tokens", type=int, default=50)
    ap.add_argument("--long-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    args = ap.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
