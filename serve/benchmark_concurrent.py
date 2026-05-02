"""
serve/benchmark_concurrent.py

Concurrency stress-test for the inference gateway.

Approach
────────
1. Warm up with a single-client sweep to establish a latency baseline.
2. Scale up to N concurrent workers, each pulling prompts from a shared queue.
3. Collect structured ``Measurement`` records and compute percentiles.
4. Optionally dump raw data to CSV for downstream analysis (Part C).

    python serve/benchmark_concurrent.py
    python serve/benchmark_concurrent.py --workers 10 --rounds 5 --csv results/bench.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path

from client import InferenceSession, SamplingConfig


# ── Corpus of prompts (varied length) ────────────────────────────────────────

SHORT_PROMPTS = [
    "Define entropy.",
    "What is a quark?",
    "Explain DNS.",
    "What causes tides?",
    "Name three noble gases.",
]

LONG_PROMPTS = [
    "Describe in detail how photosynthesis converts light energy into chemical "
    "energy, including the light-dependent and light-independent reactions.",
    "Explain the process of mRNA translation at the ribosome, covering the "
    "roles of tRNA, codons, and release factors.",
    "Compare and contrast TCP and UDP protocols in terms of reliability, "
    "ordering, use cases, and overhead.",
    "Walk through the lifecycle of an HTTP request from the moment a user "
    "types a URL in the browser to the rendered page.",
    "Discuss the causes, key events, and consequences of the 2008 global "
    "financial crisis in a structured essay format.",
]


@dataclass
class Measurement:
    worker_id: int
    prompt_chars: int
    latency_s: float
    server_s: float
    tokens_out: int
    finish: str
    error: str


def _pct(values: list[float], p: int) -> float:
    """Percentile (nearest-rank)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = int(math.ceil(p / 100.0 * len(s))) - 1
    return s[max(k, 0)]


# ── Worker ───────────────────────────────────────────────────────────────────

async def _worker(
    wid: int,
    queue: asyncio.Queue[str],
    sess: InferenceSession,
    sampling: SamplingConfig,
    results: list[Measurement],
) -> None:
    while True:
        try:
            prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        t0 = time.perf_counter()
        try:
            r = await sess.complete(prompt, sampling)
            results.append(Measurement(
                worker_id=wid,
                prompt_chars=len(prompt),
                latency_s=round(time.perf_counter() - t0, 4),
                server_s=r.get("timing", {}).get("total_s", 0),
                tokens_out=r.get("usage", {}).get("completion_tokens", 0),
                finish=r.get("finish_reason", ""),
                error="",
            ))
        except Exception as exc:
            results.append(Measurement(
                worker_id=wid,
                prompt_chars=len(prompt),
                latency_s=round(time.perf_counter() - t0, 4),
                server_s=0,
                tokens_out=0,
                finish="error",
                error=str(exc),
            ))
        finally:
            queue.task_done()


# ── Main benchmark ───────────────────────────────────────────────────────────

async def run_benchmark(
    url: str,
    workers: int,
    rounds: int,
    max_tokens: int,
    temperature: float,
    csv_path: str | None,
) -> None:
    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    all_prompts = (SHORT_PROMPTS + LONG_PROMPTS) * rounds

    print(f"\n{'═' * 64}")
    print(f"  Concurrency Stress Test")
    print(f"  gateway     : {url}")
    print(f"  workers     : {workers}")
    print(f"  total tasks : {len(all_prompts)}")
    print(f"  max_tokens  : {max_tokens}")
    print(f"{'═' * 64}")

    async with InferenceSession(url=url, pool_size=workers + 10) as sess:

        # ── Phase 1: baseline (sequential) ───────────────────────────────
        print("\n  Phase 1 · baseline (sequential, 5 prompts) …")
        baseline: list[float] = []
        for p in SHORT_PROMPTS:
            t0 = time.perf_counter()
            try:
                await sess.complete(p, sampling)
                baseline.append(time.perf_counter() - t0)
            except Exception:
                pass
        baseline_p50 = _pct(baseline, 50) if baseline else float("inf")
        print(f"    P50 latency = {baseline_p50:.3f}s\n")

        # ── Phase 2: concurrent workers ──────────────────────────────────
        print(f"  Phase 2 · {workers} concurrent workers …")
        queue: asyncio.Queue[str] = asyncio.Queue()
        for p in all_prompts:
            queue.put_nowait(p)

        results: list[Measurement] = []
        wall = time.perf_counter()
        tasks = [
            asyncio.create_task(_worker(w, queue, sess, sampling, results))
            for w in range(workers)
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - wall

    # ── Analyse ──────────────────────────────────────────────────────────
    ok = [m for m in results if not m.error]
    errs = [m for m in results if m.error]

    if not ok:
        print("  ⚠  No successful requests.")
        return

    lats = [m.latency_s for m in ok]
    total_tok = sum(m.tokens_out for m in ok)
    conc_p50 = _pct(lats, 50)
    degradation = ((conc_p50 - baseline_p50) / baseline_p50 * 100) if baseline_p50 else 0

    print(f"\n{'─' * 64}")
    print(f"  Results")
    print(f"{'─' * 64}")
    print(f"  Completed   : {len(ok)}  (errors: {len(errs)})")
    print(f"  Wall time   : {wall:.2f}s")
    print(f"  Throughput  : {len(ok)/wall:.1f} req/s   {total_tok/wall:.0f} tok/s")
    print()
    print(f"  Latency P50 : {_pct(lats, 50):.3f}s")
    print(f"  Latency P95 : {_pct(lats, 95):.3f}s")
    print(f"  Latency P99 : {_pct(lats, 99):.3f}s")
    print(f"  Latency max : {max(lats):.3f}s")
    print()
    print(f"  Baseline P50      : {baseline_p50:.3f}s")
    print(f"  Concurrent P50    : {conc_p50:.3f}s")
    print(f"  Δ degradation     : {degradation:+.1f}%")

    verdict = (
        "✅ Healthy (<20%)" if degradation < 20
        else "⚠️  Moderate (20-50%)" if degradation < 50
        else "❌ Severe (>50%)"
    )
    print(f"  Verdict           : {verdict}")
    print(f"{'─' * 64}\n")

    # ── CSV export ───────────────────────────────────────────────────────
    if csv_path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        hdr = [f.name for f in fields(Measurement)]
        with open(out, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(hdr)
            for m in results:
                w.writerow([getattr(m, h) for h in hdr])
        print(f"  Raw data → {out}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrency stress-test")
    ap.add_argument("--url", default="http://localhost:9000")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    asyncio.run(run_benchmark(
        url=args.url,
        workers=args.workers,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        csv_path=args.csv,
    ))


if __name__ == "__main__":
    main()
