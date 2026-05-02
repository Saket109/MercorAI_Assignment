# serve/sample_run.py
"""
End-to-end showcase of the inference gateway.

Runs a sequence of *scenarios* — each one exercises a different capability
(decoding presets, streaming, fan-out, determinism).  Output is self-documenting,
so this doubles as acceptance-test evidence for Part A.

    python serve/sample_run.py
    python serve/sample_run.py --gateway http://my-host:8080
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Coroutine

from client import InferenceSession, SamplingConfig, DETERMINISTIC, CREATIVE


# ── Helpers ──────────────────────────────────────────────────────────────────

def heading(text: str) -> None:
    w = 62
    print(f"\n┌{'─' * w}┐")
    print(f"│ {text:<{w - 1}}│")
    print(f"└{'─' * w}┘")


@dataclass
class Scenario:
    name: str
    prompt: str
    sampling: SamplingConfig


# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = [
    Scenario(
        "Low temperature — factual",
        "Define the Heisenberg uncertainty principle in one paragraph.",
        SamplingConfig(max_tokens=120, temperature=0.15),
    ),
    Scenario(
        "High temperature — creative writing",
        "Write a four-line poem about a robot learning to paint.",
        SamplingConfig(max_tokens=80, temperature=0.92, top_p=0.95),
    ),
    Scenario(
        "Deterministic code generation",
        "Implement binary search in Python.\n```python\n",
        SamplingConfig(max_tokens=200, temperature=0.0, stop=("```",), seed=7),
    ),
    Scenario(
        "Stop-sequence test",
        "Count from one to twenty, separated by commas.",
        SamplingConfig(max_tokens=100, stop=("ten",)),
    ),
]


# ── Runners ──────────────────────────────────────────────────────────────────

async def demo_batch(sess: InferenceSession) -> None:
    """Non-streaming completions with different sampling presets."""
    heading("1 ▸ Batch completions — varying decoding configs")

    for sc in SCENARIOS:
        print(f"\n  ● {sc.name}")
        print(f"    prompt      : {sc.prompt[:55]}…")
        print(f"    temperature : {sc.sampling.temperature}  "
              f"top_p: {sc.sampling.top_p}  "
              f"stop: {sc.sampling.stop}")

        try:
            r = await sess.complete(sc.prompt, sc.sampling)
            text = r["text"].strip().replace("\n", "\n    ")
            print(f"    ──────────")
            print(f"    {text[:280]}")
            print(f"    ──────────")
            print(f"    finish : {r['finish_reason']}  "
                  f"latency : {r['timing']['total_s']}s  "
                  f"tokens  : {r.get('usage', {}).get('completion_tokens', '?')}")
        except Exception as exc:
            print(f"    ✗ {exc}")


async def demo_stream(sess: InferenceSession) -> None:
    """Token-by-token SSE streaming."""
    heading("2 ▸ Streaming output")
    prompt = "Explain what a neural network is in exactly three sentences."
    print(f"  prompt: {prompt}\n  ", end="")

    try:
        async for piece in sess.stream(prompt, SamplingConfig(max_tokens=150)):
            if piece.get("fin"):
                print(f"\n\n  ⏱  tokens={piece['tokens']}  "
                      f"ttft={piece['ttft_s']}s  "
                      f"tpot={piece['tpot_s']}s  "
                      f"total={piece['total_s']}s")
            else:
                print(piece.get("t", ""), end="", flush=True)
    except Exception as exc:
        print(f"\n  ✗ {exc}")


async def demo_fanout(sess: InferenceSession) -> None:
    """Concurrent fan-out proving the gateway handles parallel load."""
    heading("3 ▸ Concurrent fan-out (8 prompts)")
    questions = [
        "What is photosynthesis?",
        "How do vaccines work?",
        "Explain supply and demand.",
        "What causes earthquakes?",
        "Why do leaves change color?",
        "How does GPS work?",
        "What is dark matter?",
        "Explain the Doppler effect.",
    ]
    sampling = SamplingConfig(max_tokens=60, temperature=0.5)

    wall = time.perf_counter()
    results = await sess.complete_many(questions, sampling)
    wall = time.perf_counter() - wall

    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [{idx + 1}] ✗ {r}")
        else:
            snippet = r["text"][:70].replace("\n", " ")
            print(f"  [{idx + 1}] {snippet}…  ({r['timing']['total_s']}s)")

    ok_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"\n  wall={wall:.2f}s  throughput={ok_count / wall:.1f} req/s")


async def demo_determinism(sess: InferenceSession) -> None:
    """Verify that fixed seed + temp=0 gives reproducible output."""
    heading("4 ▸ Determinism check (seed=42, temp=0)")
    prompt = "What is 7 × 8? Reply with only the number."

    outputs: list[str] = []
    for trial in range(1, 4):
        r = await sess.complete(prompt, DETERMINISTIC)
        text = r["text"].strip()
        outputs.append(text)
        print(f"  run {trial}: \"{text}\"")

    if len(set(outputs)) == 1:
        print("  ✅  All runs identical — determinism confirmed.")
    else:
        print("  ⚠   Divergence detected — see guardrails/ for analysis.")


# ── Entrypoint ───────────────────────────────────────────────────────────────

async def run_all(url: str) -> None:
    async with InferenceSession(url=url) as sess:
        # quick connectivity check
        try:
            info = await sess.ping()
            print(f"Gateway reachable — models: {info.get('models', '?')}")
        except Exception as exc:
            print(f"⚠  Cannot reach gateway at {url}: {exc}")
            return

        await demo_batch(sess)
        await demo_stream(sess)
        await demo_fanout(sess)
        await demo_determinism(sess)

    heading("All scenarios complete ✓")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample inference scenarios")
    ap.add_argument("--gateway", default="http://localhost:9000")
    args = ap.parse_args()
    asyncio.run(run_all(args.gateway))


if __name__ == "__main__":
    main()
