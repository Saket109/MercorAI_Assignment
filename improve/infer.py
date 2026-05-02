# improve/infer.py
"""
MMLU STEM inference pipeline with ablation study.

Runs four configurations against the gateway and computes
bootstrap confidence intervals + a permutation significance test.

Configs:
  1. bare       — 0-shot, minimal prompt, T=0
  2. instruct   — + expert-persona instruction header
  3. fewshot     — + subject-matched 5-shot exemplars
  4. normalized  — + robust answer extraction regex

    python improve/infer.py
    python improve/infer.py --url http://localhost:9000 --samples 0
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_DIR = Path(__file__).resolve().parent
_DATA = _DIR / "data"
sys.path.insert(0, str(_DIR))

from optimize_prompt import (
    baseline_prompt,
    instruction_only_prompt,
    normalize_answer,
    optimized_prompt,
)


# ── Gateway helper ──────────────────────────────────────────────────────────

def ask(
    client: httpx.Client,
    url: str,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 15,
) -> str:
    """Send a non-streaming completion and return the generated text."""
    r = client.post(
        f"{url.rstrip('/')}/v1/completions",
        json=dict(prompt=prompt, max_tokens=max_tokens,
                  temperature=temp, top_p=1.0, stream=False),
    )
    r.raise_for_status()
    return r.json().get("text", "")


# ── Bootstrap confidence interval ──────────────────────────────────────────

def bootstrap_ci(
    correct: list[bool], n_boot: int = 2000, alpha: float = 0.05,
) -> tuple[float, float]:
    """Non-parametric bootstrap 95 % CI for accuracy."""
    rng = random.Random(42)
    n = len(correct)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(
        sum(rng.choices(correct, k=n)) / n for _ in range(n_boot)
    )
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (round(lo * 100, 2), round(hi * 100, 2))


# ── Permutation significance test ─────────────────────────────────────────

def permutation_test(
    base_correct: list[bool],
    opt_correct: list[bool],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """
    Paired permutation test for difference in accuracy.
    Returns observed delta, p-value, and whether p < 0.05.
    """
    rng = random.Random(42)
    n = len(base_correct)
    obs_delta = sum(opt_correct) / n - sum(base_correct) / n

    count_ge = 0
    diffs = [int(o) - int(b) for b, o in zip(base_correct, opt_correct)]
    for _ in range(n_perm):
        flipped = [d * rng.choice([-1, 1]) for d in diffs]
        perm_delta = sum(flipped) / n
        if perm_delta >= obs_delta:
            count_ge += 1

    p = count_ge / n_perm
    return dict(
        observed_delta_pct=round(obs_delta * 100, 2),
        p_value=round(p, 4),
        significant=p < 0.05,
        n_permutations=n_perm,
    )


# ── Run one configuration ─────────────────────────────────────────────────

def run_config(
    name: str,
    items: list[dict],
    client: httpx.Client,
    url: str,
    prompt_builder,
    temp: float = 0.0,
    use_normalizer: bool = False,
) -> dict[str, Any]:
    """Evaluate a single ablation config. Returns accuracy + per-item results."""
    print(f"\n{'─' * 56}")
    print(f"  {name}   (T={temp}  normalizer={'ON' if use_normalizer else 'OFF'})")
    print(f"{'─' * 56}")

    correct = 0
    results: list[dict] = []
    t0 = time.perf_counter()

    for i, item in enumerate(items):
        prompt = prompt_builder(item)
        raw = ask(client, url, prompt, temp=temp)

        if use_normalizer:
            pred = normalize_answer(raw)
        else:
            # Simple first-char extraction (baseline behaviour)
            first = raw.strip()[:1].upper()
            pred = first if first in "ABCD" else None

        ok = pred == item["answer"]
        if ok:
            correct += 1

        results.append(dict(
            question=item["question"][:60],
            expected=item["answer"],
            predicted=pred,
            ok=ok,
            raw=raw.strip()[:120],
            subject=item["subject"],
        ))

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(items)}]  acc so far: {correct/(i+1)*100:.1f}%")

    elapsed = time.perf_counter() - t0
    acc = correct / len(items) * 100
    flags = [r["ok"] for r in results]
    ci = bootstrap_ci(flags)

    print(f"  ✓  {name}: {acc:.1f}%  ({correct}/{len(items)})  "
          f"CI=[{ci[0]}, {ci[1]}]  {elapsed:.1f}s")

    return dict(
        name=name, accuracy_pct=round(acc, 2),
        correct=correct, total=len(items),
        ci_95=list(ci), elapsed_s=round(elapsed, 1),
        temperature=temp, normalizer=use_normalizer,
        results=results, flags=flags,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="MMLU STEM ablation pipeline")
    ap.add_argument("--url", default="http://localhost:9000",
                    help="Gateway URL")
    ap.add_argument("--samples", type=int, default=0,
                    help="Limit per subject (0 = all)")
    ap.add_argument("--out", default=str(_DIR),
                    help="Output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    manifest_path = _DATA / "manifest.json"
    if not manifest_path.exists():
        print("Data not found — running prepare_data.py …")
        from prepare_data import download_and_save
        download_and_save()

    manifest = json.loads(manifest_path.read_text())
    all_items: list[dict] = []
    pools: dict[str, list[dict]] = {}

    for subj, meta in manifest.items():
        data = json.loads(Path(meta["path"]).read_text())
        pool = data["fewshot_pool"]
        test = data["test"]
        if args.samples:
            test = test[:args.samples]
        all_items.extend(test)
        pools[subj] = pool

    random.seed(42)
    random.shuffle(all_items)
    print(f"  Loaded {len(all_items)} test items across {len(manifest)} subjects\n")

    # ── Define prompt builders per config ──────────────────────────────
    def build_bare(item: dict) -> str:
        return baseline_prompt(item["question"], item["choices"])

    def build_instruct(item: dict) -> str:
        return instruction_only_prompt(
            item["question"], item["choices"], item["subject"],
        )

    def build_fewshot(item: dict) -> str:
        return optimized_prompt(
            item["question"], item["choices"],
            item["subject"], pools.get(item["subject"], []),
        )

    # ── Run ablation configs ──────────────────────────────────────────
    client = httpx.Client(timeout=120)
    ablation: list[dict] = []

    # Config 1: Bare 0-shot
    c1 = run_config("bare_0shot", all_items, client, args.url,
                     build_bare, temp=0.0, use_normalizer=False)
    ablation.append(c1)

    # Config 2: + Instruction header
    c2 = run_config("instruct_0shot", all_items, client, args.url,
                     build_instruct, temp=0.0, use_normalizer=False)
    ablation.append(c2)

    # Config 3: + Subject-matched 5-shot
    c3 = run_config("instruct_5shot", all_items, client, args.url,
                     build_fewshot, temp=0.0, use_normalizer=False)
    ablation.append(c3)

    # Config 4: + Robust answer normalizer
    c4 = run_config("instruct_5shot_norm", all_items, client, args.url,
                     build_fewshot, temp=0.0, use_normalizer=True)
    ablation.append(c4)

    client.close()

    # ── Statistical test: baseline vs best ────────────────────────────
    perm = permutation_test(c1["flags"], c4["flags"])

    # ── Before/after examples ─────────────────────────────────────────
    examples = []
    for bl, opt in zip(c1["results"], c4["results"]):
        if not bl["ok"] and opt["ok"]:
            examples.append(dict(
                question=bl["question"],
                subject=bl["subject"],
                expected=bl["expected"],
                base_pred=bl["predicted"],
                base_raw=bl["raw"][:150],
                opt_pred=opt["predicted"],
                opt_raw=opt["raw"][:150],
            ))
    examples = examples[:15]

    # ── Save results ──────────────────────────────────────────────────
    summary = dict(
        ablation=[
            dict(
                name=a["name"],
                accuracy_pct=a["accuracy_pct"],
                ci_95=a["ci_95"],
                correct=a["correct"],
                total=a["total"],
                elapsed_s=a["elapsed_s"],
                delta=round(a["accuracy_pct"] - c1["accuracy_pct"], 2),
            )
            for a in ablation
        ],
        significance=perm,
        before_after=examples,
    )

    (out / "baseline_results.json").write_text(
        json.dumps({k: v for k, v in c1.items() if k != "flags"}, indent=2, default=str))
    (out / "optimized_results.json").write_text(
        json.dumps({k: v for k, v in c4.items() if k != "flags"}, indent=2, default=str))
    (out / "ablation_results.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  ABLATION SUMMARY")
    print(f"{'═' * 62}")
    print(f"  {'Config':<25} {'Acc':>6} {'Δ':>6}  {'95% CI':>16}  {'Time':>6}")
    print(f"  {'─'*25} {'─'*6} {'─'*6}  {'─'*16}  {'─'*6}")
    for a in summary["ablation"]:
        d = f"+{a['delta']:.1f}" if a["delta"] > 0 else f"{a['delta']:.1f}"
        ci = f"[{a['ci_95'][0]}, {a['ci_95'][1]}]"
        print(f"  {a['name']:<25} {a['accuracy_pct']:>5.1f}% {d:>6}  {ci:>16}  {a['elapsed_s']:>5.1f}s")

    print(f"\n  Permutation test (bare vs optimized):")
    print(f"    Δ = {perm['observed_delta_pct']:.1f}%  p = {perm['p_value']:.4f}"
          f"  significant = {perm['significant']}")
    print(f"\n  Improvement examples: {len(examples)}")
    print(f"  Results → {out}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
