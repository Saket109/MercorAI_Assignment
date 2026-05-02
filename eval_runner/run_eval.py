# eval_runner/run_eval.py
"""
Evaluation pipeline — orchestrates lm-evaluation-harness runs.

Evaluates on:
  1. MMLU         — 5-shot, 57-subject knowledge exam
  2. HellaSwag    — 0-shot, commonsense completion (loglikelihood)
  3. ml_reasoning — 0-shot, custom 15-question ML/CS benchmark

Produces  results/  with per-task JSON + a Markdown summary.

    python eval_runner/run_eval.py
    python eval_runner/run_eval.py --tasks mmlu hellaswag
    python eval_runner/run_eval.py --endpoint https://my-vllm:8000 --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Trigger @register_model decorator
import eval_runner.vllm_model  # noqa: F401

from lm_eval import evaluator
from lm_eval.tasks import TaskManager

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
CUSTOM_DIR = Path(__file__).parent / "custom_task"

# ── Task registry ───────────────────────────────────────────────────────────
TASKS = {
    "mmlu": dict(harness="mmlu", fewshot=5, about="57-subject knowledge exam"),
    "hellaswag": dict(harness="hellaswag", fewshot=0, about="Commonsense sentence completion"),
    "ml_reasoning": dict(harness="ml_reasoning", fewshot=0, about="Custom ML/CS benchmark (15 Qs)"),
}


# ── Pipeline ─────────────────────────────────────────────────────────────────

class EvalPipeline:
    """Runs one or more benchmark tasks and collects results."""

    def __init__(self, model_args: str, limit: int | None = None) -> None:
        self._model_args = model_args
        self._limit = limit
        self._collected: list[dict] = []

    def execute(self, task_key: str) -> dict:
        """Run a single task, save JSON, return raw results."""
        spec = TASKS[task_key]

        print(f"\n{'═' * 66}")
        print(f"  {task_key}  —  {spec['about']}")
        print(f"{'═' * 66}\n")

        # For the custom task, register its directory
        tm_kwargs: dict = {}
        saved_cwd = os.getcwd()

        if task_key == "ml_reasoning":
            os.environ["LM_EVAL_CUSTOM_TASK_DIR"] = str(CUSTOM_DIR)
            tm_kwargs["include_path"] = str(CUSTOM_DIR)
            os.chdir(str(CUSTOM_DIR.resolve()))

        try:
            tm = TaskManager(**tm_kwargs)
            kw: dict = dict(
                model="remote_vllm",
                model_args=self._model_args,
                tasks=[spec["harness"]],
                num_fewshot=spec["fewshot"],
                batch_size=1,
                task_manager=tm,
            )
            if self._limit:
                kw["limit"] = self._limit

            t0 = time.perf_counter()
            raw = evaluator.simple_evaluate(**kw)
            elapsed = round(time.perf_counter() - t0, 2)
        finally:
            os.chdir(saved_cwd)

        raw["_run"] = dict(task=task_key, seconds=elapsed,
                           ts=datetime.now(timezone.utc).isoformat())

        # Persist per-task JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{task_key}.json"
        out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        print(f"  → {out}")

        metrics = self._extract(raw, task_key, elapsed)
        self._collected.append(metrics)
        print(f"  metrics: {metrics}")
        return raw

    # ── metrics extraction ───────────────────────────────────────────

    @staticmethod
    def _extract(raw: dict, task_key: str, elapsed: float) -> dict:
        row: dict = {"task": task_key, "seconds": elapsed}
        for _name, data in raw.get("results", {}).items():
            if not isinstance(data, dict):
                continue
            for k, v in data.items():
                if not isinstance(v, (int, float)):
                    continue
                # Match metric keys like "acc,none", "acc_norm,none",
                # "exact_match,extract_letter", etc.
                base = k.split(",")[0]
                if base in ("acc", "acc_norm", "exact_match") and "stderr" not in k:
                    row[base] = round(v * 100 if v <= 1.0 else v, 2)
        return row

    # ── summary generation ───────────────────────────────────────────

    def write_summary(self) -> Path:
        # Merge current run + any existing result JSONs into one table
        by_task: dict[str, dict] = {}

        # Load previously saved results from disk
        for fp in sorted(RESULTS_DIR.glob("*.json")):
            if fp.name.startswith("eval_cache"):
                continue
            task_key = fp.stem
            try:
                raw = json.loads(fp.read_text())
                elapsed = raw.get("_meta", {}).get("elapsed_s", "-")
                by_task[task_key] = self._extract(raw, task_key, elapsed)
            except Exception:
                pass

        # Override with freshly collected metrics (more up-to-date)
        for m in self._collected:
            by_task[m["task"]] = m

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# Evaluation Results",
            "",
            f"**Date:** {ts}",
            "",
            "| Task | Accuracy % | Acc Norm % | Exact Match % | Time (s) |",
            "|------|-----------|-----------|--------------|----------|",
        ]
        for m in by_task.values():
            lines.append(
                f"| {m.get('task','-')} "
                f"| {m.get('acc','-')} "
                f"| {m.get('acc_norm','-')} "
                f"| {m.get('exact_match','-')} "
                f"| {m.get('seconds','-')} |"
            )
        lines += [
            "",
            "## Notes",
            "- MMLU: 5-shot, accuracy across 57 subjects",
            "- HellaSwag: 0-shot, real token-level loglikelihood via text-completions API",
            "- ml_reasoning: 0-shot, exact letter match on 15 custom ML/CS questions",
            "- All results cached via SHA-256 DiskMemo for deterministic re-runs",
        ]

        out = RESULTS_DIR / "summary.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run LM evaluation benchmarks")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS),
                    choices=list(TASKS), help="Tasks to run (default: all)")
    ap.add_argument("--endpoint", default="http://localhost:8000",
                    help="vLLM base URL")
    ap.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples per task (for quick tests)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model_args = (
        f"endpoint={args.endpoint},"
        f"model_name={args.model_name},"
        f"max_gen_tokens={args.max_tokens},"
        f"temperature={args.temperature}"
    )

    print(f"\n  Endpoint : {args.endpoint}")
    print(f"  Model    : {args.model_name}")
    print(f"  Tasks    : {', '.join(args.tasks)}")
    print(f"  Limit    : {args.limit}")

    pipe = EvalPipeline(model_args, limit=args.limit)

    for tk in args.tasks:
        try:
            pipe.execute(tk)
        except Exception as exc:
            print(f"\n  ✗ {tk} failed: {exc}")
            import traceback; traceback.print_exc()

    summary_path = pipe.write_summary()
    print(f"\n{'═' * 66}")
    print(f"  Summary → {summary_path}")
    print(f"{'═' * 66}\n")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
