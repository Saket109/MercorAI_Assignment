# guardrails/validate.py
"""
Reliability guardrails for the LLM evaluation pipeline.

Provides:
  1. GreedyConfig      — locked decoding params (temp=0, seed, top_p=1)
  2. Reproducibility   — same prompt N times → identical SHA-256 hashes
  3. Format validators — regex for MCQ letter, schema for dataset.json,
                         sanity checks for free-form output
  4. CLI runner        — executes the full suite against the gateway

    python guardrails/validate.py
    python guardrails/validate.py --url http://localhost:9000 --repeats 5 -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "eval_runner" / "custom_task" / "dataset.json"
_RESULTS = _ROOT / "eval_runner" / "results"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Greedy decoding configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GreedyConfig:
    """
    Deterministic decoding parameters.

    temperature=0  disables sampling (argmax),
    top_p=1        no nucleus truncation,
    seed=42        vLLM server-side RNG seed.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 256

    def as_payload(self, prompt: str, *, stream: bool = False) -> dict:
        """Build a /v1/completions payload for our gateway."""
        return dict(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stream=stream,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. HTTP helper
# ═══════════════════════════════════════════════════════════════════════════════

def _call(url: str, prompt: str, cfg: GreedyConfig, timeout: int = 120) -> str:
    """Fire a single completion request and return the generated text."""
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{url.rstrip('/')}/v1/completions", json=cfg.as_payload(prompt))
        r.raise_for_status()
        data = r.json()
    # Our gateway returns  {"text": "...", ...}
    return data.get("text", data.get("output", ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Reproducibility check
# ═══════════════════════════════════════════════════════════════════════════════

_TEST_PROMPTS = [
    "What is 2 + 2?",
    "Name the three primary colours of light.",
    "Return ONLY the letter: A, B, C, or D.\nAnswer:",
]


def check_reproducibility(
    url: str,
    prompts: list[str] | None = None,
    cfg: GreedyConfig | None = None,
    repeats: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Send each prompt ``repeats`` times and verify SHA-256 hash identity.

    Returns {"passed": bool, "checks": [{"prompt", "ok", "unique", ...}, ...]}
    """
    cfg = cfg or GreedyConfig()
    prompts = prompts or _TEST_PROMPTS
    checks: list[dict] = []
    all_ok = True

    for prompt in prompts:
        texts: list[str] = []
        for _ in range(repeats):
            texts.append(_call(url, prompt, cfg))

        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        unique = len(set(hashes))
        ok = unique == 1

        entry: dict[str, Any] = dict(
            prompt=prompt[:80], ok=ok, unique=unique,
        )
        if verbose or not ok:
            entry["responses"] = texts
        if not ok:
            all_ok = False

        checks.append(entry)

    return dict(passed=all_ok, checks=checks)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Output validators
# ═══════════════════════════════════════════════════════════════════════════════

# 4a — MCQ letter extractor ──────────────────────────────────────────────────
#  Accepts:  "C", "C)", "C) Paris", "A. correct", "  b "
_LETTER_RE = re.compile(r"^\s*([A-Da-d])\b", re.DOTALL)


def check_mcq_format(raw: str) -> dict[str, Any]:
    """Validate that a response starts with a single A–D letter."""
    m = _LETTER_RE.match(raw.strip())
    if m:
        return dict(valid=True, letter=m.group(1).upper(), raw=raw.strip()[:60])
    return dict(valid=False, letter=None,
                raw=raw.strip()[:60],
                reason=f"No leading A-D letter in: {raw.strip()[:40]!r}")


# 4b — Free-form output sanity ──────────────────────────────────────────────

def check_freeform(
    text: str,
    *,
    min_chars: int = 10,
    max_chars: int = 10_000,
    forbidden: list[str] | None = None,
) -> dict[str, Any]:
    """
    Lightweight checks: non-empty, bounded length, no degenerate
    repetition, no forbidden substrings (e.g. leaked system tokens).
    """
    issues: list[str] = []
    clean = text.strip()
    n = len(clean)

    if n < min_chars:
        issues.append(f"too short ({n} < {min_chars})")
    if n > max_chars:
        issues.append(f"too long ({n} > {max_chars})")
    if clean and len(set(clean)) <= 2:
        issues.append(f"degenerate output: {clean[:30]!r}")
    for pat in (forbidden or []):
        if re.search(pat, clean, re.IGNORECASE):
            issues.append(f"forbidden pattern: {pat!r}")

    return dict(valid=not issues, chars=n, issues=issues)


# 4c — Dataset schema validator ─────────────────────────────────────────────

_REQUIRED_KEYS = {"question", "choices", "answer"}
_VALID_ANSWERS = {"A", "B", "C", "D"}


def check_dataset_schema(path: Path | None = None) -> dict[str, Any]:
    """
    Validate eval_runner/custom_task/dataset.json conforms to:
      [{"question": str, "choices": {"A":…,"B":…,"C":…,"D":…}, "answer": "A-D"}, …]
    """
    path = path or _DATASET
    errors: list[str] = []

    if not path.exists():
        return dict(valid=False, total=0, errors=[f"Not found: {path}"])

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return dict(valid=False, total=0, errors=[f"Bad JSON: {e}"])

    if not isinstance(data, list):
        return dict(valid=False, total=0, errors=["Root must be an array"])

    for i, item in enumerate(data):
        tag = f"[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{tag} not an object")
            continue
        missing = _REQUIRED_KEYS - item.keys()
        if missing:
            errors.append(f"{tag} missing {missing}")
        if "choices" in item:
            ch = item["choices"]
            if isinstance(ch, dict):
                if set(ch.keys()) != _VALID_ANSWERS:
                    errors.append(f"{tag} choices keys must be A/B/C/D")
            elif isinstance(ch, list):
                if len(ch) != 4:
                    errors.append(f"{tag} choices must have 4 entries")
            else:
                errors.append(f"{tag} choices must be dict or list")
        if "answer" in item and item["answer"] not in _VALID_ANSWERS:
            errors.append(f"{tag} answer must be A-D, got {item['answer']!r}")

    return dict(valid=not errors, total=len(data), errors=errors)


# 4d — Result file validation ──────────────────────────────────────────────

def check_result_files(results_dir: Path | None = None) -> dict[str, Any]:
    """Verify that result JSON files exist and contain a 'results' key."""
    d = results_dir or _RESULTS
    errors: list[str] = []
    checked = 0

    if not d.exists():
        return dict(valid=False, checked=0, errors=[f"Dir missing: {d}"])

    for f in sorted(d.glob("*.json")):
        if f.name.startswith("eval_cache"):
            continue
        checked += 1
        try:
            obj = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{f.name}: bad JSON ({e})")
            continue
        if "results" not in obj:
            errors.append(f"{f.name}: missing 'results' key")

    if checked == 0:
        errors.append("No result JSON files found")

    return dict(valid=not errors, checked=checked, errors=errors)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Live MCQ guardrail
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_mcq(
    url: str,
    cfg: GreedyConfig | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Send each custom benchmark question through the gateway in
    deterministic mode and validate the A–D letter format.
    """
    cfg = cfg or GreedyConfig(max_tokens=5)

    if not _DATASET.exists():
        return dict(total=0, ok=0, bad=0, error="dataset.json not found")

    items = json.loads(_DATASET.read_text())
    details: list[dict] = []
    ok_count = 0

    for item in items:
        q = item["question"]
        choices = item["choices"]
        if isinstance(choices, dict):
            opts = "\n".join(f"{k}) {v}" for k, v in choices.items())
        else:
            labels = "ABCD"
            opts = "\n".join(f"{labels[i]}) {c}" for i, c in enumerate(choices))

        prompt = (
            f"Question: {q}\n{opts}\n\n"
            "Reply with ONLY the answer letter (A, B, C, or D)."
        )

        raw = _call(url, prompt, cfg)
        result = check_mcq_format(raw)

        entry = dict(q=q[:50], expected=item["answer"], **result)
        if result["valid"]:
            entry["correct"] = result["letter"] == item["answer"]
            ok_count += 1
        details.append(entry)

    return dict(
        total=len(items), ok=ok_count, bad=len(items) - ok_count,
        details=details if verbose else [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _hdr(title: str) -> None:
    print(f"\n{'━' * 66}")
    print(f"  {title}")
    print(f"{'━' * 66}")


def _row(label: str, ok: bool, note: str = "") -> None:
    tag = "✓" if ok else "✗"
    print(f"  {tag}  {label}{('  ← ' + note) if note else ''}")


def cli() -> int:
    ap = argparse.ArgumentParser(description="Guardrail & determinism suite")
    ap.add_argument("--url", default="http://localhost:9000",
                    help="Gateway URL")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repetitions for reproducibility check")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="Skip live endpoint tests")
    args = ap.parse_args()

    cfg = GreedyConfig()
    ok_all = True

    # 1 — Dataset schema ──────────────────────────────────────────────────
    _hdr("1 · Dataset Schema")
    ds = check_dataset_schema()
    _row(f"dataset.json ({ds['total']} questions)", ds["valid"],
         "; ".join(ds["errors"][:3]) if ds["errors"] else "")
    ok_all &= ds["valid"]

    # 2 — Result files ────────────────────────────────────────────────────
    _hdr("2 · Evaluation Result Files")
    rf = check_result_files()
    _row(f"{rf['checked']} result files", rf["valid"],
         "; ".join(rf["errors"][:3]) if rf["errors"] else "")
    ok_all &= rf["valid"]

    if args.offline:
        print("\n  ⏭  Live tests skipped (--offline)")
    else:
        # Connectivity probe
        try:
            r = httpx.get(f"{args.url}/status", timeout=5)
            reachable = r.is_success
        except Exception:
            reachable = False

        if not reachable:
            print(f"\n  ⚠  Cannot reach {args.url}/status — skipping live tests.")
            print("     Start the gateway: python serve/serve.py")
        else:
            # 3 — Reproducibility ─────────────────────────────────────────
            _hdr("3 · Reproducibility (Identical Prompts → Identical Output)")
            print(f"  config: {asdict(cfg)}")
            print(f"  repeats: {args.repeats}\n")

            rep = check_reproducibility(args.url, cfg=cfg,
                                        repeats=args.repeats, verbose=args.verbose)
            for ck in rep["checks"]:
                note = ""
                if not ck["ok"]:
                    note = f"{ck['unique']} unique responses"
                    for i, t in enumerate(ck.get("responses", [])):
                        print(f"      run {i+1}: {t[:100]!r}")
                _row(f"Prompt: {ck['prompt']!r}", ck["ok"], note)
            _row("Overall reproducibility", rep["passed"])
            ok_all &= rep["passed"]

            # 4 — MCQ format (live) ───────────────────────────────────────
            _hdr("4 · MCQ Output Format (Live)")
            mcq = run_live_mcq(args.url, verbose=args.verbose)
            _row(f"Valid A–D: {mcq['ok']}/{mcq['total']}",
                 mcq["bad"] == 0,
                 f"{mcq['bad']} invalid" if mcq["bad"] else "")
            if args.verbose:
                for d in mcq.get("details", []):
                    tag = "✓" if d.get("valid") else "✗"
                    corr = ""
                    if d.get("valid"):
                        corr = " ✓" if d.get("correct") else " ✗"
                    print(f"      {tag} {d['q']}  → {d.get('letter','?')}"
                          f"  (expected {d['expected']}){corr}")
            ok_all &= mcq["bad"] == 0

            # 5 — Free-form sanity ────────────────────────────────────────
            _hdr("5 · Free-form Generation Sanity")
            for prompt in [
                "Explain what a neural network is in one paragraph.",
                "Write a Python function that returns the factorial of n.",
            ]:
                text = _call(args.url, prompt, GreedyConfig(max_tokens=256))
                ff = check_freeform(text, forbidden=[r"<\|system\|>", r"<\|endoftext\|>"])
                _row(f"{prompt[:50]!r}  ({ff['chars']} chars)", ff["valid"],
                     "; ".join(ff["issues"][:2]) if ff["issues"] else "")
                ok_all &= ff["valid"]

    # Summary ─────────────────────────────────────────────────────────────
    _hdr("Summary")
    if ok_all:
        print("  ✅  All guardrail checks passed.\n")
    else:
        print("  ❌  Some checks failed — review output above.\n")

    # Persist report
    report = _ROOT / "guardrails" / "report.json"
    report.write_text(json.dumps(dict(
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        url=args.url, config=asdict(cfg), passed=ok_all,
    ), indent=2))
    print(f"  Report → {report}\n")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(cli())
