# improve/optimize_prompt.py
"""
Prompt templates for MMLU STEM improvement.

Provides baseline (bare minimum) and optimized (expert persona +
subject-matched few-shot + format-locked output) builders, plus
a robust multi-pattern answer normalizer.

    python improve/optimize_prompt.py   # quick self-test
"""

from __future__ import annotations

import re

# ── Human-readable subject names ────────────────────────────────────────────
_PRETTY = {
    "college_physics": "College Physics",
    "machine_learning": "Machine Learning",
    "high_school_chemistry": "High-School Chemistry",
    "computer_security": "Computer Security",
    "astronomy": "Astronomy",
}

LABELS = ("A", "B", "C", "D")


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline: minimal 0-shot prompt
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_prompt(question: str, choices: list[str]) -> str:
    """
    Bare-bones prompt: no instructions, no examples, just the question
    and lettered choices followed by 'Answer:'.
    """
    opts = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}\nAnswer:"


# ═══════════════════════════════════════════════════════════════════════════════
#  Optimized: expert instruction + subject-matched few-shot
# ═══════════════════════════════════════════════════════════════════════════════

def optimized_prompt(
    question: str,
    choices: list[str],
    subject: str,
    exemplars: list[dict],
    n_shots: int = 5,
) -> str:
    """
    Improved prompt with:
      1. Expert persona tied to the subject
      2. Explicit format instruction
      3. Subject-matched few-shot demonstrations
      4. Format-locked output anchor ("The answer is:")
    """
    pretty = _PRETTY.get(subject, subject.replace("_", " ").title())
    lines: list[str] = []

    # Instruction header
    lines.append(
        f"You are a world-class expert in {pretty}. "
        "For each multiple-choice question below, select the single best "
        "answer and reply with ONLY the corresponding letter (A, B, C, or D). "
        "Do not include explanations.\n"
    )

    # Few-shot exemplars (subject-matched)
    for ex in exemplars[:n_shots]:
        opts = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(ex["choices"]))
        lines.append(f"Question: {ex['question']}\n{opts}\nThe answer is: {ex['answer']}\n")

    # Target question
    opts = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(choices))
    lines.append(f"Question: {question}\n{opts}\nThe answer is:")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Instruction-only (no few-shot) — for ablation
# ═══════════════════════════════════════════════════════════════════════════════

def instruction_only_prompt(question: str, choices: list[str], subject: str) -> str:
    """Optimized instruction header but no few-shot examples."""
    pretty = _PRETTY.get(subject, subject.replace("_", " ").title())
    opts = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"You are a world-class expert in {pretty}. "
        "Select the single best answer and reply with ONLY the letter "
        "(A, B, C, or D).\n\n"
        f"Question: {question}\n{opts}\nThe answer is:"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Answer normalization
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered from most specific to least specific
_PATTERNS = [
    re.compile(r"(?:the\s+answer\s+is|answer)\s*:\s*([A-Da-d])", re.I),
    re.compile(r"^\s*\(?([A-Da-d])\)?[\s.)]*", re.MULTILINE),
    re.compile(r"\b([A-Da-d])\s*$"),
]


def normalize_answer(raw: str) -> str | None:
    """
    Extract a single A-D letter from model output using cascading regex.
    Returns uppercase letter or None.
    """
    for pat in _PATTERNS:
        m = pat.search(raw.strip())
        if m:
            return m.group(1).upper()
    return None


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = "What is the SI unit of force?"
    ch = ["Joule", "Newton", "Watt", "Pascal"]

    print("=== BASELINE ===")
    print(baseline_prompt(q, ch))

    print("\n=== INSTRUCTION ONLY ===")
    print(instruction_only_prompt(q, ch, "college_physics"))

    pool = [{"question": "Speed of light?", "choices": ["3e8 m/s", "343 m/s", "1500 m/s", "0 m/s"], "answer": "A"}]
    print("\n=== OPTIMIZED (1-shot) ===")
    print(optimized_prompt(q, ch, "college_physics", pool, n_shots=1))

    # Test normalizer
    for sample in ["B", " (C) ", "The answer is: D", "A. Newton", "blah blah\nB"]:
        print(f"  normalize({sample!r:30s}) → {normalize_answer(sample)}")
