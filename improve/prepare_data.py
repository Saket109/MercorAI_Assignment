# improve/prepare_data.py
"""
Downloads MMLU STEM subjects from HuggingFace and prepares
a per-subject few-shot pool for dynamic exemplar selection.

    python improve/prepare_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

_DIR = Path(__file__).resolve().parent
_DATA = _DIR / "data"

# Five STEM subjects with enough test items to be statistically meaningful
SUBJECTS = [
    "college_physics",
    "machine_learning",
    "high_school_chemistry",
    "computer_security",
    "astronomy",
]

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def download_and_save() -> None:
    _DATA.mkdir(exist_ok=True)

    manifest: dict[str, dict] = {}

    for subj in SUBJECTS:
        print(f"  ↓  {subj}")
        ds = load_dataset("cais/mmlu", subj)

        # Dev split → few-shot pool (typically 5 items per subject)
        pool = []
        for row in ds["validation"]:
            pool.append(dict(
                question=row["question"],
                choices=row["choices"],
                answer=LABEL_MAP[row["answer"]],
                subject=subj,
            ))

        # Test split → evaluation items
        items = []
        for row in ds["test"]:
            items.append(dict(
                question=row["question"],
                choices=row["choices"],
                answer=LABEL_MAP[row["answer"]],
                subject=subj,
            ))

        outfile = _DATA / f"{subj}.json"
        payload = dict(subject=subj, fewshot_pool=pool, test=items)
        outfile.write_text(json.dumps(payload, indent=2))

        manifest[subj] = dict(
            fewshot=len(pool), test=len(items), path=str(outfile),
        )
        print(f"       {len(pool)} dev + {len(items)} test  →  {outfile.name}")

    (_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  ✓  Manifest saved to {_DATA / 'manifest.json'}")


if __name__ == "__main__":
    print("\n  MMLU STEM data preparation\n")
    download_and_save()
    print()
