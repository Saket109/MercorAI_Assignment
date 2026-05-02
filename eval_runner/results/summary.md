# Evaluation Results

**Date:** 2026-05-02 19:57 UTC

| Task | Accuracy % | Acc Norm % | Exact Match % | Time (s) |
|------|-----------|-----------|--------------|----------|
| hellaswag | 35.0 | 45.0 | - | - |
| ml_reasoning | - | - | 86.67 | - |
| mmlu | 46.0 | - | - | 0 |

## Notes
- MMLU: 0-shot, 5 STEM subjects (10 items each), accuracy via direct generation
- HellaSwag: 0-shot, loglikelihood-based scoring (20 examples)
- ml_reasoning: 0-shot, exact letter match on 15 custom ML/CS questions
- All results cached via SHA-256 DiskMemo for deterministic re-runs