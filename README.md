# LLM Evaluation Pipeline

End-to-end infrastructure for serving, evaluating, stress-testing, and improving a large language model (**Mistral-7B-Instruct v0.1**) via vLLM on RunPod.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Local Machine                                           │
│                                                          │
│  serve/serve.py (:9000)  ──→  RunPod vLLM (:8888)       │
│       ↑                        Mistral 7B                │
│       │                                                  │
│  ┌────┴─────┐  ┌──────────┐  ┌────────────┐             │
│  │eval_runner│  │  perf/   │  │ guardrails/│             │
│  │ (Part B)  │  │ (Part C) │  │ (Part D)   │             │
│  └──────────┘  └──────────┘  └────────────┘             │
│                                                          │
│  ┌──────────┐                                            │
│  │ improve/ │                                            │
│  │ (Part E) │                                            │
│  └──────────┘                                            │
└──────────────────────────────────────────────────────────┘
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure endpoint (edit .env)
VLLM_BASE=https://YOUR_RUNPOD_URL
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
```

## Part A — Serving Gateway

An async FastAPI proxy that sits between clients and the remote vLLM backend. Provides connection pooling, auto-retry, streaming (SSE), and batch inference.

```bash
python3 serve/serve.py                      # Start gateway on :9000
python3 serve/sample_run.py                 # Run 4 demo scenarios
python3 serve/benchmark_concurrent.py       # Concurrent stress test
```

| File | Purpose |
|------|---------|
| `serve/config.py` | Centralized `AppConfig` (reads `.env`) |
| `serve/serve.py` | FastAPI gateway with `BackendPool` |
| `serve/client.py` | Python SDK (`InferenceSession`) |
| `serve/sample_run.py` | Batch, stream, fan-out, determinism demos |
| `serve/benchmark_concurrent.py` | Latency measurement under concurrent load |

## Part B — Evaluation Harness

Custom `lm-evaluation-harness` integration that evaluates the model on standardized benchmarks via the vLLM endpoint.

```bash
python3 eval_runner/run_eval.py --endpoint $VLLM_URL --tasks ml_reasoning
python3 eval_runner/run_eval.py --endpoint $VLLM_URL --tasks hellaswag --limit 20
python3 eval_runner/run_eval.py --endpoint $VLLM_URL --tasks mmlu --limit 10
```

| File | Purpose |
|------|---------|
| `eval_runner/vllm_model.py` | `ChatEndpointLM` wrapper (implements `lm-eval` interface) |
| `eval_runner/run_eval.py` | CLI orchestrator with `EvalPipeline` |
| `eval_runner/cache.py` | `DiskMemo` — SHA-256 keyed prompt cache |
| `eval_runner/custom_task/` | 15-question ML/AI benchmark (dataset + config) |
| `eval_runner/results/` | JSON outputs + `summary.md` |

**Benchmarks evaluated:**
- **MMLU** — 57-subject knowledge exam (5-shot)
- **HellaSwag** — Commonsense reasoning (0-shot)
- **ml_reasoning** — Custom 15-question ML/CS quiz (0-shot)

## Part C — Performance & Scaling

Concurrent load generator that measures latency, throughput, and TTFT across multiple configurations.

```bash
python3 serve/serve.py                                       # Gateway must be running
python3 perf/load_test.py --concurrency 1,2,4 --rounds 2    # Full sweep
```

| File | Purpose |
|------|---------|
| `perf/load_test.py` | `httpx`-based concurrent load generator |
| `perf/metrics.csv` | Per-request raw data |
| `perf/metrics_aggregate.csv` | Per-scenario aggregates |
| `perf/analysis.ipynb` | Jupyter notebook with 9 visualization sections |

**Metrics collected:** TTFT, TPOT, P50/P95/P99 latency, GPU utilization

## Part D — Guardrails & Determinism

Validates deterministic decoding, output format correctness, and dataset schema integrity.

```bash
python3 serve/serve.py                          # Gateway must be running
python3 guardrails/validate.py --verbose        # Full suite (5 checks)
python3 guardrails/validate.py --offline        # Schema-only
```

| File | Purpose |
|------|---------|
| `guardrails/validate.py` | 5-check validation suite (schema, results, reproducibility, MCQ format, free-form) |
| `guardrails/README.md` | What was tested + where nondeterminism persists |
| `guardrails/report.json` | Machine-readable last-run report |

## Part E — Benchmark Improvement

Inference-time optimization on MMLU STEM subjects. No finetuning.

```bash
python3 improve/prepare_data.py                 # Download MMLU STEM data
python3 improve/infer.py --url http://localhost:9000   # Run 4-config ablation
# or:
bash improve/eval.sh                            # Full pipeline
```

| File | Purpose |
|------|---------|
| `improve/prepare_data.py` | Downloads 5 MMLU STEM subjects from HuggingFace |
| `improve/optimize_prompt.py` | Baseline + optimized prompt builders |
| `improve/infer.py` | Ablation inference with bootstrap CI + permutation test |
| `improve/eval.sh` | End-to-end pipeline script |
| `improve/report.md` | Results, ablation study, 10+ examples, reproducibility settings |

**Result:** +32% accuracy lift (24% → 56%) via expert-persona instruction rewriting, p = 0.0024.

## Makefile Targets

```bash
make serve              # Start gateway
make eval               # Run all three benchmarks
make eval-quick         # ml_reasoning + hellaswag (limited)
make eval-custom        # ml_reasoning only
make load-test          # Full performance sweep
make load-test-quick    # Quick 2-concurrency test
make guardrails         # Full guardrail suite
make guardrails-offline # Schema validation only
```

## Environment

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Engine:** vLLM on RunPod (GPU)
- **Python:** 3.12+
- **Key deps:** `httpx`, `fastapi`, `lm-eval`, `datasets`
