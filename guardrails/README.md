# Guardrails & Determinism

Validation suite that enforces deterministic generation and output correctness
across the evaluation pipeline.

## What We Test

### 1. Reproducibility (Idempotency)

We lock decoding to **greedy mode** (`temperature=0`, `top_p=1`, `seed=42`)
and fire the same prompt multiple times through the gateway.  Every response
is SHA-256 hashed; all hashes must match.

| Parameter     | Value | Why                                        |
|---------------|-------|--------------------------------------------|
| `temperature` | 0.0   | Disables sampling → pure argmax            |
| `top_p`       | 1.0   | No nucleus truncation                      |
| `seed`        | 42    | vLLM server-side RNG seed for tie-breaking |
| `max_tokens`  | 256   | Bounded output length                      |

### 2. Dataset Schema Validation

Verifies `eval_runner/custom_task/dataset.json` conforms to the expected
schema: an array of objects each containing `question`, `choices` (dict or
list of 4), and `answer` (A–D).

### 3. Evaluation Result Files

Checks that `eval_runner/results/*.json` files are valid JSON and contain
the `results` key produced by `lm-evaluation-harness`.

### 4. MCQ Output Format (Live)

Sends each custom benchmark question through the gateway in deterministic
mode and validates that the response begins with a single letter A–D.
The regex `^\s*([A-Da-d])\b` is permissive enough to accept common model
patterns like `"C"`, `"C)"`, `"C) Paris"`, `"A. The answer"`.

### 5. Free-form Generation Sanity

Validates that longer completions are non-empty, bounded in length, not
degenerate (single repeated character), and don't leak system tokens like
`<|system|>` or `<|endoftext|>`.

## Where Nondeterminism Can Persist

Even with greedy decoding + a fixed seed, bit-exact reproducibility is
**not guaranteed** in all environments:

| Source | Mechanism |
|--------|-----------|
| **FlashAttention reduction order** | Batching different request counts changes the warp-level reduction, introducing ~1e-7 float differences that occasionally flip an argmax at token boundaries |
| **Tensor parallelism** | Different shard counts change all-reduce ordering |
| **Prefix caching** | Whether a KV prefix is cached or recomputed can shift intermediate activations |
| **Hardware / driver** | Different GPU models or CUDA versions may use different fused kernels |

In practice, **synchronous single-request greedy decoding is fully
deterministic** on our current setup.  Nondeterminism only appears under
concurrent load where batch composition varies between runs.

## Usage

```bash
# Full suite (gateway must be running on :9000)
python guardrails/validate.py --verbose

# Offline checks only (schema + result files)
python guardrails/validate.py --offline

# Custom endpoint and more repeats
python guardrails/validate.py --url http://localhost:9000 --repeats 5
```

Output is printed to stdout and a JSON report is saved to
`guardrails/report.json`.
