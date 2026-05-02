.PHONY: install serve status sample bench eval load-test

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Part A: Serving ──────────────────────────────────────────────────────────
serve:
	cd serve && python serve.py

status:
	@curl -s http://localhost:9000/status | python3 -m json.tool

sample:
	cd serve && python sample_run.py

bench:
	cd serve && python benchmark_concurrent.py --workers 5 --rounds 2 --csv ../results/concurrency.csv

# ── Part B: Evaluation ───────────────────────────────────────────────────────
eval:
	python eval_runner/run_eval.py

eval-quick:
	python eval_runner/run_eval.py --limit 10

eval-custom:
	python eval_runner/run_eval.py --tasks ml_reasoning

# ── Part C: Performance ──────────────────────────────────────────────────────
load-test:
	python perf/load_test.py

load-test-quick:
	python perf/load_test.py --concurrency 1,2 --rounds 1

# ── Part D: Guardrails ───────────────────────────────────────────────────────
guardrails:
	python guardrails/validate.py --verbose

guardrails-offline:
	python guardrails/validate.py --offline
