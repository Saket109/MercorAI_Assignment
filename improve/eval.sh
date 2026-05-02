#!/usr/bin/env bash
# improve/eval.sh — MMLU STEM optimisation pipeline
#
#   bash improve/eval.sh
#   GATEWAY=http://localhost:9000 SAMPLES=20 bash improve/eval.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

GATEWAY="${GATEWAY:-http://localhost:9000}"
SAMPLES="${SAMPLES:-0}"          # 0 = full test set

echo "════════════════════════════════════════════════════════════"
echo "  MMLU STEM Optimisation Pipeline"
echo "  Gateway : $GATEWAY"
echo "  Samples : ${SAMPLES:-all}"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "1 · Installing dependencies …"
pip install -q datasets httpx 2>/dev/null || true

echo ""
echo "2 · Preparing few-shot data …"
python prepare_data.py

echo ""
echo "3 · Running ablation (4 configs) …"
python infer.py --url "$GATEWAY" --samples "$SAMPLES"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "  Artifacts:"
echo "    baseline_results.json"
echo "    optimized_results.json"
echo "    ablation_results.json"
echo "    report.md"
echo "════════════════════════════════════════════════════════════"
