#!/usr/bin/env bash
# sweep_eval4.sh — Evaluate checkpoints that were trained but never tested
# 3 evals:
#   1. topk,      ZN, 3 hypotheses, PLM       (trained in sweep2, never evaluated)
#   2. sparsemax, ZN, 5 hypotheses, eps=0.1   (trained, never evaluated)
#   3. softmax,   ZN, 1 hypothesis,  PLM      (manual run started 2026-09-15)
# Waits for any in-progress train.py to finish before starting.

set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG="sweep_eval4_$(date +%Y%m%d_%H%M%S).log"

run() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
    $PYTHON evaluate.py "$@"
}

{
while pgrep -f 'train\.py' > /dev/null; do
    echo "$(date '+%H:%M:%S')  waiting for in-progress train.py to finish..."
    sleep 60
done

run --checkpoint data/best_model_v9_zn_topk_plm_wta3.pt \
    --attn-mode topk --num-hypotheses 3 --metal-filter ZN --use-plm

run --checkpoint data/best_model_v9_zn_sparsemax_e0.1_wta5.pt \
    --attn-mode sparsemax --num-hypotheses 5 --metal-filter ZN

run --checkpoint data/best_model_v9_zn_softmax_plm.pt \
    --attn-mode softmax --num-hypotheses 1 --metal-filter ZN --use-plm

echo ""
echo "All evaluations complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
