#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG="sweep_eval3_$(date +%Y%m%d_%H%M%S).log"

run() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
    $PYTHON evaluate.py "$@"
}

{
run --checkpoint data/best_model_v9_bm_softmax_wta3.pt \
    --attn-mode softmax --num-hypotheses 3 --metal-filter bare --baseline

run --checkpoint data/best_model_v9_bm_sparsemax_wta3_v2.pt \
    --attn-mode sparsemax --num-hypotheses 3 --metal-filter bare --baseline

run --checkpoint data/best_model_v9_bm_topk_wta3.pt \
    --attn-mode topk --num-hypotheses 3 --metal-filter bare --baseline

run --checkpoint data/best_model_v9_zn_sparsemax_plm_m0_wta3.pt \
    --attn-mode sparsemax --num-hypotheses 3 --metal-filter ZN --use-plm --baseline

echo ""
echo "All evaluations complete."
} 2>&1 | tee "$LOG"
