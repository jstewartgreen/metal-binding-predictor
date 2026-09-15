#!/usr/bin/env bash
# sweep2.sh — Second sweep: sparsemax variants
# 4 runs:
#   1. sparsemax, ZN, 5 hypotheses
#   2. sparsemax, ZN, 3 hypotheses + PLM
#   3. sparsemax, ZN, 3 hypotheses, k=32
#   4. sparsemax, bare-metal filter, 3 hypotheses

set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG="sweep2_$(date +%Y%m%d_%H%M%S).log"

run() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
    $PYTHON train.py "$@"
}

{
run --attn-mode sparsemax --num-hypotheses 5 --metal-filter ZN

run --attn-mode sparsemax --num-hypotheses 3 --metal-filter ZN --use-plm

run --attn-mode sparsemax --num-hypotheses 3 --metal-filter ZN --k-neighbors 32 \
    --checkpoint data/best_model_v9_zn_sparsemax_k32_wta3.pt

run --attn-mode sparsemax --num-hypotheses 3 --metal-filter bare \
    --checkpoint data/best_model_v9_bm_sparsemax_wta3_v2.pt

echo ""
echo "All runs complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
