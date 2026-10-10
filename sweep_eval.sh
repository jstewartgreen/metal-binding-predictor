#!/usr/bin/env bash
# sweep.sh — Run attention mode × hypothesis count grid
# 6 runs: {softmax, sparsemax, topk} × {1, 3} hypotheses
# All runs: single-metal-only (default), ZN filter, 25 epochs, k=16, batch=16, lr=1e-3

set -euo pipefail

PYTHON="${PYTHON:-python}"

run() {
    echo ""
    echo "========================================"
    echo "  attn=$1  hypotheses=$2"
    echo "========================================"
    $PYTHON evaluate.py --checkpoint data/best_model_v9_zn_"$1""$3" --attn-mode "$1" --num-hypotheses "$2" --metal-filter ZN
}

run softmax   1 .pt
run softmax   3 _wta3.pt
run sparsemax 1 .pt
run sparsemax 3 _wta3.pt
run topk      1 .pt
run topk      3 _wta3.pt

echo ""
echo "All runs complete. Results in data/runs.csv"
