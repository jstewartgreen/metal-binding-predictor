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
    $PYTHON train.py --attn-mode "$1" --num-hypotheses "$2" --metal-filter ZN
}

run softmax   1
run softmax   3
run sparsemax 1
run sparsemax 3
run topk      1
run topk      3

echo ""
echo "All runs complete. Results in data/runs.csv"
