#!/usr/bin/env bash
# sweep4.sh — Residue-relative coords, single hypothesis, ZN, no PLM
# 2 runs, each trained then evaluated immediately so runs.csv gets a test row:
#   1. softmax,   ZN, 1 hypothesis, --relative-coords
#   2. sparsemax, ZN, 1 hypothesis, --relative-coords
# (topk was run manually: data/best_model_v9_zn_topk_rel.pt)
# All runs: single-metal-only (default), 3 MPNN rounds, 25 epochs, batch=16, lr=1e-3
# Waits for any in-progress train.py to finish before starting.

set -euo pipefail
export PYTHONUNBUFFERED=1   # flush epoch lines through tee as they happen

PYTHON="${PYTHON:-python}"
LOG="sweep4_$(date +%Y%m%d_%H%M%S).log"

# run <attn_mode>
run() {
    local attn=$1
    local ckpt="data/best_model_v9_zn_${attn}_rel.pt"

    echo ""
    echo "========================================"
    echo "  train: attn=$attn  filter=ZN  hypotheses=1  relative-coords"
    echo "========================================"
    $PYTHON train.py --attn-mode "$attn" --num-hypotheses 1 --metal-filter ZN --relative-coords

    echo ""
    echo "  eval: $ckpt"
    $PYTHON evaluate.py --checkpoint "$ckpt" --attn-mode "$attn" --num-hypotheses 1 \
        --metal-filter ZN --relative-coords
}

{
while pgrep -f 'train\.py' > /dev/null; do
    echo "$(date '+%H:%M:%S')  waiting for in-progress train.py to finish..."
    sleep 60
done

run softmax
run sparsemax

echo ""
echo "All runs complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
