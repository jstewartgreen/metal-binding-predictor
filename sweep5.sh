#!/usr/bin/env bash
# sweep5.sh — Centroid-centered coords, single hypothesis, ZN, no PLM
# 3 runs, each trained then evaluated immediately so runs.csv gets a test row:
#   1. softmax,   ZN, 1 hypothesis, --coord-frame centroid
#   2. sparsemax, ZN, 1 hypothesis, --coord-frame centroid
#   3. topk,      ZN, 1 hypothesis, --coord-frame centroid
# Compare against the absolute rows (zn_{attn}.pt) and per-residue rows (zn_{attn}_rel.pt).
# All runs: single-metal-only (default), 3 MPNN rounds, 25 epochs, batch=16, lr=1e-3
# Waits for any other sweep or train.py to finish before starting (they share runs.csv).

set -euo pipefail
export PYTHONUNBUFFERED=1   # flush epoch lines through tee as they happen

PYTHON="${PYTHON:-python}"
LOG="sweep5_$(date +%Y%m%d_%H%M%S).log"

# run <attn_mode>
run() {
    local attn=$1
    local ckpt="data/best_model_v9_zn_${attn}_cen.pt"

    echo ""
    echo "========================================"
    echo "  train: attn=$attn  filter=ZN  hypotheses=1  coord-frame=centroid"
    echo "========================================"
    $PYTHON train.py --attn-mode "$attn" --num-hypotheses 1 --metal-filter ZN --coord-frame centroid

    echo ""
    echo "  eval: $ckpt"
    $PYTHON evaluate.py --checkpoint "$ckpt" --attn-mode "$attn" --num-hypotheses 1 \
        --metal-filter ZN --coord-frame centroid
}

{
# Wait on running Python workers only; matching other sweep scripts by name would match this one too.
while pgrep -f '[Pp]ython.*(train|evaluate)\.py' > /dev/null; do
    echo "$(date '+%H:%M:%S')  waiting for an in-progress train.py / evaluate.py to finish..."
    sleep 60
done

run softmax
run sparsemax
run topk

echo ""
echo "All runs complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
