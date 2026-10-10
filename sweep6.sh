#!/usr/bin/env bash
# sweep6.sh — MPNN-ablated (ESM-2 only), single hypothesis, ZN
# 3 runs, each trained then evaluated immediately so runs.csv gets a test row:
#   1. sparsemax, ZN, 1 hypothesis, --use-plm --num-mpnn-rounds 0
#   2. topk,      ZN, 1 hypothesis, --use-plm --num-mpnn-rounds 0
#   3. softmax,   ZN, 1 hypothesis, --use-plm --num-mpnn-rounds 0
# With no message passing the node features are unused, so the coordinate frame is
# irrelevant (absolute default). Compare against zn_{attn}_plm.pt (ESM-2 + 3 MPNN rounds).
# All runs: single-metal-only (default), 25 epochs, batch=16, lr=1e-3
# Waits for any in-progress train.py / evaluate.py to finish before starting.

set -euo pipefail
export PYTHONUNBUFFERED=1   # flush epoch lines through tee as they happen

PYTHON="${PYTHON:-python}"
LOG="sweep6_$(date +%Y%m%d_%H%M%S).log"

# run <attn_mode>
run() {
    local attn=$1
    local ckpt="data/best_model_v9_zn_${attn}_plm_m0.pt"

    echo ""
    echo "========================================"
    echo "  train: attn=$attn  filter=ZN  hypotheses=1  plm  mpnn-rounds=0"
    echo "========================================"
    $PYTHON train.py --attn-mode "$attn" --num-hypotheses 1 --metal-filter ZN --use-plm --num-mpnn-rounds 0

    echo ""
    echo "  eval: $ckpt"
    $PYTHON evaluate.py --checkpoint "$ckpt" --attn-mode "$attn" --num-hypotheses 1 \
        --metal-filter ZN --use-plm
}

{
# Wait on running Python workers only; matching other sweep scripts by name would match this one too.
while pgrep -f '[Pp]ython.*(train|evaluate)\.py' > /dev/null; do
    echo "$(date '+%H:%M:%S')  waiting for an in-progress train.py / evaluate.py to finish..."
    sleep 60
done

run sparsemax
run topk
run softmax

echo ""
echo "All runs complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
