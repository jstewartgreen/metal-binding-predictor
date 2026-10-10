#!/usr/bin/env bash
# sweep3.sh — Single-hypothesis PLM runs, plus the missing K=3 PLM softmax run
# 6 runs, each trained then evaluated immediately so runs.csv gets a test row:
#   1. sparsemax, ZN,   1 hypothesis,  PLM
#   2. topk,      ZN,   1 hypothesis,  PLM
#   3. softmax,   ZN,   3 hypotheses,  PLM
#   4. sparsemax, bare, 1 hypothesis,  PLM
#   5. topk,      bare, 1 hypothesis,  PLM
#   6. softmax,   bare, 1 hypothesis,  PLM
# All runs: single-metal-only (default), 3 MPNN rounds, 25 epochs, k=16, batch=16, lr=1e-3
# Waits for any in-progress train.py to finish before starting.

set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG="sweep3_$(date +%Y%m%d_%H%M%S).log"

# run <attn_mode> <metal_filter> <num_hypotheses>
run() {
    local attn=$1 filter=$2 k=$3
    local ftag wtag=""
    if [ "$filter" = ZN ]; then ftag=zn; else ftag=bm; fi
    if [ "$k" -gt 1 ];       then wtag="_wta$k"; fi
    local ckpt="data/best_model_v9_${ftag}_${attn}_plm${wtag}.pt"

    echo ""
    echo "========================================"
    echo "  train: attn=$attn  filter=$filter  hypotheses=$k  plm"
    echo "========================================"
    $PYTHON train.py --attn-mode "$attn" --num-hypotheses "$k" --metal-filter "$filter" --use-plm

    echo ""
    echo "  eval: $ckpt"
    $PYTHON evaluate.py --checkpoint "$ckpt" --attn-mode "$attn" --num-hypotheses "$k" \
        --metal-filter "$filter" --use-plm
}

{
while pgrep -f 'train\.py' > /dev/null; do
    echo "$(date '+%H:%M:%S')  waiting for in-progress train.py to finish..."
    sleep 60
done

run sparsemax ZN   1
run topk      ZN   1
run softmax   ZN   3
run sparsemax bare 1
run topk      bare 1
run softmax   bare 1

echo ""
echo "All runs complete. Results in data/runs.csv"
} 2>&1 | tee "$LOG"
