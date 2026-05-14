#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/sft_warmup.py \
    --config     configs/grpo_v5.json \
    --corpus     /home/sebastian/Documents/data_store/dataset/corpus/train \
    --steps      3000 \
    --out        /home/sebastian/Documents/data_store/ckpts/sft_v1 \
    --batch-size 8 \
    --grad-accum 4
