#!/bin/bash

backend=$1
temperature=$2

python run.py \
    --backend $backend \
    --data_split test \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature $temperature  \
    --iterations 1 \
    --save_path trajectories \
    --log logs/eval-llama31.log \
    --max_depth 10 \
    --algorithm simple \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 