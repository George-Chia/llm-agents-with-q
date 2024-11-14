#!/bin/bash

python run.py \
    --backend llama31-2 \
    --data_split test \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1 \
    --iterations 30 \
    --save_path trajectories-MCTS \
    --log logs/llama31.log \
    --max_depth 7 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 