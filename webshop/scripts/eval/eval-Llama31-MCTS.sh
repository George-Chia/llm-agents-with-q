#!/bin/bash

python run.py \
    --backend llama31-0 \
    --data_split test \
    --n_generate_sample 5 \
    --temperature 1 \
    --iterations 30 \
    --log logs/eval_MCTS.log \
    --save_path trajectories \
    --max_depth 10 \
    --rollout_width 1 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 