#!/bin/bash

python run.py \
    --backend llama31-0 \
    --data_split train \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --rollout_width 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --save_path trajectories \
    --log logs/collect_trajectories_part-${j}.log \
    --max_depth 7 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --conv_template llama-3 \
    ${@}