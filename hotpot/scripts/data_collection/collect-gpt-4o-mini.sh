#!/bin/bash

num_workers=10


explore_model_name=gpt-4o-mini
exp_name=gpt-4o-mini

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
        --backend ${explore_model_name} \
        --data_split train \
        --part_num ${num_workers} \
        --part_idx ${part_idx} \
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
        --enable_fastchat_conv &
    echo $! >> logs/${exp_name}-eval_pid.txt
done