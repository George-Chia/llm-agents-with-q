#!/bin/bash

num_workers=7
node_num=9

explore_model_name=llama31
exp_name=llama31-exploration


for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
    --backend ${explore_model_name}-${j} \
    --data_split test \
    --part_num ${num_workers} \
    --part_idx ${part_idx} \
    --n_generate_sample 3 \
    --temperature 1 \
    --iterations 20 \
    --log logs/eval_part-${j}.log \
    --save_path trajectories-MCTS-KTO-critique-rwc \
    --max_depth 10 \
    --rollout_width 1 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 \
    --expansion_sampling_method critique \
    --critique_backend Llama31-KTO-webshop \
    --critique_prompt_template template_v1 \
    --enable_rollout_with_critique &
    echo $! >> logs/${exp_name}-eval_pid.txt
done