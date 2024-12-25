#!/bin/bash
num_workers=3


explore_model_name=llama31-hotpot
exp_name=llama31-policy-iteration1-collection-disable_early_stop

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
        --backend ${explore_model_name}-${j} \
        --data_split train \
        --part_num ${num_workers} \
        --part_idx ${part_idx} \
        --n_generate_sample 3 \
        --n_evaluate_sample 1 \
        --rollout_width 1 \
        --prompt_sample cot \
        --temperature 1.0 \
        --iterations 20 \
        --save_path trajectories-MCTS-n3-critique-disable_early_stop \
        --log logs/collect_trajectories_part-${j}.log \
        --max_depth 7 \
        --algorithm mcts \
        --enable_fastchat_conv \
        --conv_template llama-3 \
        --disable_early_stop \
        --expansion_sampling_method critique \
        --critique_backend ${explore_model_name}-${j} \
        --critique_prompt_template template_v1 &
    echo $! >> logs/${exp_name}-collection_pid.txt
done