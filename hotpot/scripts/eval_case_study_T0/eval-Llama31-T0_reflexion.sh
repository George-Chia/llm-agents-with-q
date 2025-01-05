#!/bin/bash
python run.py \
    --backend llama31-3 \
    --data_split test \
    --algorithm refine \
    --expansion_sampling_method memory \
    --refine_num 3 \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 0 \
    --iterations 1 \
    --save_path trajectories-T0-reflexion \
    --log logs/llama31.log \
    --max_depth 7 \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3