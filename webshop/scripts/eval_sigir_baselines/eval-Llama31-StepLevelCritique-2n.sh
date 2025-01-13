#!/bin/bash

python run.py \
    --backend webshop-policy-llama31 \
    --data_split test \
    --n_generate_sample 2 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1  \
    --iterations 1 \
    --save_path trajectories-StepLevelCritique-2n \
    --log logs/eval-llama31.log \
    --max_depth 10 \
    --algorithm simple \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 \
    --expansion_sampling_method critique \
    --critique_backend webshop-critique-llama31 \
    --critique_prompt_template template_v1 