#!/bin/bash

python run.py \
    --backend Phi-3-0 \
    --data_split test \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories_epoch1 \
    --log logs/Phi-3-beam_epoch1.log \
    --max_depth 7 \
    --algorithm beam \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template phi3 \
    --policy_model_name_or_path checkpoints-hotpot-Phi-1_5-StepLevelVerifier-Phi3-iteration1/epoch1 \
    --reference_model_name_or_path /home/zhaiyuanzhao/llm/phi-1_5 \
    ${@}