#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python run.py \
    --backend gpt-4-turbo-2024-04-09 \
    --data_split test \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories_Phi3Qepoch2 \
    --log logs/gpt-4-turbo-beam_epoch1.log \
    --max_depth 7 \
    --algorithm beam \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template phi3 \
    --policy_model_name_or_path checkpoints-hotpot-Phi-1_5-StepLevelVerifier-Phi3-iteration1-0.1beta/epoch2-chosen \
    --reference_model_name_or_path /home/zhaiyuanzhao/llm/phi-1_5 \
    ${@}