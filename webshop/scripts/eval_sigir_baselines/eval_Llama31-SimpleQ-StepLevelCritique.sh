#!/bin/bash

python run.py \
    --backend webshop-policy-llama31 \
    --algorithm simple \
    --data_split test \
    --n_generate_sample 3 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1 \
    --iterations 1 \
    --log logs/log-george.log \
    --save_path trajectories-SimpleQ-StepLevelCritique-checkpoint52 \
    --max_depth 10 \
    --rollout_width 1 \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 \
    --expansion_sampling_method critique \
    --critique_backend webshop-critique-llama31 \
    --critique_prompt_template template_v1 \
    --policy_model_name_or_path /home/zhaiyuanzhao/LLM-Agents-with-Q/checkpoints-Phi-1_5-StepLevelVerifier-iteration1/checkpoint-52 \
    --reference_model_name_or_path /home/zhaiyuanzhao/llm/phi-1_5 \
    --q_model_conv_template phi3 \
    --enable_Q_value_model_for_critique \