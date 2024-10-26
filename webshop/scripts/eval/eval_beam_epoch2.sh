CUDA_VISIBLE_DEVICES=5 python run.py \
    --backend Phi-3-2 \
    --data_split test \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories_epoch2 \
    --log logs/eval-Phi-3-beam_epoch2.log \
    --max_depth 10 \
    --algorithm beam \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template phi3 \
    --policy_model_name_or_path checkpoints-Phi-1_5-StepLevelVerifier-iteration1/epoch2 \
    --reference_model_name_or_path /home/zhaiyuanzhao/llm/phi-1_5 \
    ${@}