model_name_or_path=PATH_TO_YOUR_LLM/phi-1_5
output_dir=checkpoints-Phi-1_5-StepLevelVerifier-iteration1
beta=0.1
lr=1e-6


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20002 fastchat/train/train_dpo.py \
     --model_name_or_path ${model_name_or_path} \
     --ref_model_name_or_path ${model_name_or_path} \
     --data_path webshop/webshop-Phi3-mcts_pm_data-round1.json \
     --bf16 True \
     --output_dir ${output_dir} \
     --num_train_epochs 1 \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 2 \
     --gradient_accumulation_steps 16 \
     --evaluation_strategy "no" \
     --save_strategy "epoch" \
     --save_total_limit 5 \
     --beta ${beta} \
     --learning_rate ${lr} \
     --weight_decay 0. \
     --warmup_ratio 0.1 \
     --lr_scheduler_type "constant_with_warmup" \
     --logging_steps 5 \
     --fsdp "full_shard auto_wrap" \
     --fsdp_transformer_layer_cls_to_wrap 'PhiDecoderLayer' \
     --tf32 True \
     --model_max_length 4096 \
     --gradient_checkpointing True \
     --lazy_preprocess False \
     --trust_remote_code True \
     --conv_template phi3 \