#!/bin/bash

num_workers=3
node_num=9

model_path=/home/zhaiyuanzhao/LLM-Agents-with-Q/hotpot/trained_models/template_huan/action-round1-checkpoint-51/merged_model
explore_model_name=llama31-hotpot
exp_name=llama31-hotpot-collect-MCTS

fs_worker_port=21011
worker_idx=0


for ((j=0;j<${num_workers};j=j+1)); do
     echo "Launch the model worker on port ${fs_worker_port}"
     CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${node_num})) python -u -m fastchat.serve.vllm_worker \
          --model-path $model_path \
          --port ${fs_worker_port} \
          --model-names ${explore_model_name}-${j} \
          --worker-address http://0.0.0.0:${fs_worker_port} >> logs/${exp_name}-model_worker-${j}.log 2>&1 &
     echo $! >> logs/${exp_name}-worker_pid.txt
     fs_worker_port=$(($fs_worker_port+1))
     worker_idx=$(($worker_idx+1))
     sleep 15
done