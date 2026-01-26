#!/bin/bash

env_name=$1
stamp=$2
device=$3

# 定义 ckpt 列表
# ckpt_list=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
ckpt_list=(-1)

# 遍历列表
for ckpt in "${ckpt_list[@]}"; do
    echo "Running with checkpoint_num=${ckpt}"
    CUDA_VISIBLE_DEVICES=${device} XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    python learning/train_rsl_rl.py \
        --env_name=${env_name} \
        --vision=True \
        --num_envs=2048 \
        --play_only \
        --use_dr \
        --use_bg \
        --load_run_name=${env_name}-${stamp} \
        --checkpoint_num=${ckpt}
done
