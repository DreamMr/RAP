#!/bin/bash
export LMUData=YOUR_DATASET_PATH
export GPU=$(nvidia-smi --list-gpus | wc -l)

WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=liuhaotian/llava-v1.5-13b
MODEL_NAME=llava_v1.5_13b


work_dir=./outputs/vanilla/${MODEL_NAME}
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data HRBench4K HRBench8K vstar --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --no_rag

