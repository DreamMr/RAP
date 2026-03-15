#!/bin/bash
#Noted: Using Qwen3VL should update the transformers package.  `pip install -U transformers==4.57.6`
export LMUData=YOUR_DATASET_PATH
export GPU=$(nvidia-smi --list-gpus | wc -l)


WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
MODEL_NAME=Qwen3-VL-8B-Instruct


work_dir=./outputs/vanilla/${MODEL_NAME}
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data HRBench4K HRBench8K vstar --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --no_rag

