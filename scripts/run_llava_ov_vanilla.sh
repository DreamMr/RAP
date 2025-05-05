#!/bin/bash
export LMUData=YOUR_DATASET_PATH
export GPU=$(nvidia-smi --list-gpus | wc -l)

WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=lmms-lab/llava-onevision-qwen2-0.5b-ov
MODEL_NAME=llava_onevision_qwen2_0.5b_ov
RAG_MODEL_PATH=openbmb/VisRAG-Ret

dataset=HRBench4K
work_dir=./outputs/${MODEL_NAME}
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --no_rag

dataset=HRBench8K
work_dir=./outputs/${MODEL_NAME}
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --no_rag

dataset=vstar
work_dir=./outputs/${MODEL_NAME}
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --no_rag
