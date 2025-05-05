#!/bin/bash
export LMUData=YOUR_DATASET_PATH
export GPU=$(nvidia-smi --list-gpus | wc -l)

WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=lmms-lab/llava-onevision-qwen2-0.5b-ov
MODEL_NAME=llava_onevision_qwen2_0.5b_ov
RAG_MODEL_PATH=openbmb/VisRAG-Ret

dataset=HRBench4K
processed_dataset=HRBench4K_single
work_dir=./outputs/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 200
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH


dataset=HRBench8K
processed_dataset=HRBench8K_single
work_dir=./outputs/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 200
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH