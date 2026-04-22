#!/bin/bash
#pip install -U transformers==4.57.6
#Noted: Using Qwen3VL should update the transformers package.  `pip install -U transformers==4.57.6`
export LMUData=YOUR_DATASET_PATH
export CUDA_VISIBLE_DEVICES=4,5
export GPU=2

WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
MODEL_NAME=Qwen3-VL-8B-Instruct
RAG_MODEL_PATH=openbmb/VisRAG-Ret

dataset=HRBench4K
processed_dataset=HRBench4K_single
work_dir=./outputs/rap/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 10
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH


dataset=HRBench8K
processed_dataset=HRBench8K_single
work_dir=./outputs/rap/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 10
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH

dataset=vstar
processed_dataset=vstar
work_dir=./outputs/rap/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 10
