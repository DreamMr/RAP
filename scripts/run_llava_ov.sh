#!/bin/bash

source /mnt/code/users/liamding/tools/conda_install/anaconda3/bin/activate vlm_llava_lmdeploy
export LMUData=/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark/LMUData
export llm_path=/mnt/data/huggingface/models/Qwen2/Qwen2-7B-Instruct
export TOKENIZERS_PARALLELISM=true
export OPENAI_API_KEY=sk-testmyllm
export OPENAI_API_BASE=http://0.0.0.0:23335/v1/chat/completions
export GPU=$(nvidia-smi --list-gpus | wc -l)
#export CUDA_VISIBLE_DEVICES=0,2,3

############# DEBUG
#GPU=3
######################


# can use: MMStar MMBench_DEV_EN_V11 MME POPE HallusionBench TextVQA_VAL ChartQA_TEST OCRBench AI2D_TEST MMMU_DEV_VAL vstar HRBench4K SEEDBench_IMG ScienceQA_VAL
WORKSPACE=/mnt/code/users/wangwenbin/RAP
cd $WORKSPACE


nohup lmdeploy serve api_server /mnt/data/huggingface/models/Qwen2/Qwen2-7B-Instruct --server-port 23335 --tp $GPU --log-level INFO --model-name chatgpt-0125 --cache-max-entry-count 0.1 --api-keys sk-testmyllm >lmdeploy_deploy.log &
sleep 20

MODEL_PATH=/mnt/data/users/wenbinwang/huggingface/llava-onevision-qwen2-0.5b-ov
WORKSPACE=/mnt/code/users/wangwenbin/LVLM/Evaluation/official/VLMEvalKit
MODEL_NAME=llava_onevision_qwen2_0.5b_ov
RAG_MODEL_PATH=/mnt/data/users/wenbinwang/huggingface/VisRAG-Ret
dataset=HRBench8K
processed_dataset=HRBench8K_single
work_dir=/mnt/data/users/wenbinwang/LVLM_results/llava_ov_ob5b_vrag_hrbench8k_benchmark_astar_rap/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH

torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH


source /usr/local/lib/miniconda3/etc/profile.d/conda.sh
conda deactivate 
bash /mnt/code/users/wangwenbin/ABSA/scripts/run_task.sh 4 5074