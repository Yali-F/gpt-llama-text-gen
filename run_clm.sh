#!/bin/bash
#SBATCH --job-name=gpt2-yl          
#SBATCH --output=logs/gpt2-yl-%j.out          
#SBATCH --nodes=1                 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20   
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4

source /home/fuyali/miniconda3/bin/activate gpt2
cd /home/fuyali/gpt2_test
# export TRANSFORMERS_OFFLINE=1
# export DATASETS_OFFLINE=1

#export CUDA_VISIBLE_DEVICES=1,2
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --overwrite_output_dir True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --logging_strategy steps \
    --logging_step 1 \
    --output_dir ./checkpoints \

    
    # --num_train_epochs 1 \     

# python run_generation.py \
#       --model_type=gpt2 \
#       --model_name_or_path=./checkpoints