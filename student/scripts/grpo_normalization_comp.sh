#!/bin/bash
#SBATCH --job-name=at6646_grpo_masked_normalize
#SBATCH --output=./grpo_norm_logs/%j_%x.out
#SBATCH --error=./grpo_norm_logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue

set -e
mkdir -p grpo_norm_logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
nvidia-smi

MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_norm/grpo_model_masked_normalize"

export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GRPO masked_normalize run..."

uv run python /gpfs/scratch/an4462/at6646/llmr-a3/student/grpo_experiment_v2.py \
    --model "$MODEL" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --policy-device cuda:1 \
    --vllm-device cuda:0 \
    --n-grpo-steps 200 \
    --rollout-batch-size 16 \
    --train-batch-size 16 \
    --group-size 8 \
    --gradient-accumulation-steps 8 \
    --epochs-per-rollout-batch 1 \
    --learning-rate 5e-5 \
    --sampling-temperature 0.7 \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type reinforce_with_baseline \
    --norm-type masked_normalize \
    --use-std-normalization \
    --eval-every 10 \
    --n-eval-examples 200 \
    --gpu-memory-utilization 0.45

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"