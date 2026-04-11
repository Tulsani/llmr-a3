#!/bin/bash
#SBATCH --job-name=at6646_grpo_lr
#SBATCH --output=./grpo_sweep_logs/%j_%x_%a.out
#SBATCH --error=./grpo_sweep_logs/%j_%x_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-3%2

set -e
mkdir -p logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# --- Learning rate sweep ---
LR_CONFIGS=(1e-6 1e-5 5e-5 1e-4)
LEARNING_RATE=${LR_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# --- Paths ---
MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_lr_sweep/grpo_model_lr${LEARNING_RATE}"

# --- Cache dirs ---
export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GRPO LR sweep..."
echo "  learning_rate: $LEARNING_RATE"
echo "  output_dir:    $OUTPUT_DIR"

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
    --learning-rate $LEARNING_RATE \
    --sampling-temperature 0.7 \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-every 10 \
    --n-eval-examples 200 \
    --gpu-memory-utilization 0.45

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"