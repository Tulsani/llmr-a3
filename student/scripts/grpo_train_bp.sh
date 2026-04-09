#!/bin/bash
#SBATCH --job-name=at6646_grpo
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue

set -e
mkdir -p logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
nvidia-smi

# --- Paths ---
MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_model"

# --- Cache dirs (avoid home quota) ---
export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- On-policy hyperparams (as per assignment defaults) ---
# Key constraint: train_batch_size == rollout_batch_size for on-policy
# gradient_accumulation_steps must divide train_batch_size
# micro_train_batch_size = train_batch_size / gradient_accumulation_steps = 16/8 = 2
ROLLOUT_BATCH_SIZE=16
TRAIN_BATCH_SIZE=16       # must equal rollout_batch_size for on-policy
GROUP_SIZE=8
GRAD_ACCUM_STEPS=8        # micro_batch = 16/8 = 2 (fits in A100 memory)
EPOCHS_PER_ROLLOUT=1      # on-policy = 1 epoch per rollout batch

LEARNING_RATE=1e-5
N_GRPO_STEPS=200
SAMPLING_TEMP=0.7
EVAL_EVERY=10
N_EVAL_EXAMPLES=200
GPU_MEM_UTIL=0.45         # split across 2 GPUs: vLLM on cuda:0, policy on cuda:1

echo "Starting GRPO training..."
echo "  model:       $MODEL"
echo "  train_path:  $TRAIN_PATH"
echo "  val_path:    $VAL_PATH"
echo "  output_dir:  $OUTPUT_DIR"

uv run python student/grpo_experiments.py \
    --model "$MODEL" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --policy-device cuda:1 \
    --vllm-device cuda:0 \
    --n-grpo-steps $N_GRPO_STEPS \
    --rollout-batch-size $ROLLOUT_BATCH_SIZE \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --group-size $GROUP_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM_STEPS \
    --epochs-per-rollout-batch $EPOCHS_PER_ROLLOUT \
    --learning-rate $LEARNING_RATE \
    --sampling-temperature $SAMPLING_TEMP \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-every $EVAL_EVERY \
    --n-eval-examples $N_EVAL_EXAMPLES \
    --gpu-memory-utilization $GPU_MEM_UTIL

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"