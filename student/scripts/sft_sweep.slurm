#!/bin/bash
#SBATCH --job-name=at6646_sft
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --array=0-4%2

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# Dataset sizes — index 4 = full dataset
MAX_EXAMPLES_CONFIGS=(128 256 512 1024 "None")
MAX_EXAMPLES=${MAX_EXAMPLES_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Hyperparameters
# Epochs scaled by dataset size — smaller datasets need more epochs
# to achieve ~40% loss decrease
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
GRAD_CLIP=1.0
SEED=42
GPU_MEMORY_UTILIZATION=0.4

# Epochs per dataset size: more epochs for smaller datasets
EPOCHS_CONFIGS=(20 15 10 5 2)
NUM_EPOCHS=${EPOCHS_CONFIGS[$SLURM_ARRAY_TASK_ID]}

EVAL_POINTS=10

# Paths
MODEL="Qwen/Qwen2.5-Math-1.5B"
TRAIN_PATH="data/intellect_math_train_dev_test/train"

# Resolve run config
if [ "$MAX_EXAMPLES" = "None" ]; then
    RUN_NAME="sft-full"
    OUTPUT_DIR="/scratch/at6646/sft_checkpoints/full"
    MAX_EXAMPLES_ARG=""
    N_EXAMPLES=$(uv run python -c \
        "from datasets import load_from_disk; print(len(load_from_disk('$TRAIN_PATH')))")
else
    RUN_NAME="sft-${MAX_EXAMPLES}"
    OUTPUT_DIR="/scratch/at6646/sft_checkpoints/${MAX_EXAMPLES}"
    MAX_EXAMPLES_ARG="--max-examples $MAX_EXAMPLES"
    N_EXAMPLES=$MAX_EXAMPLES
fi

# Compute total optimizer steps and eval interval
EFFECTIVE_BATCH_SIZE=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))
TOTAL_STEPS=$(( (N_EXAMPLES * NUM_EPOCHS) / EFFECTIVE_BATCH_SIZE ))
EVAL_INTERVAL=$(( TOTAL_STEPS / EVAL_POINTS ))
if [ "$EVAL_INTERVAL" -eq 0 ]; then EVAL_INTERVAL=1; fi

echo "=== SFT CONFIG ==="
echo "Dataset size:       $MAX_EXAMPLES (actual: $N_EXAMPLES)"
echo "Epochs:             $NUM_EPOCHS"
echo "Effective batch:    $EFFECTIVE_BATCH_SIZE"
echo "Total steps:        $TOTAL_STEPS"
echo "Eval every:         $EVAL_INTERVAL steps"
echo "Output:             $OUTPUT_DIR"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

export HF_HOME=/scratch/at6646/hf_cache

echo "Starting SFT training..."
uv run python ../student/sft_experiments.py \
    --model "$MODEL" \
    --data-path "$TRAIN_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_EXAMPLES_ARG \
    --epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --eval-every $EVAL_INTERVAL \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --seed $SEED

echo "############### End: $(date +%Y-%m-%d_%H:%M:%S) ###############"