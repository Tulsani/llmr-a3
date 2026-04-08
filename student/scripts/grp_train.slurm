#!/bin/bash
#SBATCH --job-name=at6646_grpo
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

set -e
mkdir -p logs

export HF_HOME=/scratch/at6646/hf_cache

echo "Starting GRPO training..."
uv run python student/grpo_experiments.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --train-path data/countdown/train \
    --val-path data/countdown/dev \
    --output-dir /scratch/at6646/grpo_model \
    --n-grpo-steps 200 \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-every 10

echo "Done!"