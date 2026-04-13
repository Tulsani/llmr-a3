import argparse
import os
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from datasets import load_dataset

from student.sft_helpers import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from student.math_baseline_script import evaluate, load_prompt

# load dataset
class InstructDataset(Dataset):
    def __init__(self, examples, max_examples=None):
        if max_examples is not None:
            examples = examples.select(range(min(max_examples, len(examples))))
        # Convert to list of dicts properly
        self.examples = [examples[i] for i in range(len(examples))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        msgs = ex.get("messages", [])
        sys_msg  = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"),   "")
        prompt   = (sys_msg + "\n\n" + user_msg).strip() if sys_msg else user_msg
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return {"prompt": prompt, "response": asst_msg}


def collate_fn(batch, tokenizer):
    prompts   = [b["prompt"]   for b in batch]
    responses = [b["response"] for b in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)

## vllm helper 

def init_vllm(model_id, device, seed, gpu_memory_utilization=0.85):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch  = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_eval(llm, policy, prompt_template, math_ds, max_eval=500):
    """Load policy weights into vLLM and evaluate on MATH."""
    load_policy_into_vllm_instance(policy, llm)
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts     = [ex["answer"] for ex in math_ds]
    acc, _, _, _ = evaluate(llm, prompts[:max_eval], gts[:max_eval])
    return acc

def train(args):
    wandb.init(project="llm-reasoners-sft", config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # set devices
    device = "cuda:1"
    vllm_device = "cuda:0"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # init eval
    print("Initializing vLLM for evaluation...")
    llm = init_vllm(args.model, vllm_device, seed=42,
                    gpu_memory_utilization=args.gpu_memory_utilization)
    
    # math eval datasets
    math_ds = load_dataset("hiyouga/math12k", split="test")
    prompt_template = load_prompt("intellect")

    # sft datasets
    print(f"Loading Prime Intellect data from {args.data_path}...")
    raw = load_from_disk(args.data_path)
    dataset = InstructDataset(raw, max_examples=args.max_examples)
    print(f"  Using {len(dataset)} examples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)

    train_step = 0
    eval_step  = 0

    # Initial eval
    print("Running initial evaluation...")
    acc = run_eval(llm, policy, prompt_template, math_ds)
    wandb.log({"eval/math_accuracy": acc, "eval_step": eval_step})
    print(f"[eval {eval_step}] MATH accuracy: {acc:.4f}")
    eval_step += 1

    print("Starting SFT training...")
    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            input_ids     = batch["input_ids"].to(device)
            labels        = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Forward pass to get log probs
            log_probs_out = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = log_probs_out["log_probs"]

            # Microbatch train step (backward included)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            # Optimizer step after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": loss.item(),
                    "train/num_response_tokens": metadata["num_response_tokens"],
                    "train_step": train_step,
                })
                ## adding print statement for better outputs
                print(f"[train step={train_step}] loss={loss.item():.4f}")
                train_step += 1

            # Periodic evaluation
            if train_step > 0 and train_step % args.eval_every == 0:
                policy.eval()
                acc = run_eval(llm, policy, prompt_template, math_ds)
                wandb.log({"eval/math_accuracy": acc, "eval_step": eval_step})
                print(f"[eval {eval_step}] step={train_step} MATH accuracy: {acc:.4f}")
                eval_step += 1
                policy.train()

    # save
    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-path",  default="data/intellect_math_train_dev_test/train")
    parser.add_argument("--output-dir", default="/scratch/at6646/sft_model")
    parser.add_argument("--max-examples",   type=int,   default=None,
                        help="Limit dataset size e.g. 128/256/512/1024 or None for full")
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate",  type=float, default=2e-5)
    parser.add_argument("--eval-every",     type=int,   default=50,
                        help="Evaluate every N optimizer steps")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-seq-len",default=512)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()