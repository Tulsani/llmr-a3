import argparse
import os
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from student.grpo_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    mask_mean
)
from student.sft_helpers import get_response_log_probs, tokenize_prompt_and_output
from student.drgrpo_grader import question_only_reward_fn,r1_zero_reward_fn


def init_vllm(model_id, device, seed, gpu_memory_utilization=0.85):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
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
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_countdown_prompt():
    from pathlib import Path
    path = Path(__file__).parent / "prompts" / "countdown.prompt"
    return path.read_text()

def evaluate_countdown(llm, prompts, ground_truths, reward_fn):
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</answer>"],
    )
    outputs = llm.generate(prompts, params)

    total_reward = format_reward = answer_reward = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        result = reward_fn(text, ground_truths[i])
        total_reward  += result["reward"]
        format_reward += result["format_reward"]
        answer_reward += result["answer_reward"]

    n = len(outputs)
    return {
        "reward":        total_reward  / n,
        "format_reward": format_reward / n,
        "answer_reward": answer_reward / n,
    }

def make_prompt(prompt_template,ex):
        return prompt_template + "\n\n" + ex["problem"]

def generate_rollouts(llm, prompts, group_size, sampling_temperature,
                      sampling_min_tokens, sampling_max_tokens):

    params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        n=group_size,
    )
    outputs = llm.generate(prompts, params)

    rollout_responses = []
    for output in outputs:
        for completion in output.outputs:
            rollout_responses.append(completion.text)
    return rollout_responses


def grpo_train_loop(args):
    wandb.init(project="llm-reasoners-grpo", config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    device = args.policy_device
    vllm_device = args.vllm_device

    n_grpo_steps= args.n_grpo_steps
    learning_rate = args.learning_rate
    advantage_eps = args.advantage_eps
    rollout_batch_size = args.rollout_batch_size
    group_size = args.group_size
    sampling_temperature = args.sampling_temperature
    sampling_min_tokens = args.sampling_min_tokens
    sampling_max_tokens = args.sampling_max_tokens
    epochs_per_rollout_batch = args.epochs_per_rollout_batch
    train_batch_size  = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    loss_type = args.loss_type
    use_std_normalization = args.use_std_normalization
    eval_every = args.eval_every

    ## suggested sanity checks
    assert train_batch_size % gradient_accumulation_steps == 0
    assert rollout_batch_size % group_size == 0
    assert train_batch_size >= group_size

    micro_train_batch_size       = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch  = rollout_batch_size // group_size
    n_microbatches_per_rollout   = rollout_batch_size // micro_train_batch_size

    ## model setup
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # vllm
    print("Initializing vLLM...")
    llm = init_vllm(args.model, vllm_device, seed=42,
                    gpu_memory_utilization=args.gpu_memory_utilization)
    
    # dataset
    print("Loading Countdown datasets...")
    train_ds = load_from_disk(args.train_path)
    val_ds   = load_from_disk(args.val_path)

    prompt_template = load_countdown_prompt()

    val_prompts = [make_prompt(prompt_template,ex) for ex in val_ds]
    val_gts     = [ex["answer"]    for ex in val_ds]

    # reward fn
    reward_fn = r1_zero_reward_fn

    #intial policy eval
    eval_step = 0
    print("Initial evaluation...")
    load_policy_into_vllm_instance(policy, llm)
    eval_results = evaluate_countdown(
        llm, val_prompts[:args.n_eval_examples], 
        val_gts[:args.n_eval_examples], reward_fn
    )
    wandb.log({f"eval/{k}": v for k, v in eval_results.items()} | {"eval_step": eval_step})
    print(f"[eval 0] {eval_results}")
    eval_step += 1

    ## grpo look
    train_data = list(train_ds)
    train_step = 0

    for grpo_step in range(n_grpo_steps):
        policy.eval()

        #  Sample batch of questions
        indices = torch.randint(0, len(train_data), (n_prompts_per_rollout_batch,)).tolist()
        batch   = [train_data[i] for i in indices]
        prompts = [make_prompt(ex) for ex in batch]
        gts     = [ex["answer"]   for ex in batch]

        # generate rollouts
        load_policy_into_vllm_instance(policy, llm)
        rollout_responses = generate_rollouts(
            llm, prompts, group_size,
            sampling_temperature, sampling_min_tokens, sampling_max_tokens,
        )

        # repeated ground truth
        repeated_gts = [gt for gt in gts for _ in range(group_size)]

        # compute rewards & advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # log train rewards
        wandb.log({
            "train/mean_reward":   reward_metadata["mean_reward"],
            "train/mean_format_reward": reward_metadata.get("mean_format_reward", 0),
            "train/mean_answer_reward": reward_metadata.get("mean_answer_reward", 0),
            "train_step": train_step,
        })

        # tokenize rollouts
        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )
        input_ids     = tokenized["input_ids"].to(device)
        labels        = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        # advantages
        advantages = advantages.to(device).unsqueeze(-1)

        # get old log probs 
        old_log_probs = None
        if loss_type == "grpo_clip":
            policy.eval()
            with torch.inference_mode():
                old_out = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                )
            old_log_probs = old_out["log_probs"].detach()

        # training loop
        policy.train()
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

            # microbatches
            for mb_idx in range(n_microbatches_per_rollout):
                start = mb_idx * micro_train_batch_size
                end   = start  + micro_train_batch_size

                mb_input_ids     = input_ids[start:end]
                mb_labels        = labels[start:end]
                mb_response_mask = response_mask[start:end]
                mb_advantages    = advantages[start:end]
                mb_old_log_probs = old_log_probs[start:end] if old_log_probs is not None else None

                # Forward pass
                log_probs_out = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=True,
                )
                mb_policy_log_probs = log_probs_out["log_probs"]
                token_entropy       = log_probs_out["token_entropy"]

                # GRPO microbatch step
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=args.cliprange,
                )

            # optimizer
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            avg_entropy = mask_mean(token_entropy, mb_response_mask).item() \
                if token_entropy is not None else 0.0

            log_dict = {
                "train/loss":          loss.item(),
                "train/grad_norm":     grad_norm.item(),
                "train/token_entropy": avg_entropy,
                "train_step":          train_step,
            }
            if "clip_fraction" in metadata:
                log_dict["train/clip_fraction"] = metadata["clip_fraction"].item()
            wandb.log(log_dict)
            train_step += 1

        # periodic evaluation
        if (grpo_step + 1) % eval_every == 0:
            policy.eval()
            load_policy_into_vllm_instance(policy, llm)
            eval_results = evaluate_countdown(
                llm, val_prompts[:args.n_eval_examples],
                val_gts[:args.n_eval_examples], reward_fn,
            )
            wandb.log({f"eval/{k}": v for k, v in eval_results.items()} | {"eval_step": eval_step})
            print(f"[grpo_step {grpo_step+1}] eval: {eval_results}")
            eval_step += 1

    # save
    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()
    print("\nDone")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train-path",      default="data/countdown/train")
    parser.add_argument("--val-path",        default="data/countdown/dev")
    parser.add_argument("--output-dir",      default="/scratch/at6646/llmr-a3/llmr-a3/grpo_model")
    parser.add_argument("--policy-device",   default="cuda:1")
    parser.add_argument("--vllm-device",     default="cuda:0")
    parser.add_argument("--n-grpo-steps",         type=int,   default=200)
    parser.add_argument("--learning-rate",         type=float, default=1e-5)
    parser.add_argument("--advantage-eps",         type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size",    type=int,   default=16)
    parser.add_argument("--group-size",            type=int,   default=8)
    parser.add_argument("--sampling-temperature",  type=float, default=0.7)
    parser.add_argument("--sampling-min-tokens",   type=int,   default=4)
    parser.add_argument("--sampling-max-tokens",   type=int,   default=1024)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size",      type=int,   default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--loss-type",   default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use-std-normalization",  action="store_true", default=True)
    parser.add_argument("--cliprange",              type=float, default=0.2)
    parser.add_argument("--eval-every",             type=int,   default=10)
    parser.add_argument("--n-eval-examples",        type=int,   default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    #start training
    print("\n========  starting training ==========")
    grpo_train_loop(args)


if __name__ == "__main__":
    main()