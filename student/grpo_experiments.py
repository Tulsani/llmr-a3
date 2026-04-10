import argparse
import json
import os
import random
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from student.grpo_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    mask_mean,
)
from student.sft_helpers import get_response_log_probs, tokenize_prompt_and_output

def load_countdown_data(path: str) -> list[dict]:
    """
    Load countdown parquet (or HF arrow dir) and normalise to
    {'numbers': [...], 'target': int} dicts.
    Handles the schema: cols 'nums'/'numbers' and 'target'/'answer'.
    """
    p = Path(path)
    if p.is_dir():
        parquet_files = list(p.glob("*.parquet"))
        if parquet_files:
            ds = load_dataset("parquet",
                              data_files=[str(f) for f in parquet_files],
                              split="train")
        else:
            from datasets import load_from_disk
            ds = load_from_disk(path)
    else:
        ds = load_dataset("parquet", data_files=str(p), split="train")

    examples = []
    for ex in ds:
        numbers = list(ex.get("nums", ex.get("numbers", [])))
        target  = int(ex.get("target", ex.get("answer", 0)))
        examples.append({"numbers": numbers, "target": target})
    return examples


def load_countdown_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "countdown.prompt"
    return path.read_text().strip()


def make_prompt(prompt_template: str, ex: dict) -> str:
    numbers = ex["numbers"]
    target  = ex["target"]
    problem = (
        f"Using the numbers in the list {numbers}, "
        f"create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) "
        f"and each number can only be used once."
    )
    return prompt_template.replace("{question}", problem)


def make_ground_truth(ex: dict) -> str:
    """Serialise ground truth as JSON so reward_fn can parse it."""
    return json.dumps({"target": ex["target"],
                       "numbers": sorted(ex["numbers"])})


def _try_evaluate(expr: str):
    """Safely evaluate a arithmetic expression string."""
    safe_chars = set("0123456789+-*/() .\t\n")
    if not all(c in safe_chars for c in expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def countdown_reward_fn(response: str, ground_truth: str) -> dict:
    import re

    gt = json.loads(ground_truth)
    target  = int(gt["target"])
    allowed = sorted(int(x) for x in gt["numbers"])

    # Format check
    if "<answer>" not in response or "</answer>" not in response:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    answer_text = response.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()

    # Try each line from last to first looking for a line that evaluates to target
    for line in reversed(answer_text.splitlines()):
        # Strip "Step X:" prefixes
        line = re.sub(r"^\s*Step\s*\d+\s*[:.]\s*", "", line).strip()
        if not line:
            continue

        # If line has '=', take the LHS
        if "=" in line:
            lhs = line.rsplit("=", 1)[0].strip()
        else:
            lhs = line

        result = _try_evaluate(lhs)
        if result is not None and abs(result - target) < 1e-6:
            # For single-line answers: also verify numbers used match allowed set
            nums_in_lhs = sorted(int(n) for n in re.findall(r"\b\d+\b", lhs))
            if nums_in_lhs == allowed:
                return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
            # For multi-step: accept if the full answer block uses exactly the right numbers
            nums_in_block = sorted(int(n) for n in re.findall(r"\b\d+\b", answer_text)
                                   if int(n) in allowed)
            all_nums = sorted(int(n) for n in re.findall(r"\b\d+\b", answer_text))
            # Check every number in allowed appears exactly once across all steps
            from collections import Counter
            block_counter  = Counter(int(n) for n in re.findall(r"\b\d+\b", answer_text))
            allowed_counter = Counter(allowed)
            if all(block_counter[k] >= v for k, v in allowed_counter.items()):
                return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    # Also try evaluating the entire answer block as a single expression
    single = _try_evaluate(answer_text.replace("\n", " "))
    if single is not None and abs(single - target) < 1e-6:
        nums_used = sorted(int(n) for n in re.findall(r"\b\d+\b", answer_text))
        if nums_used == allowed:
            return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


def init_vllm(model_id: str, device: str, seed: int,
              gpu_memory_utilization: float = 0.8) -> LLM:
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


def load_policy_into_vllm(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())



def evaluate_countdown(llm: LLM, examples: list[dict],
                       prompt_template: str, reward_fn) -> dict:
    prompts       = [make_prompt(prompt_template, ex) for ex in examples]
    ground_truths = [make_ground_truth(ex) for ex in examples]

    params  = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
    outputs = llm.generate(prompts, params)

    total = fmt = ans = 0.0
    for out, gt in zip(outputs, ground_truths):
        text   = out.outputs[0].text
        result = reward_fn(text, gt)
        total += result["reward"]
        fmt   += result["format_reward"]
        ans   += result["answer_reward"]

    n = len(examples)
    return {"reward": total / n, "format_reward": fmt / n, "answer_reward": ans / n}


def grpo_train_loop(args):
    wandb.init(project="llm-reasoners-grpo", config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    device      = args.policy_device
    vllm_device = args.vllm_device

    n_grpo_steps                = args.n_grpo_steps
    learning_rate               = args.learning_rate
    advantage_eps               = args.advantage_eps
    rollout_batch_size          = args.rollout_batch_size
    group_size                  = args.group_size
    sampling_temperature        = args.sampling_temperature
    sampling_min_tokens         = args.sampling_min_tokens
    sampling_max_tokens         = args.sampling_max_tokens
    epochs_per_rollout_batch    = args.epochs_per_rollout_batch
    train_batch_size            = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    loss_type                   = args.loss_type
    norm_type                   = args.norm_type
    use_std_normalization       = args.use_std_normalization
    eval_every                  = args.eval_every

    # Sanity checks
    assert train_batch_size % gradient_accumulation_steps == 0, (
        f"train_batch_size ({train_batch_size}) must be divisible by "
        f"gradient_accumulation_steps ({gradient_accumulation_steps})"
    )
    assert rollout_batch_size % group_size == 0
    assert train_batch_size >= group_size

    micro_train_batch_size      = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    n_microbatches_per_rollout  = rollout_batch_size // micro_train_batch_size

    # normalize_constant for masked_normalize (use max_tokens as the constant)
    normalize_constant = float(sampling_max_tokens) if norm_type == "masked_normalize" else None

    # Model
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

    # vLLM
    print("Initializing vLLM...")
    llm = init_vllm(args.model, vllm_device, seed=42,
                    gpu_memory_utilization=args.gpu_memory_utilization)

    # Data
    print("Loading Countdown datasets...")
    train_data = load_countdown_data(args.train_path)
    val_data   = load_countdown_data(args.val_path)
    print(f"  train: {len(train_data)} | val: {len(val_data)}")

    prompt_template = load_countdown_prompt()
    reward_fn       = countdown_reward_fn
    val_examples    = val_data[:args.n_eval_examples]

    # Initial eval
    eval_step = 0
    print("Initial evaluation...")
    load_policy_into_vllm(policy, llm)
    eval_results = evaluate_countdown(llm, val_examples, prompt_template, reward_fn)
    wandb.log({f"eval/{k}": v for k, v in eval_results.items()} | {"eval_step": eval_step})
    print(f"[eval 0] {eval_results}")
    eval_step += 1

    # GRPO loop
    train_step = 0

    for grpo_step in range(n_grpo_steps):
        policy.eval()

        #  Sample questions
        batch   = random.sample(train_data, n_prompts_per_rollout_batch)
        prompts = [make_prompt(prompt_template, ex) for ex in batch]
        gts     = [make_ground_truth(ex) for ex in batch]

        #  Generate rollouts (n=group_size per prompt)
        load_policy_into_vllm(policy, llm)
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            stop=["</answer>"],
            n=group_size,
        )
        outputs = llm.generate(prompts, sampling_params)
        rollout_responses = [
            completion.text
            for output in outputs
            for completion in output.outputs
        ]

        # Expand prompts/gts to match rollout_batch_size
        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_gts     = [gt for gt in gts for _ in range(group_size)]

        #  Rewards & advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        wandb.log({
            "train/mean_reward":        reward_metadata["mean_reward"],
            "train/mean_format_reward": reward_metadata.get("mean_format_reward", 0),
            "train/mean_answer_reward": reward_metadata.get("mean_answer_reward", 0),
            "train_step": train_step,
        })

        # Tokenize
        tokenized     = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer)
        input_ids     = tokenized["input_ids"].to(device)
        labels        = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        # (rollout_batch_size, 1) for broadcasting in loss functions
        adv_tensor = advantages.to(device).unsqueeze(-1)
        raw_tensor = raw_rewards.to(device).unsqueeze(-1)

        #  Old log probs for grpo_clip
        old_log_probs = None
        if loss_type == "grpo_clip":
            policy.eval()
            with torch.inference_mode():
                old_log_probs = get_response_log_probs(
                    model=policy, input_ids=input_ids, labels=labels,
                )["log_probs"].detach()

        #  Inner training loop
        policy.train()
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

            for mb_idx in range(n_microbatches_per_rollout):
                start = mb_idx * micro_train_batch_size
                end   = start  + micro_train_batch_size

                mb_input_ids     = input_ids[start:end]
                mb_labels        = labels[start:end]
                mb_response_mask = response_mask[start:end]
                mb_adv           = adv_tensor[start:end]
                mb_raw           = raw_tensor[start:end]
                mb_old_lp        = (old_log_probs[start:end]
                                    if old_log_probs is not None else None)

                log_probs_out = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=True,
                )
                mb_policy_log_probs = log_probs_out["log_probs"]
                token_entropy       = log_probs_out["token_entropy"]

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                    normalize_constant=normalize_constant,
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            avg_entropy = (mask_mean(token_entropy, mb_response_mask).item()
                           if token_entropy is not None else 0.0)

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

        # 7. Periodic eval
        if (grpo_step + 1) % eval_every == 0:
            policy.eval()
            load_policy_into_vllm(policy, llm)
            eval_results = evaluate_countdown(
                llm, val_examples, prompt_template, reward_fn)
            wandb.log({f"eval/{k}": v for k, v in eval_results.items()}
                      | {"eval_step": eval_step})
            print(f"[grpo_step {grpo_step+1}] eval: {eval_results}")
            eval_step += 1

    # Save
    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()
    print("Done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train-path",    default="data/countdown/train")
    parser.add_argument("--val-path",      default="data/countdown/dev")
    parser.add_argument("--output-dir",    default="/scratch/at6646/llmr-a3/grpo_model")
    parser.add_argument("--policy-device", default="cuda:1")
    parser.add_argument("--vllm-device",   default="cuda:0")

    parser.add_argument("--n-grpo-steps",              type=int,   default=200)
    parser.add_argument("--learning-rate",             type=float, default=1e-5)
    parser.add_argument("--advantage-eps",             type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size",        type=int,   default=16)
    parser.add_argument("--group-size",                type=int,   default=8)
    parser.add_argument("--sampling-temperature",      type=float, default=0.7)
    parser.add_argument("--sampling-min-tokens",       type=int,   default=4)
    parser.add_argument("--sampling-max-tokens",       type=int,   default=1024)
    parser.add_argument("--epochs-per-rollout-batch",  type=int,   default=1)
    parser.add_argument("--train-batch-size",          type=int,   default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    parser.add_argument("--loss-type", default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--norm-type", default="masked_mean",
                        choices=["masked_mean", "masked_normalize"])
    parser.add_argument("--use-std-normalization",  action="store_true", default=True)
    parser.add_argument("--no-std-normalization",   dest="use_std_normalization",
                        action="store_false")

    parser.add_argument("--cliprange",              type=float, default=0.2)
    parser.add_argument("--eval-every",             type=int,   default=10)
    parser.add_argument("--n-eval-examples",        type=int,   default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)

    args = parser.parse_args()
    print("\n========  starting training ==========")
    grpo_train_loop(args)


if __name__ == "__main__":
    main()