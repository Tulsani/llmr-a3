import torch

def compute_group_normalized_rewards(reward_fn,
                                     rollout_responses,
                                     repeated_ground_truths,
                                     group_size,
                                     advantage_eps,
                                     normalize_by_std):
    
    return

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages,
                                       policy_log_probs):
    return

def compute_grpo_clip_loss(advantages,
                           policy_log_probs,
                           old_log_probs,
                           cliprange):
    return

def compute_policy_gradient_loss(policy_log_probs,
                                 loss_type,
                                 raw_rewards,
                                 advanatages,
                                 old_log_probs,
                                 cliprange):
    return


def mask_mean(tensor,mask,dim):
    return

def grpo_microsbatch_train_step(policy_log_probs,
                                response_mask,
                                gradient_accumulation_steps,
                                loss_type,
                                raw_rewards,
                                advantages,
                                old_log_probs,
                                cliprange):
    return