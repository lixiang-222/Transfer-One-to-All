"""A data-free CPU smoke run of one paper-aligned training iteration."""

from __future__ import annotations

import copy
import json

import torch
from torch import nn

from .config import PAPER_CONFIG
from .grpo import grpo_objective, objective_gradient
from .method import (
    build_iteration_signals,
    interval_pairwise_advantages,
    response_score_advantages,
)

SMOKE_CLIP_DELTA = 0.2
SMOKE_EPSILON = 1.0e-6


class TinyPolicy(nn.Module):
    def __init__(self, feature_dim: int, vocabulary_size: int):
        super().__init__()
        self.output = nn.Linear(feature_dim, vocabulary_size)

    def log_probs(
        self,
        features: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.output(features)
        return logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)


def _fixed_log_probs(
    policy: TinyPolicy,
    features: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return policy.log_probs(features, token_ids)


def _policy_objective(
    policy: TinyPolicy,
    old_policy: TinyPolicy,
    reference_policy: TinyPolicy,
    features: torch.Tensor,
    token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_delta: float,
    kl_coefficient: float,
) -> torch.Tensor:
    shape = features.shape
    flat_features = features.reshape(-1, shape[-2], shape[-1])
    flat_tokens = token_ids.reshape(-1, shape[-2])
    flat_advantages = advantages.reshape(-1)
    new = policy.log_probs(flat_features, flat_tokens)
    old = _fixed_log_probs(old_policy, flat_features, flat_tokens)
    reference = _fixed_log_probs(reference_policy, flat_features, flat_tokens)
    return grpo_objective(
        new,
        old,
        reference,
        flat_advantages,
        torch.ones_like(new, dtype=torch.bool),
        clip_delta=clip_delta,
        kl_coefficient=kl_coefficient,
    )


def run_demo(seed: int = 7) -> dict[str, object]:
    """Run one paper-shaped Algorithm 1 iteration on synthetic tensors.

    The clipping threshold and numerical epsilon are smoke-test choices because
    the paper does not report their values.
    """
    torch.manual_seed(seed)
    batch_size = PAPER_CONFIG.train_prompts_per_step
    validation_batch_size = PAPER_CONFIG.validation_prompts_per_step
    rubric_count = PAPER_CONFIG.rubric_sets_per_query
    response_count = PAPER_CONFIG.responses_per_query
    token_count, feature_dim, vocabulary_size = 3, 5, 7

    policy = TinyPolicy(feature_dim, vocabulary_size)
    old_policy = copy.deepcopy(policy).requires_grad_(False)
    reference_policy = copy.deepcopy(policy).requires_grad_(False)
    with torch.no_grad():
        for parameter in reference_policy.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))

    train_features = torch.randn(
        batch_size,
        response_count,
        token_count,
        feature_dim,
    )
    validation_features = torch.randn(
        validation_batch_size,
        response_count,
        token_count,
        feature_dim,
    )
    train_token_ids = torch.randint(
        vocabulary_size,
        (batch_size, response_count, token_count),
    )
    validation_token_ids = torch.randint(
        vocabulary_size,
        (validation_batch_size, response_count, token_count),
    )

    train_quality = train_features[..., 0].mean(dim=-1)
    validation_center = validation_features[..., 0].mean(dim=-1)
    half_width = torch.tensor([0.10, 0.20, 0.15, 0.25]).expand_as(
        validation_center
    )
    validation_advantages = interval_pairwise_advantages(
        validation_center,
        validation_center - half_width,
        validation_center + half_width,
        epsilon=SMOKE_EPSILON,
    )
    reference_objective = _policy_objective(
        policy,
        old_policy,
        reference_policy,
        validation_features,
        validation_token_ids,
        validation_advantages,
        clip_delta=SMOKE_CLIP_DELTA,
        kl_coefficient=PAPER_CONFIG.kl_coefficient,
    )
    reference_gradient = objective_gradient(
        reference_objective,
        policy.parameters(),
    )

    judge_scores = torch.stack(
        [
            train_quality,
            -train_quality,
            train_quality.roll(1, dims=-1),
            torch.randn_like(train_quality),
        ],
        dim=1,
    )
    answer_advantages = response_score_advantages(
        judge_scores,
        epsilon=SMOKE_EPSILON,
    )
    candidate_gradients = torch.empty(
        batch_size,
        rubric_count,
        reference_gradient.numel(),
    )
    for query_index in range(batch_size):
        query_features = train_features[query_index : query_index + 1]
        query_tokens = train_token_ids[query_index : query_index + 1]
        for rubric_index in range(rubric_count):
            objective = _policy_objective(
                policy,
                old_policy,
                reference_policy,
                query_features,
                query_tokens,
                answer_advantages[
                    query_index : query_index + 1,
                    rubric_index,
                ],
                clip_delta=SMOKE_CLIP_DELTA,
                kl_coefficient=PAPER_CONFIG.kl_coefficient,
            )
            candidate_gradients[query_index, rubric_index] = objective_gradient(
                objective,
                policy.parameters(),
            )

    # A toy discrimination score stands in for the baseline-specific R_base.
    base_rewards = judge_scores.std(dim=-1, unbiased=False)
    signals = build_iteration_signals(
        judge_scores,
        base_rewards,
        candidate_gradients,
        reference_gradient,
        epsilon=SMOKE_EPSILON,
        alignment_weight=PAPER_CONFIG.alignment_weight,
    )

    rubric_logits = nn.Parameter(torch.zeros(batch_size, rubric_count))
    rubric_optimizer = torch.optim.Adam(
        [rubric_logits],
        lr=PAPER_CONFIG.learning_rate,
    )
    old_rubric_log_probs = rubric_logits.detach().log_softmax(dim=-1)
    new_rubric_log_probs = rubric_logits.log_softmax(dim=-1)
    rubric_objective = grpo_objective(
        new_rubric_log_probs.reshape(-1, 1),
        old_rubric_log_probs.reshape(-1, 1),
        old_rubric_log_probs.reshape(-1, 1),
        signals.rubric_advantages.reshape(-1),
        torch.ones(batch_size * rubric_count, 1, dtype=torch.bool),
        clip_delta=SMOKE_CLIP_DELTA,
        kl_coefficient=PAPER_CONFIG.kl_coefficient,
    )
    rubric_optimizer.zero_grad()
    (-rubric_objective).backward()
    rubric_optimizer.step()

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=PAPER_CONFIG.learning_rate,
    )
    policy_objective = _policy_objective(
        policy,
        old_policy,
        reference_policy,
        train_features,
        train_token_ids,
        signals.selected_response_advantages,
        clip_delta=SMOKE_CLIP_DELTA,
        kl_coefficient=PAPER_CONFIG.kl_coefficient,
    )
    policy_optimizer.zero_grad()
    (-policy_objective).backward()
    policy_optimizer.step()

    return {
        "status": "ok",
        "shape": {
            "training_queries": batch_size,
            "validation_queries": validation_batch_size,
            "rubric_sets_per_query": rubric_count,
            "responses_per_query": response_count,
        },
        "paper_schedule": {
            "alternating_stages": PAPER_CONFIG.alternating_stages,
            "update_steps_per_stage": PAPER_CONFIG.update_steps_per_stage,
            "total_update_steps": PAPER_CONFIG.total_update_steps,
            "total_train_prompt_slots": PAPER_CONFIG.total_train_prompt_slots,
            "total_validation_prompt_slots": (
                PAPER_CONFIG.total_validation_prompt_slots
            ),
        },
        "eligible_queries": int(signals.eligible_queries.sum()),
        "selected_rubric_indices": signals.selected_rubric_indices.tolist(),
        "mean_alignment": float(signals.alignment_rewards.mean()),
        "rubric_objective": float(rubric_objective.detach()),
        "policy_objective": float(policy_objective.detach()),
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
