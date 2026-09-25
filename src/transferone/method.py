"""Paper-aligned reward construction and rubric-set selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PAPER_CONFIG


def _standardize_last_dim(
    values: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = values.float()
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True, unbiased=False)
    has_variance = std.squeeze(-1) > 0
    standardized = (values - mean) / (std + epsilon)
    return standardized, has_variance


def response_score_advantages(
    scores: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Normalize judge scores across responses as in paper Equation 1."""
    if scores.ndim < 1 or scores.shape[-1] <= 1:
        raise ValueError("scores must end in a response dimension larger than one")
    advantages, _ = _standardize_last_dim(scores, epsilon=epsilon)
    return advantages


def interval_pairwise_advantages(
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Compute interval-weighted pairwise advantages from Equations 8 and 9."""
    if center.shape != lower.shape or center.shape != upper.shape:
        raise ValueError("center, lower, and upper must have equal shapes")
    if center.ndim < 1 or center.shape[-1] <= 1:
        raise ValueError("inputs must end in a response dimension larger than one")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if torch.any(lower > upper):
        raise ValueError("lower bounds cannot exceed upper bounds")

    center = center.float()
    lower = lower.float()
    upper = upper.float()
    response_count = center.shape[-1]
    widths = upper - lower
    advantages = torch.zeros_like(center)

    for left in range(response_count):
        pair_terms = []
        for right in range(response_count):
            if left == right:
                continue
            overlap = (
                torch.minimum(upper[..., left], upper[..., right])
                - torch.maximum(lower[..., left], lower[..., right])
            ).clamp_min(0.0)
            narrower_width = torch.minimum(widths[..., left], widths[..., right])
            denominator = narrower_width.clamp_min(epsilon)
            weight = 1.0 - overlap / denominator
            direction = torch.sign(center[..., left] - center[..., right])
            pair_terms.append(weight * direction)
        advantages[..., left] = torch.stack(pair_terms, dim=-1).mean(dim=-1)
    return advantages


@dataclass(frozen=True)
class IterationSignals:
    """All fixed rewards and advantages needed for the two GRPO updates."""

    response_advantages: torch.Tensor
    alignment_rewards: torch.Tensor
    total_rubric_rewards: torch.Tensor
    rubric_advantages: torch.Tensor
    selected_response_advantages: torch.Tensor
    selected_rubric_indices: torch.Tensor
    eligible_queries: torch.Tensor
    rubric_groups_with_variance: torch.Tensor


def build_iteration_signals(
    judge_scores: torch.Tensor,
    base_rubric_rewards: torch.Tensor,
    candidate_gradients: torch.Tensor,
    reference_gradient: torch.Tensor,
    *,
    epsilon: float,
    alignment_weight: float = PAPER_CONFIG.alignment_weight,
) -> IterationSignals:
    """Build Algorithm 1 signals after all policy gradients are measured.

    Shapes are ``judge_scores=[B,K,G]``, ``base_rubric_rewards=[B,K]``,
    ``candidate_gradients=[B,K,D]``, and ``reference_gradient=[D]``.
    """
    if judge_scores.ndim != 3:
        raise ValueError("judge_scores must have shape [queries, rubrics, responses]")
    batch_size, rubric_count, _ = judge_scores.shape
    if base_rubric_rewards.shape != (batch_size, rubric_count):
        raise ValueError("base_rubric_rewards must have shape [queries, rubrics]")
    if candidate_gradients.ndim != 3:
        raise ValueError("candidate_gradients must have shape [queries, rubrics, D]")
    if candidate_gradients.shape[:2] != (batch_size, rubric_count):
        raise ValueError("candidate gradient query/rubric dimensions differ")
    if reference_gradient.ndim != 1:
        raise ValueError("reference_gradient must have shape [D]")
    if candidate_gradients.shape[-1] != reference_gradient.numel():
        raise ValueError("candidate and reference gradient dimensions differ")
    devices = {
        judge_scores.device,
        base_rubric_rewards.device,
        candidate_gradients.device,
        reference_gradient.device,
    }
    if len(devices) != 1:
        raise ValueError("all inputs must be on the same device")
    if alignment_weight < 0:
        raise ValueError("alignment_weight must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    response_advantages = response_score_advantages(
        judge_scores,
        epsilon=epsilon,
    )
    candidates = candidate_gradients.float()
    reference = reference_gradient.float()
    dot_products = torch.einsum("bkd,d->bk", candidates, reference)
    candidate_norms = torch.linalg.vector_norm(candidates, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference)
    norm_products = candidate_norms * reference_norm
    alignment = torch.where(
        norm_products > 0,
        dot_products / torch.where(
            norm_products > 0,
            norm_products,
            torch.ones_like(norm_products),
        ),
        torch.zeros_like(dot_products),
    )

    total_rewards = base_rubric_rewards.float() + alignment_weight * alignment
    rubric_advantages, has_variance = _standardize_last_dim(
        total_rewards,
        epsilon=epsilon,
    )

    best_alignment, best_indices = alignment.max(dim=-1)
    eligible = best_alignment > 0.0
    rubric_advantages = torch.where(
        eligible.unsqueeze(-1),
        rubric_advantages,
        torch.zeros_like(rubric_advantages),
    )

    rows = torch.arange(batch_size, device=response_advantages.device)
    selected = response_advantages[rows, best_indices]
    selected = torch.where(
        eligible.unsqueeze(-1),
        selected,
        torch.zeros_like(selected),
    )
    selected_indices = torch.where(
        eligible,
        best_indices,
        torch.full_like(best_indices, -1),
    )

    return IterationSignals(
        response_advantages=response_advantages,
        alignment_rewards=alignment,
        total_rubric_rewards=total_rewards,
        rubric_advantages=rubric_advantages,
        selected_response_advantages=selected,
        selected_rubric_indices=selected_indices,
        eligible_queries=eligible,
        rubric_groups_with_variance=has_variance,
    )
