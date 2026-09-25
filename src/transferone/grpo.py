"""The GRPO objective and gradient utilities used by the paper."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def grpo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_delta: float,
    kl_coefficient: float,
) -> torch.Tensor:
    """Return the token-normalized GRPO objective from paper Equation 2.

    The returned value is maximized. Optimizers that minimize a loss should use
    ``loss = -grpo_objective(...)``.
    """
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must have equal shapes")
    if new_log_probs.shape != reference_log_probs.shape:
        raise ValueError("reference_log_probs must match new_log_probs")
    if new_log_probs.shape != response_mask.shape:
        raise ValueError("response_mask must match new_log_probs")
    if new_log_probs.ndim != 2:
        raise ValueError("log probabilities must have shape [responses, tokens]")
    if advantages.shape != new_log_probs.shape[:1]:
        raise ValueError("advantages must have shape [responses]")
    if not 0.0 < clip_delta < 1.0:
        raise ValueError("clip_delta must be in (0, 1)")
    if kl_coefficient < 0.0:
        raise ValueError("kl_coefficient must be non-negative")

    mask = response_mask.to(dtype=new_log_probs.dtype)
    lengths = mask.sum(dim=-1)
    if torch.any(lengths <= 0):
        raise ValueError("each response must contain at least one valid token")

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    advantage = advantages.to(new_log_probs.dtype).unsqueeze(-1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_delta, 1.0 + clip_delta) * advantage

    log_kappa = reference_log_probs - new_log_probs
    kappa = torch.exp(log_kappa)
    kl_penalty = kappa - log_kappa - 1.0

    token_objective = torch.minimum(unclipped, clipped)
    token_objective = token_objective - kl_coefficient * kl_penalty
    response_objective = (token_objective * mask).sum(dim=-1) / lengths
    return response_objective.mean()


def flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten gradients, using zeros for parameters unused by an objective."""
    if len(gradients) != len(parameters):
        raise ValueError("gradients and parameters must have equal lengths")
    if not parameters:
        raise ValueError("at least one parameter is required")
    flattened = []
    for gradient, parameter in zip(gradients, parameters):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        flattened.append(value.reshape(-1))
    return torch.cat(flattened)


def objective_gradient(
    objective: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Return a detached flattened ascent gradient for an objective."""
    parameter_list = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameter_list:
        raise ValueError("no trainable parameters were supplied")
    gradients = torch.autograd.grad(
        objective,
        parameter_list,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return flatten_gradients(gradients, parameter_list).detach()


def cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return cosine alignment, assigning zero if either gradient is zero."""
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("gradient vectors must be rank-1 with equal shapes")
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if bool(left_norm == 0) or bool(right_norm == 0):
        return left.new_zeros(())
    return torch.dot(left, right) / (left_norm * right_norm)
