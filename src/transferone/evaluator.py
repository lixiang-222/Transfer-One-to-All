"""A small embedding-based evaluator matching paper Equations 6 and 7."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def pinball_loss(residual: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return mean quantile loss ``max(tau*v, (tau-1)*v)``."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


@dataclass(frozen=True)
class IntervalPrediction:
    center: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class CoarsePreferenceEvaluator(nn.Module):
    """Score head plus lower/upper residual-quantile heads.

    The frozen reward model encoder is intentionally external. This module
    consumes its fixed query-response embeddings and never requires model or
    dataset files.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        alpha: float,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.score_head = _MLP(input_dim, hidden_dim, 1)
        self.lower_quantile_head = _MLP(input_dim, hidden_dim, 1)
        self.upper_quantile_head = _MLP(input_dim, hidden_dim, 1)

    def center(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.score_head(embeddings).squeeze(-1)

    def interval(self, embeddings: torch.Tensor) -> IntervalPrediction:
        center = self.center(embeddings)
        lower_offset = self.lower_quantile_head(embeddings).squeeze(-1)
        upper_offset = self.upper_quantile_head(embeddings).squeeze(-1)
        return IntervalPrediction(
            center=center,
            lower=center + lower_offset,
            upper=center + upper_offset,
        )

    def fit_score_head(
        self,
        train_embeddings: torch.Tensor,
        train_scores: torch.Tensor,
        *,
        steps: int,
        learning_rate: float,
    ) -> list[float]:
        """Fit ``m(q,a)`` by squared error on the score-training split."""
        _validate_examples(train_embeddings, train_scores)
        if steps <= 0 or learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive")
        self.score_head.requires_grad_(True)
        self.lower_quantile_head.requires_grad_(False)
        self.upper_quantile_head.requires_grad_(False)
        optimizer = torch.optim.Adam(self.score_head.parameters(), lr=learning_rate)
        losses = []
        for _ in range(steps):
            loss = torch.mean((self.center(train_embeddings) - train_scores.float()) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.score_head.requires_grad_(False)
        return losses

    def fit_quantile_heads(
        self,
        calibration_embeddings: torch.Tensor,
        calibration_scores: torch.Tensor,
        *,
        steps: int,
        learning_rate: float,
    ) -> list[float]:
        """Fit residual quantiles on data disjoint from score-head training."""
        _validate_examples(calibration_embeddings, calibration_scores)
        if steps <= 0 or learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive")
        self.score_head.requires_grad_(False)
        self.lower_quantile_head.requires_grad_(True)
        self.upper_quantile_head.requires_grad_(True)
        with torch.no_grad():
            residual_targets = (
                calibration_scores.float() - self.center(calibration_embeddings)
            )
        optimizer = torch.optim.Adam(
            [
                *self.lower_quantile_head.parameters(),
                *self.upper_quantile_head.parameters(),
            ],
            lr=learning_rate,
        )
        low_tau = self.alpha / 2.0
        high_tau = 1.0 - self.alpha / 2.0
        losses = []
        for _ in range(steps):
            lower = self.lower_quantile_head(calibration_embeddings).squeeze(-1)
            upper = self.upper_quantile_head(calibration_embeddings).squeeze(-1)
            loss = pinball_loss(
                residual_targets - lower,
                low_tau,
            ) + pinball_loss(
                residual_targets - upper,
                high_tau,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.lower_quantile_head.requires_grad_(False)
        self.upper_quantile_head.requires_grad_(False)
        return losses

    def freeze(self) -> "CoarsePreferenceEvaluator":
        self.requires_grad_(False)
        self.eval()
        return self


def _validate_examples(embeddings: torch.Tensor, scores: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [examples, dimensions]")
    if scores.ndim != 1 or scores.shape[0] != embeddings.shape[0]:
        raise ValueError("scores must have shape [examples]")
    if embeddings.shape[0] == 0:
        raise ValueError("at least one example is required")
