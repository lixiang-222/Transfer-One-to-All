"""Preference-anchored learning for query-specific rubric sets."""

from .config import PAPER_CONFIG, PaperConfig, TrainingStage
from .evaluator import CoarsePreferenceEvaluator, pinball_loss
from .grpo import (
    cosine_similarity,
    flatten_gradients,
    grpo_objective,
    objective_gradient,
)
from .method import (
    IterationSignals,
    build_iteration_signals,
    interval_pairwise_advantages,
    response_score_advantages,
)
from .schedule import UpdateContext, run_paper_schedule

__all__ = [
    "CoarsePreferenceEvaluator",
    "IterationSignals",
    "PAPER_CONFIG",
    "PaperConfig",
    "TrainingStage",
    "UpdateContext",
    "build_iteration_signals",
    "cosine_similarity",
    "flatten_gradients",
    "grpo_objective",
    "interval_pairwise_advantages",
    "objective_gradient",
    "pinball_loss",
    "response_score_advantages",
    "run_paper_schedule",
]

__version__ = "0.1.0"
