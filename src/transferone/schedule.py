"""Execution skeleton for the paper's alternating training schedule."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import PAPER_CONFIG, ModuleName, PaperConfig


@dataclass(frozen=True)
class UpdateContext:
    stage_index: int
    step_in_stage: int
    global_update_step: int
    active_module: ModuleName
    train_prompts: int
    validation_prompts: int
    rubric_sets_per_query: int
    responses_per_query: int


UpdateFunction = Callable[[UpdateContext], Any]


def run_paper_schedule(
    update_fn: UpdateFunction,
    *,
    first_module: ModuleName,
    config: PaperConfig = PAPER_CONFIG,
) -> list[Any]:
    """Run all 10 x 20 update slots from the current paper configuration.

    The paper does not state which module owns the first stage, so callers must
    choose it explicitly. Every other schedule value is fixed by ``PaperConfig``.
    """
    results = []
    global_step = 0
    for stage in config.stage_plan(first_module=first_module):
        for step_in_stage in range(stage.update_steps):
            context = UpdateContext(
                stage_index=stage.index,
                step_in_stage=step_in_stage,
                global_update_step=global_step,
                active_module=stage.module,
                train_prompts=stage.train_prompts_per_step,
                validation_prompts=stage.validation_prompts_per_step,
                rubric_sets_per_query=config.rubric_sets_per_query,
                responses_per_query=config.responses_per_query,
            )
            results.append(update_fn(context))
            global_step += 1
    return results
