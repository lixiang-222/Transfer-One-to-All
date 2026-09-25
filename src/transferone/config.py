"""Parameters explicitly reported by the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ModuleName = Literal["rubric_generator", "answer_policy"]


@dataclass(frozen=True)
class TrainingStage:
    index: int
    module: ModuleName
    update_steps: int
    train_prompts_per_step: int
    validation_prompts_per_step: int


@dataclass(frozen=True)
class PaperConfig:
    """The complete set of numerical settings reported in Section A.3."""

    alignment_weight: float = 0.1
    rubric_sets_per_query: int = 4
    responses_per_query: int = 4
    learning_rate: float = 1.0e-6
    kl_coefficient: float = 0.001
    alternating_stages: int = 10
    update_steps_per_stage: int = 20
    train_prompts_per_step: int = 64
    validation_prompts_per_step: int = 64
    lora_rank: int = 16
    rubric_generator_backbone: str = "Qwen3-4B"
    answer_policy_backbone: str = "Qwen3-4B"
    frozen_judge: str = "Qwen3-8B"
    frozen_evaluator_encoder: str = "FsfairX-LLaMA3-RM-v0.1"
    training_dataset: str = "Tulu 3"
    coarse_preference_dataset: str = "HelpSteer 1"
    coarse_preference_dimensions: tuple[str, ...] = (
        "helpfulness",
        "correctness",
        "coherence",
    )
    evaluation_judges: tuple[str, ...] = (
        "DeepSeek V4-Flash",
        "GPT-5.6-Luna",
        "Gemini 3.7-Flash",
    )
    rubric_benchmarks: tuple[str, ...] = ("RewardBench 2", "JudgeBench")
    policy_benchmarks: tuple[str, ...] = ("ResearchQA", "UltraFeedback")

    def __post_init__(self) -> None:
        if self.alignment_weight != 0.1:
            raise ValueError("the paper sets alignment_weight to 0.1")
        if self.rubric_sets_per_query != 4:
            raise ValueError("the paper sets K to 4")
        if self.responses_per_query != 4:
            raise ValueError("the paper sets G to 4")
        if self.learning_rate != 1.0e-6:
            raise ValueError("the paper sets the learning rate to 1e-6")
        if self.kl_coefficient != 0.001:
            raise ValueError("the paper sets the KL coefficient to 0.001")
        if self.alternating_stages != 10:
            raise ValueError("the paper uses 10 alternating stages")
        if self.update_steps_per_stage != 20:
            raise ValueError("the paper uses 20 update steps before switching")
        if self.train_prompts_per_step != 64:
            raise ValueError("the paper uses 64 training prompts per update step")
        if self.validation_prompts_per_step != 64:
            raise ValueError("the paper uses a separate 64-prompt validation batch")
        if self.lora_rank != 16:
            raise ValueError("the paper uses LoRA rank 16")
        expected_strings = {
            "rubric_generator_backbone": "Qwen3-4B",
            "answer_policy_backbone": "Qwen3-4B",
            "frozen_judge": "Qwen3-8B",
            "frozen_evaluator_encoder": "FsfairX-LLaMA3-RM-v0.1",
            "training_dataset": "Tulu 3",
            "coarse_preference_dataset": "HelpSteer 1",
        }
        for field, expected in expected_strings.items():
            if getattr(self, field) != expected:
                raise ValueError(f"the paper sets {field} to {expected}")
        if self.coarse_preference_dimensions != (
            "helpfulness",
            "correctness",
            "coherence",
        ):
            raise ValueError("the paper uses three specified coarse dimensions")
        if self.evaluation_judges != (
            "DeepSeek V4-Flash",
            "GPT-5.6-Luna",
            "Gemini 3.7-Flash",
        ):
            raise ValueError("evaluation judges differ from the paper")
        if self.rubric_benchmarks != ("RewardBench 2", "JudgeBench"):
            raise ValueError("rubric benchmarks differ from the paper")
        if self.policy_benchmarks != ("ResearchQA", "UltraFeedback"):
            raise ValueError("policy benchmarks differ from the paper")

    @property
    def total_update_steps(self) -> int:
        return self.alternating_stages * self.update_steps_per_stage

    @property
    def update_steps_per_module(self) -> int:
        return self.total_update_steps // 2

    @property
    def total_train_prompt_slots(self) -> int:
        return self.total_update_steps * self.train_prompts_per_step

    @property
    def total_validation_prompt_slots(self) -> int:
        return self.total_update_steps * self.validation_prompts_per_step

    def stage_plan(self, *, first_module: ModuleName) -> tuple[TrainingStage, ...]:
        """Build the 10-stage alternation without inventing an initial module."""
        if first_module not in ("rubric_generator", "answer_policy"):
            raise ValueError(f"unsupported first module: {first_module}")
        second_module: ModuleName = (
            "answer_policy"
            if first_module == "rubric_generator"
            else "rubric_generator"
        )
        modules = (first_module, second_module)
        return tuple(
            TrainingStage(
                index=index,
                module=modules[index % 2],
                update_steps=self.update_steps_per_stage,
                train_prompts_per_step=self.train_prompts_per_step,
                validation_prompts_per_step=self.validation_prompts_per_step,
            )
            for index in range(self.alternating_stages)
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


PAPER_CONFIG = PaperConfig()
