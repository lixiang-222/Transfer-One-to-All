from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from transferone.config import PAPER_CONFIG, PaperConfig
from transferone.demo import run_demo
from transferone.evaluator import CoarsePreferenceEvaluator, pinball_loss
from transferone.grpo import cosine_similarity, grpo_objective
from transferone.method import (
    build_iteration_signals,
    interval_pairwise_advantages,
    response_score_advantages,
)
from transferone.schedule import run_paper_schedule


class MethodTests(unittest.TestCase):
    def test_response_score_advantages_are_row_local(self) -> None:
        scores = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]]])
        advantages = response_score_advantages(scores, epsilon=1e-6)
        self.assertAlmostEqual(float(advantages[0, 0].mean()), 0.0, places=6)
        self.assertTrue(torch.equal(advantages[0, 1], torch.zeros(3)))

    def test_interval_weight_matches_paper_example(self) -> None:
        advantages = interval_pairwise_advantages(
            torch.tensor([[0.75, 1.00]]),
            torch.tensor([[0.50, 0.70]]),
            torch.tensor([[1.00, 1.30]]),
            epsilon=1e-6,
        )
        self.assertTrue(
            torch.allclose(advantages, torch.tensor([[-0.4, 0.4]]), atol=1e-6)
        )

    def test_nested_intervals_have_zero_pair_weight(self) -> None:
        advantages = interval_pairwise_advantages(
            torch.tensor([[2.5, 2.2]]),
            torch.tensor([[2.0, 1.0]]),
            torch.tensor([[3.0, 3.5]]),
            epsilon=1e-6,
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))

    def test_additive_reward_and_positive_selection(self) -> None:
        judge_scores = torch.tensor(
            [
                [[3.0, 2.0, 1.0]] * 4,
                [[1.0, 2.0, 3.0]] * 4,
            ]
        )
        base_rewards = torch.tensor(
            [[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]]
        )
        candidate_gradients = torch.tensor(
            [
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[-1.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, -1.0]],
            ]
        )
        signals = build_iteration_signals(
            judge_scores,
            base_rewards,
            candidate_gradients,
            torch.tensor([1.0, 0.0]),
            epsilon=1e-6,
            alignment_weight=0.1,
        )
        expected = base_rewards + 0.1 * signals.alignment_rewards
        self.assertTrue(torch.allclose(signals.total_rubric_rewards, expected))
        self.assertEqual(signals.eligible_queries.tolist(), [True, False])
        self.assertEqual(signals.selected_rubric_indices.tolist(), [0, -1])
        self.assertTrue(
            torch.equal(
                signals.selected_response_advantages[1],
                torch.zeros(3),
            )
        )


class GrpoTests(unittest.TestCase):
    def test_objective_uses_response_length_normalization(self) -> None:
        log_probs = torch.zeros(2, 3)
        objective = grpo_objective(
            log_probs,
            log_probs,
            log_probs,
            torch.tensor([1.0, -0.5]),
            torch.tensor([[True, True, False], [True, True, True]]),
            clip_delta=0.2,
            kl_coefficient=0.001,
        )
        self.assertAlmostEqual(float(objective), 0.25, places=6)

    def test_zero_gradient_alignment_is_zero(self) -> None:
        value = cosine_similarity(torch.zeros(2), torch.ones(2))
        self.assertEqual(float(value), 0.0)

    def test_nonzero_gradient_has_no_extra_threshold(self) -> None:
        tiny = torch.tensor([1e-12, 0.0])
        value = cosine_similarity(tiny, tiny)
        self.assertAlmostEqual(float(value), 1.0)


class EvaluatorTests(unittest.TestCase):
    def test_pinball_loss(self) -> None:
        residual = torch.tensor([-1.0, 1.0])
        self.assertAlmostEqual(float(pinball_loss(residual, 0.25)), 0.5)

    def test_two_stage_evaluator_fit_and_freeze(self) -> None:
        torch.manual_seed(3)
        train_x = torch.randn(32, 3)
        train_y = 2.0 * train_x[:, 0] - train_x[:, 1]
        calibration_x = torch.randn(32, 3)
        calibration_y = 2.0 * calibration_x[:, 0] - calibration_x[:, 1]
        evaluator = CoarsePreferenceEvaluator(3, hidden_dim=8, alpha=0.2)
        score_losses = evaluator.fit_score_head(
            train_x,
            train_y,
            steps=50,
            learning_rate=0.03,
        )
        quantile_losses = evaluator.fit_quantile_heads(
            calibration_x,
            calibration_y,
            steps=50,
            learning_rate=0.03,
        )
        prediction = evaluator.freeze().interval(calibration_x[:4])
        lower_offset = evaluator.lower_quantile_head(calibration_x[:4]).squeeze(-1)
        upper_offset = evaluator.upper_quantile_head(calibration_x[:4]).squeeze(-1)
        self.assertLess(score_losses[-1], score_losses[0])
        self.assertLess(quantile_losses[-1], quantile_losses[0])
        self.assertEqual(prediction.center.shape, (4,))
        self.assertTrue(
            torch.allclose(prediction.lower, prediction.center + lower_offset)
        )
        self.assertTrue(
            torch.allclose(prediction.upper, prediction.center + upper_offset)
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in evaluator.parameters()))

    def test_quantile_outputs_are_not_reordered(self) -> None:
        evaluator = CoarsePreferenceEvaluator(2, hidden_dim=2, alpha=0.2)
        with torch.no_grad():
            for parameter in evaluator.parameters():
                parameter.zero_()
            evaluator.lower_quantile_head.layers[-1].bias.fill_(1.0)
            evaluator.upper_quantile_head.layers[-1].bias.fill_(-1.0)
        prediction = evaluator.interval(torch.zeros(1, 2))
        self.assertGreater(
            float(prediction.lower[0].detach()),
            float(prediction.upper[0].detach()),
        )


class PaperConfigurationTests(unittest.TestCase):
    def test_reported_training_contract(self) -> None:
        self.assertEqual(PAPER_CONFIG.alternating_stages, 10)
        self.assertEqual(PAPER_CONFIG.update_steps_per_stage, 20)
        self.assertEqual(PAPER_CONFIG.train_prompts_per_step, 64)
        self.assertEqual(PAPER_CONFIG.validation_prompts_per_step, 64)
        self.assertEqual(PAPER_CONFIG.rubric_sets_per_query, 4)
        self.assertEqual(PAPER_CONFIG.responses_per_query, 4)
        self.assertEqual(PAPER_CONFIG.total_update_steps, 200)
        self.assertEqual(PAPER_CONFIG.update_steps_per_module, 100)
        self.assertEqual(PAPER_CONFIG.total_train_prompt_slots, 12_800)
        self.assertEqual(PAPER_CONFIG.total_validation_prompt_slots, 12_800)

        plan = PAPER_CONFIG.stage_plan(first_module="answer_policy")
        self.assertEqual(len(plan), 10)
        self.assertTrue(all(stage.update_steps == 20 for stage in plan))
        self.assertTrue(all(stage.train_prompts_per_step == 64 for stage in plan))
        self.assertTrue(
            all(stage.validation_prompts_per_step == 64 for stage in plan)
        )
        self.assertTrue(
            all(plan[index].module != plan[index + 1].module for index in range(9))
        )

    def test_scheduler_executes_every_reported_update_slot(self) -> None:
        contexts = []
        results = run_paper_schedule(
            lambda context: contexts.append(context) or context.global_update_step,
            first_module="answer_policy",
        )
        self.assertEqual(len(results), 200)
        self.assertEqual(results, list(range(200)))
        self.assertEqual(
            sum(context.active_module == "answer_policy" for context in contexts),
            100,
        )
        self.assertEqual(
            sum(context.active_module == "rubric_generator" for context in contexts),
            100,
        )
        self.assertTrue(all(context.train_prompts == 64 for context in contexts))
        self.assertTrue(
            all(context.validation_prompts == 64 for context in contexts)
        )

    def test_reported_values_cannot_be_silently_overridden(self) -> None:
        with self.assertRaises(ValueError):
            PaperConfig(alternating_stages=2)
        with self.assertRaises(ValueError):
            PaperConfig(train_prompts_per_step=3)
        with self.assertRaises(ValueError):
            PaperConfig(frozen_judge="another-model")

    def test_yaml_matches_code_and_marks_unreported_values(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((root / "configs/paper.yaml").read_text())
        self.assertEqual(payload["method"]["alignment_weight"], 0.1)
        self.assertEqual(payload["method"]["rubric_sets_per_query"], 4)
        self.assertEqual(payload["method"]["responses_per_query"], 4)
        self.assertEqual(payload["optimization"]["learning_rate"], 1e-6)
        self.assertEqual(payload["optimization"]["kl_coefficient"], 0.001)
        self.assertEqual(payload["optimization"]["alternating_stages"], 10)
        self.assertEqual(payload["optimization"]["update_steps_per_stage"], 20)
        self.assertEqual(payload["optimization"]["train_prompts_per_step"], 64)
        self.assertEqual(
            payload["optimization"]["validation_prompts_per_step"],
            64,
        )
        self.assertEqual(payload["models"]["lora_rank"], PAPER_CONFIG.lora_rank)
        self.assertEqual(
            tuple(payload["data"]["coarse_preference_dimensions"]),
            PAPER_CONFIG.coarse_preference_dimensions,
        )
        self.assertEqual(
            tuple(payload["evaluation"]["judges"]),
            PAPER_CONFIG.evaluation_judges,
        )
        self.assertTrue(all(value is None for value in payload["not_reported"].values()))


class DemoTests(unittest.TestCase):
    def test_demo_runs_without_data_models_or_network(self) -> None:
        result = run_demo()
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["eligible_queries"], 1)
        self.assertEqual(result["shape"]["training_queries"], 64)
        self.assertEqual(result["shape"]["validation_queries"], 64)
        self.assertEqual(result["shape"]["rubric_sets_per_query"], 4)
        self.assertEqual(result["shape"]["responses_per_query"], 4)
        self.assertEqual(result["paper_schedule"]["alternating_stages"], 10)
        self.assertEqual(result["paper_schedule"]["update_steps_per_stage"], 20)
        self.assertEqual(result["paper_schedule"]["total_update_steps"], 200)
        self.assertEqual(result["paper_schedule"]["total_train_prompt_slots"], 12_800)
        self.assertEqual(
            result["paper_schedule"]["total_validation_prompt_slots"],
            12_800,
        )


if __name__ == "__main__":
    unittest.main()
