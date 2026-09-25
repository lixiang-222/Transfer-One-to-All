# Paper Alignment

This package implements the method stated in Sections 4.2 through 4.4 and
Algorithm 1.

## Included

- Row-wise response-score normalization from Equation 1
- The full token-level GRPO objective from Equation 2
- Center-score regression and residual quantile regression from Equations 6 and 7
- Interval-overlap weights and pairwise advantages from Equations 8 and 9
- Reference and candidate policy-gradient construction
- Cosine alignment rewards from Equation 11
- Additive baseline and alignment rewards from Equation 12
- Strictly positive alignment filtering
- Best-aligned rubric-set selection for the policy update
- Detached alignment rewards with no second-order differentiation
- The Section A.3 values enforced by `PaperConfig`
- All 10 x 20 update slots exposed by `run_paper_schedule`

## Deliberately External

- Dataset loading and preprocessing
- Qwen and reward-model checkpoints
- LoRA, distributed training, and rollout engines
- Frozen-judge prompting
- EvoLM and Rubric-ARM definitions of `R_base`
- Benchmark evaluation

These components are framework-specific and do not change the proposed
alignment method.

## Unreported Parameters

The paper does not report a numerical evaluator quantile level, GRPO clipping
threshold, normalization epsilon, evaluator hidden dimensions, evaluator
training length, evaluator learning rate, or aggregation rule for the three
HelpSteer 1 dimensions. The code therefore does not label any value for these
fields as a paper default. They must be supplied explicitly by a model backend.
The data-free smoke demo declares its own numerical choices separately.

The evaluator returns `m + q_l` and `m + q_h` directly. The method code does not
sort crossed quantiles, clip exponential probability ratios, add a near-zero
gradient threshold, or reject a group only because its reward variance is
small. These behaviors are absent from the paper.

## Excluded Experiment Infrastructure

The package does not include hard-case replay, checkpoint acceptance gates,
cluster scheduling, telemetry, API evaluation, or internal filesystem
contracts. Those mechanisms are not part of Algorithm 1.
