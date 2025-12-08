# TutorBench Evaluation Framework

This framework provides statistically rigorous evaluation capabilities on top of the core TutorBench implementation, following the scientific best practices from the [TutorBench paper](https://arxiv.org/abs/2510.02663).

## Features

- ✅ **Multi-run evaluations** with confidence intervals
- ✅ **Comprehensive reporting** with breakdowns by:
  - Use case (Adaptive, Assessment, Active Learning)
  - Modality (Text-only, Multimodal)
  - Subject (Biology, Physics, Chemistry, Statistics, Calculus, CS)
  - Evaluation dimension (8 dimensions)
  - Tutoring skill (8 skills)
- ✅ **Experiment tracking** for reproducibility
- ✅ **Leaderboard management** for comparing models
- ✅ **Statistical significance testing**

## Quick Start

```python
from evals.framework import run_evaluation, ExperimentConfig

# 1. Define your model function
def my_model(system_prompt: str, messages: list) -> str:
    # Your model API call here
    return response

# 2. Configure experiment
config = ExperimentConfig(
    model_name="gpt-4",
    model_version="gpt-4-2024-01-15",
    temperature=0.0,
    seed=42
)

# 3. Run evaluation
results = run_evaluation(
    model_fn=my_model,
    config=config,
    use_hf=True,  # Load from HuggingFace
    n_runs=3,     # 3 runs for confidence intervals
    output_dir="results"
)
```

## Output Structure

```
results/
├── gpt-4/
│   └── gpt-4_20250112_143022/
│       ├── config.json          # Experiment configuration
│       ├── run_1.json           # Individual run results
│       ├── run_2.json
│       ├── run_3.json
│       └── final_report.json    # Aggregated statistics with CI
└── leaderboard.json             # Cross-model comparison
```

## Modules

### `experiment.py`
- `ExperimentConfig`: Configuration dataclass for tracking experiment parameters

### `statistics.py`
- `compute_confidence_interval()`: Compute 95% CI using t-distribution
- `aggregate_multiple_runs()`: Aggregate stats across runs
- `compute_statistical_significance()`: T-test for model comparison

### `reporting.py`
- `generate_full_report()`: Create comprehensive report with all breakdowns
- `print_summary()`: Human-readable summary output
- `save_leaderboard_entry()`: Update leaderboard.json

### `run_evaluation.py`
- `run_evaluation()`: Main orchestration function

## Example Report Format

```json
{
  "n_runs": 3,
  "overall_score_mean": 0.5234,
  "overall_score_ci_margin": 0.011,
  "text_only_mean": 0.5456,
  "multimodal_mean": 0.5034,
  "by_use_case": {
    "adaptive": {"mean": 0.4712, "ci_margin": 0.015},
    "assessment": {"mean": 0.5234, "ci_margin": 0.012},
    "active_learning": {"mean": 0.5678, "ci_margin": 0.010}
  },
  "by_dimension": { ... },
  "by_skill": { ... },
  "by_subject": { ... }
}
```

## Scientific Best Practices Implemented

Following the TutorBench paper methodology:

1. ✅ **Multiple runs with CI**: 3+ runs with 95% confidence intervals
2. ✅ **Comprehensive breakdowns**: All 8 dimensions, 8 skills, 3 use cases
3. ✅ **Modality split**: Separate text-only and multimodal analysis
4. ✅ **Reproducibility**: Full parameter tracking
5. ✅ **Leaderboard format**: Matches Table 1 from paper

## See Also

- [TutorBench Paper](https://arxiv.org/abs/2510.02663)
- [HuggingFace Dataset](https://huggingface.co/datasets/ScaleAI/tutorbench_sample)
- Core implementation: `../` (runner.py, judge.py, scoring.py, etc.)
