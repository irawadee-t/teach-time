"""
TutorBench evaluation system.

Faithful implementation of the TutorBench benchmark from Scale AI.
Reference: https://arxiv.org/abs/2510.02663
"""

from .models import (
    Rubric,
    Sample,
    EvaluationResult,
    RubricRating,
    UseCase,
    EvaluationDimension,
    TutoringSkill,
)
from .scoring import (
    compute_weighted_score,
    aggregate_scores,
    aggregate_by_dimension,
    aggregate_by_skill,
)
from .judge import TutorBenchJudge
from .runner import evaluate_model, evaluate_model_async, evaluate_model_concurrent
from .prompts import get_system_prompt
from .data_loader import (
    load_samples_from_json,
    load_samples_from_hf,
    save_results_to_json,
)
from .providers import get_provider, ConcurrencyConfig, get_async_provider
from .chains import (
    ChainedPipeline,
    StageTemplate,
    ChainedPipelineResult,
    create_tutoring_chain,
    create_custom_chain,
    compare_single_vs_chain,
    TUTORING_STAGES,
)
from .best_of_n import (
    BestOfNPipeline,
    BestOfNResult,
    create_best_of_n_pipeline,
    save_bon_debug_log,
)

__all__ = [
    # Core models
    "Rubric",
    "Sample",
    "EvaluationResult",
    "RubricRating",
    "UseCase",
    "EvaluationDimension",
    "TutoringSkill",
    # Scoring
    "compute_weighted_score",
    "aggregate_scores",
    "aggregate_by_dimension",
    "aggregate_by_skill",
    # Evaluation
    "TutorBenchJudge",
    "evaluate_model",
    "get_system_prompt",
    # Data loading
    "load_samples_from_json",
    "load_samples_from_hf",
    "save_results_to_json",
    # Providers
    "get_provider",
    # Async components
    "evaluate_model_async",
    "evaluate_model_concurrent",
    "ConcurrencyConfig",
    "get_async_provider",
    # Chained pipelines
    "ChainedPipeline",
    "StageTemplate",
    "ChainedPipelineResult",
    "create_tutoring_chain",
    "create_custom_chain",
    "compare_single_vs_chain",
    "TUTORING_STAGES",
    # Best-of-N pipelines
    "BestOfNPipeline",
    "BestOfNResult",
    "create_best_of_n_pipeline",
    "save_bon_debug_log",
]
