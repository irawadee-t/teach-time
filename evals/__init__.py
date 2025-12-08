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
from .runner import evaluate_model, generate_response
from .prompts import get_system_prompt
from .data_loader import (
    load_samples_from_json,
    load_samples_from_hf,
    save_results_to_json,
)

__all__ = [
    "Rubric",
    "Sample",
    "EvaluationResult",
    "RubricRating",
    "UseCase",
    "EvaluationDimension",
    "TutoringSkill",
    "compute_weighted_score",
    "aggregate_scores",
    "aggregate_by_dimension",
    "aggregate_by_skill",
    "TutorBenchJudge",
    "evaluate_model",
    "generate_response",
    "get_system_prompt",
    "load_samples_from_json",
    "load_samples_from_hf",
    "save_results_to_json",
]
