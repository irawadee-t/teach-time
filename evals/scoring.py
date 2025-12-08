"""
Scoring functions for TutorBench evaluation.

Implements the weighted average rubric rating (ARR_w) from the paper:

    ARR_j^w = Σ(w_i^j · r_i^j) / Σ(w_i^j · 1_{w_i^j > 0})

where:
    - N_j: number of rubric criteria for example j
    - w_i^j ∈ {-5, 1, 5}: weight of rubric i
    - r_i^j ∈ {0, 1}: pass/fail rating
"""

from typing import List, Dict
import numpy as np
from .models import EvaluationResult, RubricRating, EvaluationDimension, TutoringSkill


def compute_weighted_score(rubric_ratings: List[RubricRating]) -> float:
    """
    Compute weighted average rubric rating (ARR_w).

    Args:
        rubric_ratings: List of rubric ratings with pass/fail and weights

    Returns:
        Weighted score in [0, 1] range (normalized)

    Formula:
        ARR_w = Σ(w_i · r_i) / Σ(w_i · 1_{w_i > 0})
    """
    if not rubric_ratings:
        return 0.0

    # Extract weights and ratings
    weights = np.array([r.rubric.weight for r in rubric_ratings])
    ratings = np.array([int(r.passed) for r in rubric_ratings])

    # Numerator: weighted sum of ratings
    numerator = np.sum(weights * ratings)

    # Denominator: sum of positive weights only
    positive_weights = weights[weights > 0]
    denominator = np.sum(positive_weights)

    if denominator == 0:
        return 0.0

    # Compute weighted score
    score = numerator / denominator

    # Normalize to [0, 1] - already should be, but clip for safety
    return float(np.clip(score, 0.0, 1.0))


def aggregate_scores(results: List[EvaluationResult]) -> Dict[str, float]:
    """
    Aggregate scores across multiple samples.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with aggregated metrics
    """
    if not results:
        return {"overall_score": 0.0, "n_samples": 0}

    scores = [r.weighted_score for r in results]
    pass_rates = [r.pass_rate for r in results]

    return {
        "overall_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "mean_pass_rate": float(np.mean(pass_rates)),
        "n_samples": len(results),
        "total_rubrics": sum(len(r.rubric_ratings) for r in results),
    }


def aggregate_by_dimension(
    results: List[EvaluationResult],
) -> Dict[EvaluationDimension, Dict[str, float]]:
    """
    Aggregate pass rates by evaluation dimension.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary mapping dimension to metrics
    """
    dimension_ratings = {dim: [] for dim in EvaluationDimension}

    for result in results:
        for rating in result.rubric_ratings:
            dim = rating.rubric.evaluation_dimension
            dimension_ratings[dim].append(int(rating.passed))

    # Compute metrics per dimension
    metrics = {}
    for dim, ratings in dimension_ratings.items():
        if ratings:
            metrics[dim] = {
                "pass_rate": float(np.mean(ratings)),
                "n_rubrics": len(ratings),
            }
        else:
            metrics[dim] = {"pass_rate": 0.0, "n_rubrics": 0}

    return metrics


def aggregate_by_skill(
    results: List[EvaluationResult],
) -> Dict[TutoringSkill, Dict[str, float]]:
    """
    Aggregate pass rates by tutoring skill.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary mapping skill to metrics
    """
    skill_ratings = {skill: [] for skill in TutoringSkill}

    for result in results:
        for rating in result.rubric_ratings:
            if rating.rubric.tutoring_skill is not None:
                skill = rating.rubric.tutoring_skill
                skill_ratings[skill].append(int(rating.passed))

    # Compute metrics per skill
    metrics = {}
    for skill, ratings in skill_ratings.items():
        if ratings:
            metrics[skill] = {
                "pass_rate": float(np.mean(ratings)),
                "n_rubrics": len(ratings),
            }
        else:
            metrics[skill] = {"pass_rate": 0.0, "n_rubrics": 0}

    return metrics
