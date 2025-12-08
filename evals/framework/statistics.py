"""
Statistical analysis functions for TutorBench evaluations.

Implements confidence intervals, significance testing, and multi-run aggregation
following the statistical standards from the TutorBench paper.
"""

import numpy as np
from typing import List, Dict, Any
from scipy import stats


def compute_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute confidence interval for scores using t-distribution.

    Args:
        scores: List of scores from multiple runs
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, margin) where margin is the ± value

    Example:
        >>> scores = [0.52, 0.54, 0.53]
        >>> mean, margin = compute_confidence_interval(scores)
        >>> print(f"Score: {mean:.2%} ± {margin:.2%}")
        Score: 53.00% ± 1.00%
    """
    if len(scores) == 0:
        return 0.0, 0.0

    if len(scores) == 1:
        return scores[0], 0.0

    mean = np.mean(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(confidence, len(scores) - 1, loc=mean, scale=sem)
    margin = (ci[1] - ci[0]) / 2

    return float(mean), float(margin)


def aggregate_multiple_runs(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics across multiple evaluation runs.

    This computes mean scores, confidence intervals, and aggregates
    all breakdowns (by use case, dimension, skill, etc.) across runs.

    Args:
        reports: List of individual run reports from generate_full_report()

    Returns:
        Dictionary with aggregated statistics and confidence intervals
    """
    if not reports:
        raise ValueError("No reports provided for aggregation")

    n_runs = len(reports)

    # Extract overall scores
    overall_scores = [r["overall"]["overall_score"] for r in reports]
    mean_score, ci_margin = compute_confidence_interval(overall_scores)

    # Aggregate by modality
    text_only_scores = []
    multimodal_scores = []
    for r in reports:
        if r["by_modality"]["text_only"]:
            text_only_scores.append(r["by_modality"]["text_only"]["overall_score"])
        if r["by_modality"]["multimodal"]:
            multimodal_scores.append(r["by_modality"]["multimodal"]["overall_score"])

    text_mean, text_ci = compute_confidence_interval(text_only_scores) if text_only_scores else (None, None)
    mm_mean, mm_ci = compute_confidence_interval(multimodal_scores) if multimodal_scores else (None, None)

    # Aggregate by use case
    use_case_stats = {}
    use_cases = set()
    for r in reports:
        use_cases.update(r["by_use_case"].keys())

    for use_case in use_cases:
        uc_scores = [
            r["by_use_case"][use_case]["overall_score"]
            for r in reports if use_case in r["by_use_case"]
        ]
        uc_mean, uc_ci = compute_confidence_interval(uc_scores)
        use_case_stats[use_case] = {
            "mean": uc_mean,
            "ci_margin": uc_ci,
            "scores": uc_scores
        }

    # Aggregate by dimension
    dimension_stats = {}
    dimensions = set()
    for r in reports:
        dimensions.update(r["by_dimension"].keys())

    for dim in dimensions:
        dim_pass_rates = [
            r["by_dimension"][dim]["pass_rate"]
            for r in reports if dim in r["by_dimension"]
        ]
        dim_mean, dim_ci = compute_confidence_interval(dim_pass_rates)
        dimension_stats[dim] = {
            "mean_pass_rate": dim_mean,
            "ci_margin": dim_ci,
            "pass_rates": dim_pass_rates
        }

    # Aggregate by skill
    skill_stats = {}
    skills = set()
    for r in reports:
        skills.update(r["by_skill"].keys())

    for skill in skills:
        skill_pass_rates = [
            r["by_skill"][skill]["pass_rate"]
            for r in reports if skill in r["by_skill"]
        ]
        skill_mean, skill_ci = compute_confidence_interval(skill_pass_rates)
        skill_stats[skill] = {
            "mean_pass_rate": skill_mean,
            "ci_margin": skill_ci,
            "pass_rates": skill_pass_rates
        }

    # Aggregate by subject
    subject_stats = {}
    subjects = set()
    for r in reports:
        subjects.update(r["by_subject"].keys())

    for subject in subjects:
        subj_scores = [
            r["by_subject"][subject]["overall_score"]
            for r in reports if subject in r["by_subject"]
        ]
        subj_mean, subj_ci = compute_confidence_interval(subj_scores)
        subject_stats[subject] = {
            "mean": subj_mean,
            "ci_margin": subj_ci,
            "scores": subj_scores
        }

    return {
        "n_runs": n_runs,
        "metadata": reports[0]["metadata"],  # Use first run's metadata

        # Overall statistics
        "overall_score_mean": mean_score,
        "overall_score_ci_margin": ci_margin,
        "overall_score_std": float(np.std(overall_scores)),
        "overall_score_median": float(np.median(overall_scores)),
        "overall_score_min": float(np.min(overall_scores)),
        "overall_score_max": float(np.max(overall_scores)),
        "all_runs": overall_scores,

        # By modality
        "text_only_mean": text_mean,
        "text_only_ci_margin": text_ci,
        "multimodal_mean": mm_mean,
        "multimodal_ci_margin": mm_ci,

        # Breakdowns
        "by_use_case": use_case_stats,
        "by_dimension": dimension_stats,
        "by_skill": skill_stats,
        "by_subject": subject_stats,

        # Individual run details
        "individual_runs": reports
    }


def compute_statistical_significance(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform t-test to determine if two sets of scores are significantly different.

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            "error": "Need at least 2 samples per group for significance testing"
        }

    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant": p_value < alpha,
        "alpha": alpha,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "interpretation": (
            f"Difference is {'statistically significant' if p_value < alpha else 'not significant'} "
            f"at α={alpha} (p={p_value:.4f})"
        )
    }
