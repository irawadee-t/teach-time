"""
Enhanced reporting functions that build on top of existing scoring module.
"""

import json
from typing import List, Dict, Any
from pathlib import Path

# Import from parent evals package
from ..models import EvaluationResult, Sample, UseCase
from ..scoring import (
    aggregate_scores,
    aggregate_by_dimension,
    aggregate_by_skill
)
from .experiment import ExperimentConfig


def generate_full_report(
    results: List[EvaluationResult],
    samples: List[Sample],
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with all breakdowns.

    This extends the basic scoring functions to provide complete analysis
    including use case, subject, modality breakdowns as required by the paper.

    Args:
        results: List of evaluation results from evaluate_model()
        samples: Original samples (needed for use_case, subject, modality info)
        config: Experiment configuration

    Returns:
        Complete report dictionary with all metrics and breakdowns
    """
    # Create lookup for sample metadata
    sample_lookup = {s.sample_id: s for s in samples}

    # Overall metrics (using existing function)
    overall = aggregate_scores(results)

    # Split by modality
    text_only_results = []
    multimodal_results = []
    for result in results:
        sample = sample_lookup.get(result.sample_id)
        if sample:
            if sample.is_multimodal:
                multimodal_results.append(result)
            else:
                text_only_results.append(result)

    # Split by use case
    by_use_case = {}
    for use_case in UseCase:
        uc_results = [
            r for r in results
            if sample_lookup.get(r.sample_id) and sample_lookup[r.sample_id].use_case == use_case
        ]
        if uc_results:
            by_use_case[use_case.value] = aggregate_scores(uc_results)

    # Split by subject
    subjects = set(s.subject for s in samples)
    by_subject = {}
    for subject in subjects:
        subj_results = [
            r for r in results
            if sample_lookup.get(r.sample_id) and sample_lookup[r.sample_id].subject == subject
        ]
        if subj_results:
            by_subject[subject] = aggregate_scores(subj_results)

    # By dimension and skill (using existing functions)
    by_dimension = aggregate_by_dimension(results)
    by_skill = aggregate_by_skill(results)

    # Convert enum keys to strings for JSON serialization
    by_dimension_str = {k.value: v for k, v in by_dimension.items()}
    by_skill_str = {k.value: v for k, v in by_skill.items()}

    return {
        "metadata": config.to_dict(),
        "overall": overall,
        "by_modality": {
            "text_only": aggregate_scores(text_only_results) if text_only_results else None,
            "multimodal": aggregate_scores(multimodal_results) if multimodal_results else None
        },
        "by_use_case": by_use_case,
        "by_dimension": by_dimension_str,
        "by_skill": by_skill_str,
        "by_subject": by_subject,
        "sample_results": [
            {
                "sample_id": r.sample_id,
                "weighted_score": r.weighted_score,
                "pass_rate": r.pass_rate,
                "critical_pass_rate": r.critical_rubric_pass_rate
            }
            for r in results
        ]
    }


def print_summary(stats: Dict[str, Any], verbose: bool = True):
    """
    Print human-readable summary of evaluation results.

    Args:
        stats: Aggregated statistics from aggregate_multiple_runs()
        verbose: Whether to print detailed breakdowns
    """
    metadata = stats["metadata"]

    print("\n" + "=" * 70)
    print("TUTORBENCH EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nModel: {metadata['model_name']}")
    print(f"Version: {metadata['model_version']}")
    print(f"Runs: {stats['n_runs']}")
    print(f"Timestamp: {metadata['timestamp']}")

    print("\n" + "-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)
    print(f"Score: {stats['overall_score_mean']:.2%} ± {stats['overall_score_ci_margin']:.2%}")
    print(f"  Median: {stats['overall_score_median']:.2%}")
    print(f"  Range: [{stats['overall_score_min']:.2%}, {stats['overall_score_max']:.2%}]")
    print(f"  Std: {stats['overall_score_std']:.4f}")

    if stats['text_only_mean'] is not None:
        print(f"\nText-only: {stats['text_only_mean']:.2%} ± {stats['text_only_ci_margin']:.2%}")
    if stats['multimodal_mean'] is not None:
        print(f"Multimodal: {stats['multimodal_mean']:.2%} ± {stats['multimodal_ci_margin']:.2%}")

    print(f"\nAll runs: {[f'{s:.2%}' for s in stats['all_runs']]}")

    if verbose and "by_use_case" in stats:
        print("\n" + "-" * 70)
        print("BY USE CASE")
        print("-" * 70)
        for use_case, uc_stats in stats["by_use_case"].items():
            print(f"{use_case:20s}: {uc_stats['mean']:.2%} ± {uc_stats['ci_margin']:.2%}")

    if verbose and "by_subject" in stats:
        print("\n" + "-" * 70)
        print("BY SUBJECT")
        print("-" * 70)
        for subject, subj_stats in stats["by_subject"].items():
            print(f"{subject:20s}: {subj_stats['mean']:.2%} ± {subj_stats['ci_margin']:.2%}")

    if verbose and "by_dimension" in stats:
        print("\n" + "-" * 70)
        print("BY EVALUATION DIMENSION (Pass Rate)")
        print("-" * 70)
        # Sort by pass rate descending
        sorted_dims = sorted(
            stats["by_dimension"].items(),
            key=lambda x: x[1]["mean_pass_rate"],
            reverse=True
        )
        for dim, dim_stats in sorted_dims:
            print(f"{dim:30s}: {dim_stats['mean_pass_rate']:.2%} ± {dim_stats['ci_margin']:.2%}")

    if verbose and "by_skill" in stats:
        print("\n" + "-" * 70)
        print("BY TUTORING SKILL (Pass Rate)")
        print("-" * 70)
        # Sort by pass rate descending
        sorted_skills = sorted(
            stats["by_skill"].items(),
            key=lambda x: x[1]["mean_pass_rate"],
            reverse=True
        )
        for skill, skill_stats in sorted_skills:
            print(f"{skill:40s}: {skill_stats['mean_pass_rate']:.2%} ± {skill_stats['ci_margin']:.2%}")

    print("\n" + "=" * 70 + "\n")


def save_leaderboard_entry(
    stats: Dict[str, Any],
    leaderboard_path: str = "results/leaderboard.json"
):
    """
    Save or update entry in leaderboard file.

    Args:
        stats: Aggregated statistics from aggregate_multiple_runs()
        leaderboard_path: Path to leaderboard JSON file
    """
    leaderboard_path = Path(leaderboard_path)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing leaderboard
    if leaderboard_path.exists():
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"entries": []}

    # Create new entry
    metadata = stats["metadata"]
    entry = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "run_id": metadata["run_id"],
        "timestamp": metadata["timestamp"],
        "overall_score": stats["overall_score_mean"],
        "ci_margin": stats["overall_score_ci_margin"],
        "text_only": stats.get("text_only_mean"),
        "multimodal": stats.get("multimodal_mean"),
        "n_runs": stats["n_runs"]
    }

    # Add or update entry
    existing_idx = None
    for idx, existing_entry in enumerate(leaderboard["entries"]):
        if existing_entry["run_id"] == entry["run_id"]:
            existing_idx = idx
            break

    if existing_idx is not None:
        leaderboard["entries"][existing_idx] = entry
    else:
        leaderboard["entries"].append(entry)

    # Sort by overall score descending
    leaderboard["entries"].sort(key=lambda x: x["overall_score"], reverse=True)

    # Save
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"Leaderboard updated: {leaderboard_path}")
