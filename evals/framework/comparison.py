"""Comparison utilities for baseline vs chain evaluations."""

import json
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats as scipy_stats

from ..scoring import aggregate_scores, aggregate_by_dimension, aggregate_by_skill


def compute_delta(baseline: float, chain: float) -> float:
    """Compute absolute difference."""
    return chain - baseline


def compute_improvement_pct(baseline: float, chain: float) -> float:
    """Compute percentage improvement."""
    if baseline == 0:
        return float('inf') if chain > 0 else 0.0
    return ((chain - baseline) / baseline) * 100


def significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def format_score_with_ci(mean: float, ci_margin: float = None) -> str:
    """Format score with optional CI margin."""
    if ci_margin is None or ci_margin == 0:
        return f"{mean:6.2%}"
    return f"{mean:6.2%} +/- {ci_margin:4.1%}"


def format_delta(delta: float, show_plus: bool = True) -> str:
    """Format delta with + sign."""
    sign = "+" if delta >= 0 and show_plus else ""
    return f"{sign}{delta:6.2%}"


def compute_comparison_stats(
    baseline_stats,
    chain_stats,
    is_multi_run: bool = False,
) -> Dict[str, Any]:
    """
    Compute deltas, improvements, and statistical tests.

    Args:
        baseline_stats: Either List[EvaluationResult] (single) or dict (multi)
        chain_stats: Either List[EvaluationResult] (single) or dict (multi)
        is_multi_run: Whether this is multi-run comparison

    Returns:
        Dict with baseline, chain, deltas, and statistical tests
    """
    if is_multi_run:
        return _compute_multi_run_comparison(baseline_stats, chain_stats)
    return _compute_single_run_comparison(baseline_stats, chain_stats)


def _compute_multi_run_comparison(baseline_stats: Dict, chain_stats: Dict) -> Dict:
    """Compute comparison for multi-run evaluations."""
    baseline_overall = baseline_stats["overall_score_mean"]
    baseline_ci = baseline_stats.get("overall_score_ci_margin", 0)
    chain_overall = chain_stats["overall_score_mean"]
    chain_ci = chain_stats.get("overall_score_ci_margin", 0)

    delta = compute_delta(baseline_overall, chain_overall)
    improvement = compute_improvement_pct(baseline_overall, chain_overall)

    # Statistical significance test
    t_stat, p_value, sig = None, None, ""
    if "all_runs" in baseline_stats and "all_runs" in chain_stats:
        baseline_runs = baseline_stats["all_runs"]
        chain_runs = chain_stats["all_runs"]
        if len(baseline_runs) >= 2 and len(chain_runs) >= 2:
            t_stat, p_value = scipy_stats.ttest_ind(baseline_runs, chain_runs)
            sig = significance_stars(p_value)

    comparison = {
        "metadata": {"is_multi_run": True, "n_runs": baseline_stats.get("n_runs", 1)},
        "overall": {
            "baseline": {"mean": baseline_overall, "ci_margin": baseline_ci},
            "chain": {"mean": chain_overall, "ci_margin": chain_ci},
            "delta": delta,
            "improvement_pct": improvement,
            "p_value": p_value,
            "t_stat": t_stat,
            "significance": sig,
        },
        "by_use_case": {},
        "by_dimension": {},
        "by_skill": {},
    }

    # By use case
    for uc in baseline_stats.get("by_use_case", {}):
        if uc in chain_stats.get("by_use_case", {}):
            b_uc = baseline_stats["by_use_case"][uc]
            c_uc = chain_stats["by_use_case"][uc]
            comparison["by_use_case"][uc] = _compute_breakdown_stats(b_uc, c_uc, is_multi_run=True)

    # By dimension
    for dim in baseline_stats.get("by_dimension", {}):
        if dim in chain_stats.get("by_dimension", {}):
            b_dim = baseline_stats["by_dimension"][dim]
            c_dim = chain_stats["by_dimension"][dim]
            comparison["by_dimension"][dim] = {
                "baseline": {"mean": b_dim["mean_pass_rate"], "ci_margin": b_dim.get("ci_margin", 0)},
                "chain": {"mean": c_dim["mean_pass_rate"], "ci_margin": c_dim.get("ci_margin", 0)},
                "delta": compute_delta(b_dim["mean_pass_rate"], c_dim["mean_pass_rate"]),
                "improvement_pct": compute_improvement_pct(b_dim["mean_pass_rate"], c_dim["mean_pass_rate"]),
            }

    # By skill
    for skill in baseline_stats.get("by_skill", {}):
        if skill in chain_stats.get("by_skill", {}):
            b_skill = baseline_stats["by_skill"][skill]
            c_skill = chain_stats["by_skill"][skill]
            comparison["by_skill"][skill] = {
                "baseline": {"mean": b_skill["mean_pass_rate"], "ci_margin": b_skill.get("ci_margin", 0)},
                "chain": {"mean": c_skill["mean_pass_rate"], "ci_margin": c_skill.get("ci_margin", 0)},
                "delta": compute_delta(b_skill["mean_pass_rate"], c_skill["mean_pass_rate"]),
                "improvement_pct": compute_improvement_pct(b_skill["mean_pass_rate"], c_skill["mean_pass_rate"]),
            }

    return comparison


def _compute_breakdown_stats(baseline_uc: Dict, chain_uc: Dict, is_multi_run: bool) -> Dict:
    """Compute stats for a single use case/dimension/skill breakdown."""
    result = {
        "baseline": {"mean": baseline_uc["mean"], "ci_margin": baseline_uc.get("ci_margin", 0)},
        "chain": {"mean": chain_uc["mean"], "ci_margin": chain_uc.get("ci_margin", 0)},
        "delta": compute_delta(baseline_uc["mean"], chain_uc["mean"]),
        "improvement_pct": compute_improvement_pct(baseline_uc["mean"], chain_uc["mean"]),
    }

    if is_multi_run and "scores" in baseline_uc and "scores" in chain_uc:
        if len(baseline_uc["scores"]) >= 2 and len(chain_uc["scores"]) >= 2:
            t_stat, p_value = scipy_stats.ttest_ind(baseline_uc["scores"], chain_uc["scores"])
            result["p_value"] = p_value
            result["significance"] = significance_stars(p_value)

    return result


def _compute_single_run_comparison(baseline_results: List, chain_results: List) -> Dict:
    """Compute comparison for single-run evaluations."""
    baseline_agg = aggregate_scores(baseline_results)
    chain_agg = aggregate_scores(chain_results)

    baseline_overall = baseline_agg["overall_score"]
    chain_overall = chain_agg["overall_score"]

    comparison = {
        "metadata": {"is_multi_run": False, "n_samples": len(baseline_results)},
        "overall": {
            "baseline": {"mean": baseline_overall},
            "chain": {"mean": chain_overall},
            "delta": compute_delta(baseline_overall, chain_overall),
            "improvement_pct": compute_improvement_pct(baseline_overall, chain_overall),
        },
        "by_use_case": {},
        "by_dimension": {},
        "by_skill": {},
    }

    # By dimension
    baseline_by_dim = aggregate_by_dimension(baseline_results)
    chain_by_dim = aggregate_by_dimension(chain_results)
    for dim in baseline_by_dim:
        if dim in chain_by_dim and baseline_by_dim[dim]["n_rubrics"] > 0:
            b_pr = baseline_by_dim[dim]["pass_rate"]
            c_pr = chain_by_dim[dim]["pass_rate"]
            comparison["by_dimension"][dim.value] = {
                "baseline": {"mean": b_pr},
                "chain": {"mean": c_pr},
                "delta": compute_delta(b_pr, c_pr),
                "improvement_pct": compute_improvement_pct(b_pr, c_pr),
                "n_rubrics": baseline_by_dim[dim]["n_rubrics"],
            }

    # By skill
    baseline_by_skill = aggregate_by_skill(baseline_results)
    chain_by_skill = aggregate_by_skill(chain_results)
    for skill in baseline_by_skill:
        if skill in chain_by_skill and baseline_by_skill[skill]["n_rubrics"] > 0:
            b_pr = baseline_by_skill[skill]["pass_rate"]
            c_pr = chain_by_skill[skill]["pass_rate"]
            comparison["by_skill"][skill.value] = {
                "baseline": {"mean": b_pr},
                "chain": {"mean": c_pr},
                "delta": compute_delta(b_pr, c_pr),
                "improvement_pct": compute_improvement_pct(b_pr, c_pr),
                "n_rubrics": baseline_by_skill[skill]["n_rubrics"],
            }

    return comparison


def print_comparison_table(comparison: Dict, is_multi_run: bool = False) -> None:
    """Print formatted comparison table to console."""
    print("\n" + "=" * 80)
    print("BASELINE VS CHAIN COMPARISON")
    print("=" * 80)

    # Overall
    print("\n" + "-" * 80)
    print("OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<20} {'Chain':<20} {'Delta':<12} {'Improvement'}")
    print("-" * 80)

    overall = comparison["overall"]
    b_str = format_score_with_ci(overall["baseline"]["mean"], overall["baseline"].get("ci_margin"))
    c_str = format_score_with_ci(overall["chain"]["mean"], overall["chain"].get("ci_margin"))
    d_str = format_delta(overall["delta"])
    i_str = f"+{overall['improvement_pct']:.1f}%" if overall["improvement_pct"] != float('inf') else "N/A"
    sig = f" {overall.get('significance', '')}" if is_multi_run else ""
    print(f"{'Overall Score':<30} {b_str:<20} {c_str:<20} {d_str:<12} {i_str}{sig}")

    # By use case
    if comparison.get("by_use_case"):
        print("\n" + "-" * 80)
        print("BY USE CASE")
        print("-" * 80)
        for uc, stats in sorted(comparison["by_use_case"].items()):
            _print_breakdown_row(uc, stats, is_multi_run)

    # By dimension
    if comparison.get("by_dimension"):
        print("\n" + "-" * 80)
        print("BY EVALUATION DIMENSION (Pass Rate)")
        print("-" * 80)
        for dim, stats in sorted(comparison["by_dimension"].items(), key=lambda x: x[1]["delta"], reverse=True):
            _print_breakdown_row(dim[:38], stats, is_multi_run, width=40)

    # By skill
    if comparison.get("by_skill"):
        print("\n" + "-" * 80)
        print("BY TUTORING SKILL (Pass Rate)")
        print("-" * 80)
        for skill, stats in sorted(comparison["by_skill"].items(), key=lambda x: x[1]["delta"], reverse=True):
            _print_breakdown_row(skill[:38], stats, is_multi_run, width=40)

    # Significance legend
    if is_multi_run and overall.get("p_value") is not None:
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE")
        print("=" * 80)
        print(f"\nOverall: p = {overall['p_value']:.4f} {overall.get('significance', '')}")
        print("\nLegend: *** p < 0.001, ** p < 0.01, * p < 0.05")

    print("=" * 80)


def _print_breakdown_row(name: str, stats: Dict, is_multi_run: bool, width: int = 30) -> None:
    """Print a single breakdown row."""
    b_str = format_score_with_ci(stats["baseline"]["mean"], stats["baseline"].get("ci_margin"))
    c_str = format_score_with_ci(stats["chain"]["mean"], stats["chain"].get("ci_margin"))
    d_str = format_delta(stats["delta"])
    i_str = f"+{stats['improvement_pct']:.1f}%" if stats["improvement_pct"] != float('inf') else "N/A"
    sig = f" {stats.get('significance', '')}" if is_multi_run else ""
    print(f"{name:<{width}} {b_str:<20} {c_str:<20} {d_str:<12} {i_str}{sig}")


def save_comparison_report(comparison: Dict, output_path) -> None:
    """Save comparison report to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison report saved to: {output_path}")
