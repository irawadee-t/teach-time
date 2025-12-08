"""
Run TutorBench evaluation on full dataset with specified provider.

Output Structure:
    Results are saved to: results/{provider}/{model}_{run_type}_{timestamp}/

    Run types:
        - baseline: Single-call evaluation (default)
        - chain: Multi-stage chained pipeline
        - bon{N}: Best-of-N sampling (e.g., bon5)
        - comparison: Baseline vs chain comparison

    Examples:
        results/together/default_baseline_20251207_183834/
        results/anthropic/claude-sonnet-4_chain_20251207_190000/
        results/together/default_bon5_20251207_191000/

    For comparisons: results/{provider}/{model}_comparison_{timestamp}/
        ├── baseline/           # Baseline results
        ├── chain/              # Chain results
        └── comparison_report.json

Usage:
    # Single run (quick)
    python run_eval.py --provider together
    python run_eval.py --provider anthropic --model claude-sonnet-4-20250514

    # Chained pipeline (multi-stage reasoning)
    python run_eval.py --provider anthropic --use-chain
    python run_eval.py --provider together --use-chain --chain-safety-check --chain-verbose

    # Compare baseline vs chain (single run)
    python run_eval.py --provider anthropic --compare --samples 50
    python run_eval.py --provider together --compare --chain-safety-check

    # Compare baseline vs chain (multi-run with statistical analysis)
    python run_eval.py --provider anthropic --compare --n-runs 3 --samples 100
    python run_eval.py --provider together --compare --n-runs 5

    # Multi-run with statistical analysis (rigorous)
    python run_eval.py --provider together --n-runs 3
    python run_eval.py --provider anthropic --n-runs 5 --samples 100

    # Multi-run with chained pipeline
    python run_eval.py --provider anthropic --use-chain --n-runs 3 --samples 50

    # Advanced options
    python run_eval.py --provider together --samples 10 --max-concurrent 30
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env from root or evals directory
if Path(".env").exists():
    load_dotenv()
elif Path("evals/.env").exists():
    load_dotenv("evals/.env")
else:
    load_dotenv()  # Try default anyway

from evals import (
    load_samples_from_hf,
    TutorBenchJudge,
    evaluate_model,
    aggregate_scores,
    aggregate_by_dimension,
    save_results_to_json,
    get_provider,
    ConcurrencyConfig,
    create_tutoring_chain,
)
from evals.scoring import aggregate_by_skill
from scipy import stats as scipy_stats

# Import framework for multi-run evaluations
from evals.framework import (
    ExperimentConfig,
    generate_full_report,
    aggregate_multiple_runs,
    print_summary,
    save_leaderboard_entry,
)


def get_model_identifier(provider: str, model: str = None) -> str:
    """
    Get a clean model identifier for directory naming.

    Args:
        provider: Provider name (together, anthropic, openai)
        model: Optional model name

    Returns:
        Clean identifier like 'llama-3-70b' or 'default'
    """
    if not model:
        return "default"
    # Clean up model name for filesystem (remove slashes, etc.)
    clean_model = model.replace("/", "_").replace(":", "_")
    return clean_model


def get_output_dir(base_dir: str, provider: str, model: str = None, suffix: str = None) -> Path:
    """
    Generate structured output directory path.

    Structure: results/{provider}/{model_name}_{timestamp}/

    Args:
        base_dir: Base results directory
        provider: Provider name
        model: Model name (optional)
        suffix: Optional suffix (e.g., 'chain', 'bon5', 'comparison')

    Returns:
        Path to output directory
    """
    model_id = get_model_identifier(provider, model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if suffix:
        dir_name = f"{model_id}_{suffix}_{timestamp}"
    else:
        dir_name = f"{model_id}_{timestamp}"

    output_path = Path(base_dir) / provider / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def run_single_evaluation(args, samples, model_fn, judge, concurrency_config, output_dir=None):
    """
    Run a single evaluation (legacy behavior, fast).

    Args:
        args: Command-line arguments
        samples: List of samples to evaluate
        model_fn: Model function
        judge: TutorBenchJudge instance
        concurrency_config: Concurrency configuration
        output_dir: Optional output directory (if None, generates from args)

    Returns the results list.
    """
    print(f"\n🚀 Running single evaluation...")
    print(f"   Processing {len(samples)} samples...\n")

    results = evaluate_model(
        samples=samples,
        model_fn=model_fn,
        judge=judge,
        model_name=f"{args.provider}/{args.model or 'default'}",
        verbose=True,
        concurrency_config=concurrency_config,
    )

    # Determine output directory
    if output_dir is None:
        # Determine suffix based on pipeline mode
        if args.use_chain:
            suffix = "chain"
        elif args.use_best_of_n:
            suffix = f"bon{args.best_of_n_count}"
        else:
            suffix = "baseline"

        output_dir = get_output_dir(
            base_dir=args.output_dir,
            provider=args.provider,
            model=args.model,
            suffix=suffix,
        )

    print(f"📁 Output directory: {output_dir}")

    # Save results
    results_path = output_dir / "results.json"
    print(f"\n💾 Saving results to {results_path}...")
    save_results_to_json(results, str(results_path))

    # Save chain debug log if enabled
    if args.chain_debug and hasattr(model_fn, 'get_chain_outputs'):
        from evals.chains import save_chain_debug_log
        chain_outputs = model_fn.get_chain_outputs()
        debug_path = output_dir / "chain_debug.json"
        save_chain_debug_log(chain_outputs, samples, results, str(debug_path))

    # Save best-of-n debug log if enabled
    if args.best_of_n_debug and hasattr(model_fn, 'get_bon_outputs'):
        from evals.best_of_n import save_bon_debug_log
        bon_outputs = model_fn.get_bon_outputs()
        debug_path = output_dir / "bon_debug.json"
        save_bon_debug_log(bon_outputs, samples, results, str(debug_path))

    # Print quick summary
    print_single_run_summary(results)

    return results


def run_multi_evaluation(args, samples, model_fn, judge, concurrency_config, output_dir=None):
    """
    Run multiple evaluations with statistical analysis (rigorous, slower).

    Uses the framework to compute confidence intervals and comprehensive reports.

    Args:
        args: Command-line arguments
        samples: List of samples to evaluate
        model_fn: Model function
        judge: TutorBenchJudge instance
        concurrency_config: Concurrency configuration
        output_dir: Optional output directory (if None, generates from args)
    """
    print(f"\n🚀 Running {args.n_runs} evaluations for statistical rigor...")
    print(f"   Processing {len(samples)} samples × {args.n_runs} runs")
    print(f"   This will take approximately {args.n_runs}x longer than single run\n")

    # Determine suffix based on pipeline mode
    if args.use_chain:
        suffix = "chain"
    elif args.use_best_of_n:
        suffix = f"bon{args.best_of_n_count}"
    else:
        suffix = "baseline"

    # Get model identifier for config
    model_id = get_model_identifier(args.provider, args.model)

    # Create experiment config with full model identifier
    config = ExperimentConfig(
        model_name=model_id,
        model_version=args.model or "default",
        temperature=0.0,  # Default for reproducibility
        use_async=True,
        max_samples=args.samples,
        notes=f"Run via run_eval.py with {args.n_runs} runs"
    )

    # Setup output directory
    if output_dir is None:
        output_dir = get_output_dir(
            base_dir=args.output_dir,
            provider=args.provider,
            model=args.model,
            suffix=suffix,
        )
    run_path = output_dir

    # Save config
    config.save(run_path / "config.json")
    print(f"📁 Output directory: {run_path}\n")

    # Run multiple evaluations
    all_reports = []
    for run_idx in range(args.n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{args.n_runs}")
        print(f"{'='*70}\n")

        # Run evaluation
        results = evaluate_model(
            samples=samples,
            model_fn=model_fn,
            judge=judge,
            model_name=f"{args.provider}/{args.model or 'default'}",
            verbose=True,
            concurrency_config=concurrency_config,
        )

        # Generate comprehensive report
        report = generate_full_report(results, samples, config)

        # Save individual run
        run_file = run_path / f"run_{run_idx + 1}.json"
        import json
        with open(run_file, 'w') as f:
            json.dump(report, f, indent=2)

        all_reports.append(report)
        print(f"\n✓ Run {run_idx + 1} Score: {report['overall']['overall_score']:.2%}")
        print(f"  Saved to: {run_file}")

    # Aggregate across runs
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    final_stats = aggregate_multiple_runs(all_reports)

    # Save final aggregated results
    final_file = run_path / "final_report.json"
    import json
    with open(final_file, 'w') as f:
        json.dump(final_stats, f, indent=2)

    print(f"✓ Final report saved to: {final_file}")

    # Update leaderboard (at root results directory level)
    if args.save_leaderboard:
        leaderboard_path = Path(args.output_dir) / "leaderboard.json"
        save_leaderboard_entry(final_stats, leaderboard_path)

    # Print comprehensive summary
    print_summary(final_stats, verbose=True)

    return final_stats


def print_single_run_summary(results):
    """Print summary for single run (legacy format)."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    overall = aggregate_scores(results)
    print(f"\n📊 Overall Score: {overall['overall_score']:.1%}")
    print(f"📈 Mean Pass Rate: {overall['mean_pass_rate']:.1%}")
    print(f"📉 Std Dev: {overall['std_score']:.1%}")
    print(f"📋 Samples: {overall['n_samples']}")
    print(f"📝 Total Rubrics: {overall['total_rubrics']}")

    # By dimension
    print("\n" + "-" * 70)
    print("BY EVALUATION DIMENSION")
    print("-" * 70)
    by_dim = aggregate_by_dimension(results)
    for dim, metrics in sorted(by_dim.items(), key=lambda x: x[1]['pass_rate'], reverse=True):
        if metrics['n_rubrics'] > 0:
            print(f"{dim.value:30s}: {metrics['pass_rate']:5.1%} ({metrics['n_rubrics']} rubrics)")

    # Top failures
    print("\n" + "-" * 70)
    print("SAMPLES WITH LOWEST SCORES")
    print("-" * 70)
    sorted_results = sorted(results, key=lambda r: r.weighted_score)
    for result in sorted_results[:5]:
        passed = sum(r.passed for r in result.rubric_ratings)
        total = len(result.rubric_ratings)
        sample_id_short = result.sample_id[:24]
        print(f"{sample_id_short:24s} | Score: {result.weighted_score:5.1%} | Pass: {passed}/{total}")

    print("\n" + "=" * 70)


# ============================================================================
# Baseline vs Chain Comparison Functions
# ============================================================================

def _compute_delta(baseline_val, chain_val):
    """Compute absolute difference."""
    return chain_val - baseline_val


def _compute_improvement_pct(baseline_val, chain_val):
    """Compute percentage improvement."""
    if baseline_val == 0:
        return float('inf') if chain_val > 0 else 0.0
    return ((chain_val - baseline_val) / baseline_val) * 100


def _format_score_with_ci(mean, ci_margin=None):
    """Format score with optional CI margin."""
    if ci_margin is None or ci_margin == 0:
        return f"{mean:6.2%}"
    return f"{mean:6.2%} ± {ci_margin:4.1%}"


def _format_delta(delta, show_plus=True):
    """Format delta with + sign."""
    sign = "+" if delta >= 0 and show_plus else ""
    return f"{sign}{delta:6.2%}"


def _significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def compute_comparison_stats(baseline_stats, chain_stats, is_multi_run=False):
    """
    Compute deltas, improvements, and statistical tests.

    Args:
        baseline_stats: Either List[EvaluationResult] (single) or dict (multi)
        chain_stats: Either List[EvaluationResult] (single) or dict (multi)
        is_multi_run: bool - Whether this is multi-run comparison

    Returns:
        dict with baseline, chain, deltas, and statistical tests
    """
    if is_multi_run:
        # Multi-run mode: stats are already aggregated dicts
        baseline_overall = baseline_stats["overall_score_mean"]
        baseline_ci = baseline_stats.get("overall_score_ci_margin", 0)
        chain_overall = chain_stats["overall_score_mean"]
        chain_ci = chain_stats.get("overall_score_ci_margin", 0)

        # Compute delta and improvement
        delta = _compute_delta(baseline_overall, chain_overall)
        improvement = _compute_improvement_pct(baseline_overall, chain_overall)

        # Statistical significance test
        if "all_runs" in baseline_stats and "all_runs" in chain_stats:
            baseline_runs = baseline_stats["all_runs"]
            chain_runs = chain_stats["all_runs"]
            if len(baseline_runs) >= 2 and len(chain_runs) >= 2:
                t_stat, p_value = scipy_stats.ttest_ind(baseline_runs, chain_runs)
                significance = _significance_stars(p_value)
            else:
                t_stat, p_value, significance = None, None, ""
        else:
            t_stat, p_value, significance = None, None, ""

        comparison = {
            "metadata": {
                "is_multi_run": True,
                "n_runs": baseline_stats.get("n_runs", 1),
            },
            "overall": {
                "baseline": {"mean": baseline_overall, "ci_margin": baseline_ci},
                "chain": {"mean": chain_overall, "ci_margin": chain_ci},
                "delta": delta,
                "improvement_pct": improvement,
                "p_value": p_value,
                "t_stat": t_stat,
                "significance": significance,
            },
            "by_use_case": {},
            "by_dimension": {},
            "by_skill": {},
        }

        # By use case
        for use_case in baseline_stats.get("by_use_case", {}).keys():
            if use_case in chain_stats.get("by_use_case", {}):
                baseline_uc = baseline_stats["by_use_case"][use_case]
                chain_uc = chain_stats["by_use_case"][use_case]
                uc_delta = _compute_delta(baseline_uc["mean"], chain_uc["mean"])
                uc_improvement = _compute_improvement_pct(baseline_uc["mean"], chain_uc["mean"])

                # Statistical test for use case
                if "scores" in baseline_uc and "scores" in chain_uc:
                    if len(baseline_uc["scores"]) >= 2 and len(chain_uc["scores"]) >= 2:
                        t_stat, p_value = scipy_stats.ttest_ind(baseline_uc["scores"], chain_uc["scores"])
                        uc_sig = _significance_stars(p_value)
                    else:
                        t_stat, p_value, uc_sig = None, None, ""
                else:
                    t_stat, p_value, uc_sig = None, None, ""

                comparison["by_use_case"][use_case] = {
                    "baseline": {"mean": baseline_uc["mean"], "ci_margin": baseline_uc.get("ci_margin", 0)},
                    "chain": {"mean": chain_uc["mean"], "ci_margin": chain_uc.get("ci_margin", 0)},
                    "delta": uc_delta,
                    "improvement_pct": uc_improvement,
                    "p_value": p_value,
                    "significance": uc_sig,
                }

        # By dimension (pass rates)
        for dim in baseline_stats.get("by_dimension", {}).keys():
            if dim in chain_stats.get("by_dimension", {}):
                baseline_dim = baseline_stats["by_dimension"][dim]
                chain_dim = chain_stats["by_dimension"][dim]
                dim_delta = _compute_delta(baseline_dim["mean_pass_rate"], chain_dim["mean_pass_rate"])
                dim_improvement = _compute_improvement_pct(baseline_dim["mean_pass_rate"], chain_dim["mean_pass_rate"])

                comparison["by_dimension"][dim] = {
                    "baseline": {"mean": baseline_dim["mean_pass_rate"], "ci_margin": baseline_dim.get("ci_margin", 0)},
                    "chain": {"mean": chain_dim["mean_pass_rate"], "ci_margin": chain_dim.get("ci_margin", 0)},
                    "delta": dim_delta,
                    "improvement_pct": dim_improvement,
                }

        # By skill (pass rates)
        for skill in baseline_stats.get("by_skill", {}).keys():
            if skill in chain_stats.get("by_skill", {}):
                baseline_skill = baseline_stats["by_skill"][skill]
                chain_skill = chain_stats["by_skill"][skill]
                skill_delta = _compute_delta(baseline_skill["mean_pass_rate"], chain_skill["mean_pass_rate"])
                skill_improvement = _compute_improvement_pct(baseline_skill["mean_pass_rate"], chain_skill["mean_pass_rate"])

                comparison["by_skill"][skill] = {
                    "baseline": {"mean": baseline_skill["mean_pass_rate"], "ci_margin": baseline_skill.get("ci_margin", 0)},
                    "chain": {"mean": chain_skill["mean_pass_rate"], "ci_margin": chain_skill.get("ci_margin", 0)},
                    "delta": skill_delta,
                    "improvement_pct": skill_improvement,
                }

    else:
        # Single-run mode: stats are Lists of EvaluationResults
        baseline_agg = aggregate_scores(baseline_stats)
        chain_agg = aggregate_scores(chain_stats)

        baseline_overall = baseline_agg["overall_score"]
        chain_overall = chain_agg["overall_score"]

        delta = _compute_delta(baseline_overall, chain_overall)
        improvement = _compute_improvement_pct(baseline_overall, chain_overall)

        comparison = {
            "metadata": {
                "is_multi_run": False,
                "n_samples": len(baseline_stats),
            },
            "overall": {
                "baseline": {"mean": baseline_overall},
                "chain": {"mean": chain_overall},
                "delta": delta,
                "improvement_pct": improvement,
            },
            "by_use_case": {},
            "by_dimension": {},
            "by_skill": {},
        }

        # By use case - skip for single-run mode (would need to match back to samples)
        # TODO: Add use case breakdown for single-run mode

        # By dimension
        baseline_by_dim = aggregate_by_dimension(baseline_stats)
        chain_by_dim = aggregate_by_dimension(chain_stats)

        for dim in baseline_by_dim.keys():
            if dim in chain_by_dim and baseline_by_dim[dim]["n_rubrics"] > 0:
                baseline_pass_rate = baseline_by_dim[dim]["pass_rate"]
                chain_pass_rate = chain_by_dim[dim]["pass_rate"]
                dim_delta = _compute_delta(baseline_pass_rate, chain_pass_rate)
                dim_improvement = _compute_improvement_pct(baseline_pass_rate, chain_pass_rate)

                comparison["by_dimension"][dim.value] = {
                    "baseline": {"mean": baseline_pass_rate},
                    "chain": {"mean": chain_pass_rate},
                    "delta": dim_delta,
                    "improvement_pct": dim_improvement,
                    "n_rubrics": baseline_by_dim[dim]["n_rubrics"],
                }

        # By skill
        baseline_by_skill = aggregate_by_skill(baseline_stats)
        chain_by_skill = aggregate_by_skill(chain_stats)

        for skill in baseline_by_skill.keys():
            if skill in chain_by_skill and baseline_by_skill[skill]["n_rubrics"] > 0:
                baseline_pass_rate = baseline_by_skill[skill]["pass_rate"]
                chain_pass_rate = chain_by_skill[skill]["pass_rate"]
                skill_delta = _compute_delta(baseline_pass_rate, chain_pass_rate)
                skill_improvement = _compute_improvement_pct(baseline_pass_rate, chain_pass_rate)

                comparison["by_skill"][skill.value] = {
                    "baseline": {"mean": baseline_pass_rate},
                    "chain": {"mean": chain_pass_rate},
                    "delta": skill_delta,
                    "improvement_pct": skill_improvement,
                    "n_rubrics": baseline_by_skill[skill]["n_rubrics"],
                }

    return comparison


def print_comparison_table(comparison_dict, is_multi_run=False):
    """
    Print formatted comparison table to console.

    Args:
        comparison_dict: Output from compute_comparison_stats()
        is_multi_run: bool - Whether to show CI margins and significance
    """
    print("\n" + "=" * 80)
    print("BASELINE VS CHAIN COMPARISON")
    print("=" * 80)

    # Overall performance
    print("\n" + "-" * 80)
    print("OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<20} {'Chain':<20} {'Delta':<12} {'Improvement'}")
    print("-" * 80)

    overall = comparison_dict["overall"]
    baseline_str = _format_score_with_ci(overall["baseline"]["mean"], overall["baseline"].get("ci_margin"))
    chain_str = _format_score_with_ci(overall["chain"]["mean"], overall["chain"].get("ci_margin"))
    delta_str = _format_delta(overall["delta"])
    improvement_str = f"+{overall['improvement_pct']:.1f}%" if overall["improvement_pct"] != float('inf') else "N/A"

    sig_str = f" {overall.get('significance', '')}" if is_multi_run else ""
    print(f"{'Overall Score':<30} {baseline_str:<20} {chain_str:<20} {delta_str:<12} {improvement_str}{sig_str}")

    # By use case
    if comparison_dict.get("by_use_case"):
        print("\n" + "-" * 80)
        print("BY USE CASE")
        print("-" * 80)
        print(f"{'Use Case':<30} {'Baseline':<20} {'Chain':<20} {'Delta':<12} {'Improvement'}")
        print("-" * 80)

        for use_case, stats in sorted(comparison_dict["by_use_case"].items()):
            baseline_str = _format_score_with_ci(stats["baseline"]["mean"], stats["baseline"].get("ci_margin"))
            chain_str = _format_score_with_ci(stats["chain"]["mean"], stats["chain"].get("ci_margin"))
            delta_str = _format_delta(stats["delta"])
            improvement_str = f"+{stats['improvement_pct']:.1f}%" if stats["improvement_pct"] != float('inf') else "N/A"
            sig_str = f" {stats.get('significance', '')}" if is_multi_run else ""

            print(f"{use_case:<30} {baseline_str:<20} {chain_str:<20} {delta_str:<12} {improvement_str}{sig_str}")

    # By dimension (pass rates)
    if comparison_dict.get("by_dimension"):
        print("\n" + "-" * 80)
        print("BY EVALUATION DIMENSION (Pass Rate)")
        print("-" * 80)
        print(f"{'Dimension':<40} {'Baseline':<20} {'Chain':<20} {'Delta':<12} {'Improvement'}")
        print("-" * 80)

        for dim, stats in sorted(comparison_dict["by_dimension"].items(), key=lambda x: x[1]["delta"], reverse=True):
            baseline_str = _format_score_with_ci(stats["baseline"]["mean"], stats["baseline"].get("ci_margin"))
            chain_str = _format_score_with_ci(stats["chain"]["mean"], stats["chain"].get("ci_margin"))
            delta_str = _format_delta(stats["delta"])
            improvement_str = f"+{stats['improvement_pct']:.1f}%" if stats["improvement_pct"] != float('inf') else "N/A"

            dim_display = dim[:38] if len(dim) > 38 else dim
            print(f"{dim_display:<40} {baseline_str:<20} {chain_str:<20} {delta_str:<12} {improvement_str}")

    # By skill (pass rates)
    if comparison_dict.get("by_skill"):
        print("\n" + "-" * 80)
        print("BY TUTORING SKILL (Pass Rate)")
        print("-" * 80)
        print(f"{'Skill':<40} {'Baseline':<20} {'Chain':<20} {'Delta':<12} {'Improvement'}")
        print("-" * 80)

        for skill, stats in sorted(comparison_dict["by_skill"].items(), key=lambda x: x[1]["delta"], reverse=True):
            baseline_str = _format_score_with_ci(stats["baseline"]["mean"], stats["baseline"].get("ci_margin"))
            chain_str = _format_score_with_ci(stats["chain"]["mean"], stats["chain"].get("ci_margin"))
            delta_str = _format_delta(stats["delta"])
            improvement_str = f"+{stats['improvement_pct']:.1f}%" if stats["improvement_pct"] != float('inf') else "N/A"

            skill_display = skill[:38] if len(skill) > 38 else skill
            print(f"{skill_display:<40} {baseline_str:<20} {chain_str:<20} {delta_str:<12} {improvement_str}")

    # Statistical significance (multi-run only)
    if is_multi_run and overall.get("p_value") is not None:
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE")
        print("=" * 80)

        p_value = overall["p_value"]
        sig = overall.get("significance", "")
        print(f"\nOverall: p = {p_value:.4f} {sig}")

        if sig:
            print(f"  - Baseline: {_format_score_with_ci(overall['baseline']['mean'], overall['baseline'].get('ci_margin'))}")
            print(f"  - Chain:    {_format_score_with_ci(overall['chain']['mean'], overall['chain'].get('ci_margin'))}")
            print(f"  - Effect:   Chain improves by {overall['improvement_pct']:.1f}%")

        # Show significant use cases
        sig_use_cases = [(uc, stats) for uc, stats in comparison_dict.get("by_use_case", {}).items()
                         if stats.get("significance")]
        if sig_use_cases:
            print("\nUse cases with significant improvements:")
            for uc, stats in sig_use_cases:
                p_val = stats.get("p_value", 0)
                sig = stats.get("significance", "")
                print(f"  {uc:<20} p = {p_val:.4f} {sig}")

        print("\nLegend: *** p < 0.001, ** p < 0.01, * p < 0.05")

    print("=" * 80)


def save_comparison_report(comparison_dict, output_path):
    """
    Save comparison report to JSON file.

    Args:
        comparison_dict: Output from compute_comparison_stats()
        output_path: Path - Where to save the JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(comparison_dict, f, indent=2)

    print(f"\n💾 Comparison report saved to: {output_path}")


def run_comparison_evaluation(args, samples, base_provider, judge, concurrency_config):
    """
    Run both baseline and chain evaluations, then print comparison.

    Args:
        args: Command-line arguments
        samples: List[Sample] - TutorBench samples
        base_provider: Callable - Base provider (NOT wrapped in chain)
        judge: TutorBenchJudge - Judge instance
        concurrency_config: ConcurrencyConfig - Concurrency settings

    Returns:
        tuple of (baseline_stats, chain_stats, comparison_dict)
    """
    is_multi_run = args.n_runs > 1

    # Create comparison output directory
    model_id = get_model_identifier(args.provider, args.model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = Path(args.output_dir) / args.provider / f"{model_id}_comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Comparison output directory: {comparison_dir}")

    # Create subdirectories for baseline and chain
    baseline_dir = comparison_dir / "baseline"
    chain_dir = comparison_dir / "chain"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    chain_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # BASELINE EVALUATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 1/2: BASELINE EVALUATION")
    print("=" * 70)

    if is_multi_run:
        # Multi-run mode
        baseline_stats = run_multi_evaluation(
            args=args,
            samples=samples,
            model_fn=base_provider,
            judge=judge,
            concurrency_config=concurrency_config,
            output_dir=baseline_dir,
        )
    else:
        # Single-run mode
        baseline_results = run_single_evaluation(
            args=args,
            samples=samples,
            model_fn=base_provider,
            judge=judge,
            concurrency_config=concurrency_config,
            output_dir=baseline_dir,
        )
        baseline_stats = baseline_results

    print(f"\n✓ Baseline evaluation complete")

    # ========================================================================
    # CHAIN EVALUATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2/2: CHAIN EVALUATION")
    print("=" * 70)

    # Create chain wrapper
    print(f"\n🔗 Creating chained pipeline...")
    print(f"   Stages: {'6 (with safety check)' if args.chain_safety_check else '5 (no safety check)'}")
    print(f"   Verbose: {args.chain_verbose}")
    if args.chain_debug:
        print(f"   Debug logging: ENABLED")

    chain = create_tutoring_chain(
        base_provider=base_provider,
        include_safety_check=args.chain_safety_check,
        verbose=args.chain_verbose,
    )
    chain_model_fn = chain.create_model_fn(log_chain_outputs=args.chain_debug)

    if is_multi_run:
        # Multi-run mode
        chain_stats = run_multi_evaluation(
            args=args,
            samples=samples,
            model_fn=chain_model_fn,
            judge=judge,
            concurrency_config=concurrency_config,
            output_dir=chain_dir,
        )
    else:
        # Single-run mode
        chain_results = run_single_evaluation(
            args=args,
            samples=samples,
            model_fn=chain_model_fn,
            judge=judge,
            concurrency_config=concurrency_config,
            output_dir=chain_dir,
        )
        chain_stats = chain_results

    print(f"\n✓ Chain evaluation complete")

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "=" * 70)
    print("COMPUTING COMPARISON")
    print("=" * 70)

    comparison = compute_comparison_stats(
        baseline_stats=baseline_stats,
        chain_stats=chain_stats,
        is_multi_run=is_multi_run,
    )

    # Print comparison table
    print_comparison_table(comparison, is_multi_run=is_multi_run)

    # Save comparison report (use the comparison_dir created earlier)
    comparison_path = comparison_dir / "comparison_report.json"
    save_comparison_report(comparison, comparison_path)

    return baseline_stats, chain_stats, comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run TutorBench evaluation on specified provider"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="together",
        choices=["together", "anthropic", "openai"],
        help="Model provider (default: together)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (uses provider default if not specified)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all 1,473)",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include multimodal samples (default: text-only)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base output directory. Results saved to: {output-dir}/{provider}/{model}_{run_type}_{timestamp}/ (default: results/)",
    )

    # Multi-run arguments
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of evaluation runs for computing confidence intervals (default: 1). Use 3+ for statistical rigor.",
    )
    parser.add_argument(
        "--save-leaderboard",
        action="store_true",
        help="Save results to leaderboard.json (only for multi-run)",
    )

    # Concurrency arguments
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum number of concurrent API calls (default: 50)",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=10,
        help="Number of samples to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--rubric-batch-size",
        type=int,
        default=5,
        help="Number of rubrics to evaluate concurrently per sample (default: 5)",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=int,
        default=50,
        help="Maximum API requests per second (default: 50)",
    )

    # Chained pipeline arguments (NEW)
    parser.add_argument(
        "--use-chain",
        action="store_true",
        help="Use chained LLM pipeline (6-stage tutoring) instead of single-call",
    )
    parser.add_argument(
        "--chain-safety-check",
        action="store_true",
        help="Include safety check stage in chain (6 stages instead of 5). Only used with --use-chain",
    )
    parser.add_argument(
        "--chain-verbose",
        action="store_true",
        help="Print intermediate chain outputs. Only used with --use-chain",
    )
    parser.add_argument(
        "--chain-debug",
        action="store_true",
        help="Save detailed chain debug log with all intermediate stage outputs per sample. Only used with --use-chain or --compare",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and chain evaluations, then compare results. Mutually exclusive with --use-chain",
    )

    # Best-of-N arguments
    parser.add_argument(
        "--use-best-of-n",
        action="store_true",
        help="Use best-of-n sampling (generate N responses, select best) instead of single-call",
    )
    parser.add_argument(
        "--best-of-n-count",
        type=int,
        default=5,
        help="Number of candidate responses to generate in best-of-n mode (default: 5). Only used with --use-best-of-n",
    )
    parser.add_argument(
        "--best-of-n-selector",
        type=str,
        default=None,
        help="Provider for selection (format: provider/model, e.g., 'anthropic/claude-sonnet-4'). Defaults to base provider. Only used with --use-best-of-n",
    )
    parser.add_argument(
        "--best-of-n-verbose",
        action="store_true",
        help="Print all candidate responses and selection reasoning. Only used with --use-best-of-n",
    )
    parser.add_argument(
        "--best-of-n-debug",
        action="store_true",
        help="Save detailed best-of-n debug log with all candidate responses per sample. Only used with --use-best-of-n",
    )
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.use_chain and args.compare:
        parser.error("--use-chain and --compare are mutually exclusive")

    if args.use_best_of_n and args.use_chain:
        parser.error("--use-best-of-n and --use-chain are mutually exclusive")

    if args.use_best_of_n and args.compare:
        parser.error("--use-best-of-n and --compare are mutually exclusive")

    if args.use_best_of_n and args.best_of_n_count < 2:
        parser.error("--best-of-n-count must be at least 2")

    print("=" * 70)
    print(f"TutorBench Evaluation: {args.provider.upper()}")
    if args.compare:
        print(f"Mode: COMPARISON (baseline vs chain)")
        if args.n_runs > 1:
            print(f"  Runs: {args.n_runs} per variant (total: {args.n_runs * 2})")
    elif args.n_runs > 1:
        print(f"Mode: MULTI-RUN ({args.n_runs} runs with statistical analysis)")
    else:
        print(f"Mode: SINGLE RUN (quick evaluation)")
    if args.use_chain:
        print(f"Pipeline: CHAINED ({6 if args.chain_safety_check else 5} stages)")
    elif args.use_best_of_n:
        print(f"Pipeline: BEST-OF-{args.best_of_n_count} (generate {args.best_of_n_count} responses, select best)")
    elif not args.compare:
        print(f"Pipeline: SINGLE-CALL (baseline)")
    print("=" * 70)

    # Load samples with filtering
    print(f"\n📥 Loading samples from HuggingFace...")
    text_only = not args.include_multimodal
    samples = load_samples_from_hf(
        dataset_name="ScaleAI/TutorBench",
        split="train",
        max_samples=args.samples,
        text_only=text_only,
    )

    print(f"✓ Loaded {len(samples)} {'text-only' if text_only else 'total'} samples")

    # Show sample distribution
    use_cases = {}
    subjects = {}
    for s in samples:
        use_cases[s.use_case.value] = use_cases.get(s.use_case.value, 0) + 1
        subjects[s.subject] = subjects.get(s.subject, 0) + 1

    print(f"\n📊 Sample distribution:")
    print(f"   Use cases: {dict(sorted(use_cases.items()))}")
    print(f"   Subjects: {dict(sorted(subjects.items(), key=lambda x: x[1], reverse=True))}")

    # Get model provider
    print(f"\n🤖 Initializing {args.provider} provider...")
    model_name = args.model or "default"
    try:
        base_provider = get_provider(args.provider, args.model)
        if args.model:
            print(f"   Model: {args.model}")
        else:
            print(f"   Using default model for {args.provider}")

        # Wrap in chained pipeline if requested
        if args.use_chain:
            print(f"\n🔗 Creating chained pipeline...")
            print(f"   Stages: {'6 (with safety check)' if args.chain_safety_check else '5 (no safety check)'}")
            print(f"   Verbose: {args.chain_verbose}")
            if args.chain_debug:
                print(f"   Debug logging: ENABLED")

            chain = create_tutoring_chain(
                base_provider=base_provider,
                include_safety_check=args.chain_safety_check,
                verbose=args.chain_verbose,
            )
            model_fn = chain.create_model_fn(log_chain_outputs=args.chain_debug)
            model_name = f"{model_name}-chain"

        # Wrap in best-of-n pipeline if requested
        elif args.use_best_of_n:
            print(f"\n🎲 Creating best-of-n pipeline...")
            print(f"   Candidates: {args.best_of_n_count}")

            # Parse selector provider if specified
            selection_provider = None
            if args.best_of_n_selector:
                parts = args.best_of_n_selector.split('/')
                selector_provider_name = parts[0]
                selector_model = parts[1] if len(parts) > 1 else None
                selection_provider = get_provider(selector_provider_name, selector_model)
                print(f"   Selector: {args.best_of_n_selector}")
            else:
                print(f"   Selector: Same as base provider")

            print(f"   Verbose: {args.best_of_n_verbose}")
            if args.best_of_n_debug:
                print(f"   Debug logging: ENABLED")

            from evals.best_of_n import create_best_of_n_pipeline
            bon_pipeline = create_best_of_n_pipeline(
                base_provider=base_provider,
                n=args.best_of_n_count,
                selection_provider=selection_provider,
                verbose=args.best_of_n_verbose,
            )
            model_fn = bon_pipeline.create_model_fn(log_bon_outputs=args.best_of_n_debug)
            model_name = f"{model_name}-bon{args.best_of_n_count}"

        else:
            model_fn = base_provider

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Initialize judge
    print(f"\n🔧 Initializing judge (Claude Sonnet 4)...")
    judge = TutorBenchJudge(verbose=False)

    # Setup concurrency configuration
    concurrency_config = ConcurrencyConfig(
        max_concurrent=args.max_concurrent,
        sample_batch_size=args.sample_batch_size,
        rubric_batch_size=args.rubric_batch_size,
        max_requests_per_second=args.max_requests_per_second,
    )

    print(f"\n⚡ Concurrent evaluation:")
    print(f"   Max concurrent calls: {args.max_concurrent}")
    print(f"   Sample batch size: {args.sample_batch_size}")
    print(f"   Rubric batch size: {args.rubric_batch_size}")
    print(f"   Max requests/sec: {args.max_requests_per_second}")

    # Route to appropriate evaluation mode
    if args.compare:
        # Comparison mode (baseline vs chain)
        run_comparison_evaluation(args, samples, base_provider, judge, concurrency_config)
    elif args.n_runs == 1:
        # Single run mode (legacy, fast)
        run_single_evaluation(args, samples, model_fn, judge, concurrency_config)
    else:
        # Multi-run mode (statistical, comprehensive)
        run_multi_evaluation(args, samples, model_fn, judge, concurrency_config)

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
