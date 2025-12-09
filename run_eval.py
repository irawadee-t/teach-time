"""
Run TutorBench evaluation on full dataset with specified provider.

Output Structure:
    Results are saved to: results/{provider}/{model}_{run_type}_{timestamp}/

Usage:
    # Single run using local test.csv (default)
    python run_eval.py --provider together
    python run_eval.py --provider anthropic --model claude-sonnet-4-20250514

    # Use a different CSV file
    python run_eval.py --provider together --csv evals/train.csv

    # Use HuggingFace dataset instead of local CSV
    python run_eval.py --provider together --use-hf

    # Chained pipeline (multi-stage reasoning)
    python run_eval.py --provider anthropic --use-chain

    # Compare baseline vs chain
    python run_eval.py --provider anthropic --compare --samples 50

    # Multi-run with statistical analysis
    python run_eval.py --provider together --n-runs 3
"""

import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env
for env_path in [".env", "evals/.env"]:
    if Path(env_path).exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

from evals import (
    load_samples_from_hf,
    load_samples_from_csv,
    TutorBenchJudge,
    evaluate_model,
    aggregate_scores,
    aggregate_by_dimension,
    save_results_to_json,
    get_provider,
    ConcurrencyConfig,
    create_tutoring_chain,
    save_chain_debug_log,
)
from evals.scoring import aggregate_by_skill
from evals.framework import (
    ExperimentConfig,
    generate_full_report,
    aggregate_multiple_runs,
    print_summary,
    save_leaderboard_entry,
    compute_comparison_stats,
    print_comparison_table,
    save_comparison_report,
)


def get_model_id(provider: str, model: str = None) -> str:
    """Get clean model identifier for directory naming."""
    if not model:
        return "default"
    return model.replace("/", "_").replace(":", "_")


def get_output_dir(base_dir: str, provider: str, model: str = None, suffix: str = None) -> Path:
    """Generate structured output directory path."""
    model_id = get_model_id(provider, model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{model_id}_{suffix}_{timestamp}" if suffix else f"{model_id}_{timestamp}"
    output_path = Path(base_dir) / provider / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def print_single_run_summary(results):
    """Print summary for single run."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    overall = aggregate_scores(results)
    print(f"\nOverall Score: {overall['overall_score']:.1%}")
    print(f"Mean Pass Rate: {overall['mean_pass_rate']:.1%}")
    print(f"Std Dev: {overall['std_score']:.1%}")
    print(f"Samples: {overall['n_samples']}, Rubrics: {overall['total_rubrics']}")

    print("\n" + "-" * 70)
    print("BY EVALUATION DIMENSION")
    print("-" * 70)
    by_dim = aggregate_by_dimension(results)
    for dim, m in sorted(by_dim.items(), key=lambda x: x[1]['pass_rate'], reverse=True):
        if m['n_rubrics'] > 0:
            print(f"{dim.value:30s}: {m['pass_rate']:5.1%} ({m['n_rubrics']} rubrics)")

    print("\n" + "-" * 70)
    print("LOWEST SCORING SAMPLES")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.weighted_score)[:5]:
        passed = sum(rr.passed for rr in r.rubric_ratings)
        print(f"{r.sample_id[:24]:24s} | Score: {r.weighted_score:5.1%} | Pass: {passed}/{len(r.rubric_ratings)}")

    print("=" * 70)


def run_single_evaluation(args, samples, model_fn, judge, config, output_dir=None):
    """Run a single evaluation."""
    print(f"\nRunning single evaluation on {len(samples)} samples...")

    results = evaluate_model(
        samples=samples,
        model_fn=model_fn,
        judge=judge,
        model_name=f"{args.provider}/{args.model or 'default'}",
        verbose=True,
        concurrency_config=config,
    )

    if output_dir is None:
        suffix = "chain" if args.use_chain else ("bon" + str(args.best_of_n_count) if args.use_best_of_n else "baseline")
        output_dir = get_output_dir(args.output_dir, args.provider, args.model, suffix)

    print(f"Output directory: {output_dir}")
    save_results_to_json(results, str(output_dir / "results.json"))

    if args.chain_debug and hasattr(model_fn, 'get_chain_outputs'):
        save_chain_debug_log(model_fn.get_chain_outputs(), samples, results, str(output_dir / "chain_debug.json"))

    if args.best_of_n_debug and hasattr(model_fn, 'get_bon_outputs'):
        from evals import save_bon_debug_log
        save_bon_debug_log(model_fn.get_bon_outputs(), samples, results, str(output_dir / "bon_debug.json"))

    print_single_run_summary(results)
    return results


def run_multi_evaluation(args, samples, model_fn, judge, config, output_dir=None):
    """Run multiple evaluations with statistical analysis."""
    import json

    print(f"\nRunning {args.n_runs} evaluations for statistical rigor...")
    print(f"Processing {len(samples)} samples x {args.n_runs} runs")

    suffix = "chain" if args.use_chain else ("bon" + str(args.best_of_n_count) if args.use_best_of_n else "baseline")
    model_id = get_model_id(args.provider, args.model)

    exp_config = ExperimentConfig(
        model_name=model_id,
        model_version=args.model or "default",
        temperature=0.0,
        use_async=True,
        max_samples=args.samples,
        notes=f"Run via run_eval.py with {args.n_runs} runs"
    )

    if output_dir is None:
        output_dir = get_output_dir(args.output_dir, args.provider, args.model, suffix)

    exp_config.save(output_dir / "config.json")
    print(f"Output directory: {output_dir}")

    all_reports = []
    for run_idx in range(args.n_runs):
        print(f"\n{'='*70}\nRUN {run_idx + 1}/{args.n_runs}\n{'='*70}")

        results = evaluate_model(
            samples=samples,
            model_fn=model_fn,
            judge=judge,
            model_name=f"{args.provider}/{args.model or 'default'}",
            verbose=True,
            concurrency_config=config,
        )

        report = generate_full_report(results, samples, exp_config)
        with open(output_dir / f"run_{run_idx + 1}.json", 'w') as f:
            json.dump(report, f, indent=2)

        all_reports.append(report)
        print(f"\nRun {run_idx + 1} Score: {report['overall']['overall_score']:.2%}")

    print(f"\n{'='*70}\nAGGREGATING RESULTS\n{'='*70}")
    final_stats = aggregate_multiple_runs(all_reports)

    with open(output_dir / "final_report.json", 'w') as f:
        json.dump(final_stats, f, indent=2)

    if args.save_leaderboard:
        save_leaderboard_entry(final_stats, Path(args.output_dir) / "leaderboard.json")

    print_summary(final_stats, verbose=True)
    return final_stats


def run_comparison_evaluation(args, samples, base_provider, judge, config):
    """Run baseline vs chain comparison."""
    is_multi_run = args.n_runs > 1
    model_id = get_model_id(args.provider, args.model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = Path(args.output_dir) / args.provider / f"{model_id}_comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nComparison output directory: {comparison_dir}")

    # Baseline
    print(f"\n{'='*70}\nSTEP 1/2: BASELINE EVALUATION\n{'='*70}")
    baseline_dir = comparison_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    if is_multi_run:
        baseline_stats = run_multi_evaluation(args, samples, base_provider, judge, config, baseline_dir)
    else:
        baseline_stats = run_single_evaluation(args, samples, base_provider, judge, config, baseline_dir)

    # Chain
    print(f"\n{'='*70}\nSTEP 2/2: CHAIN EVALUATION\n{'='*70}")
    chain_dir = comparison_dir / "chain"
    chain_dir.mkdir(parents=True, exist_ok=True)

    chain = create_tutoring_chain(base_provider, include_safety_check=args.chain_safety_check, verbose=args.chain_verbose)
    chain_model_fn = chain.create_model_fn(log_chain_outputs=args.chain_debug)

    if is_multi_run:
        chain_stats = run_multi_evaluation(args, samples, chain_model_fn, judge, config, chain_dir)
    else:
        chain_stats = run_single_evaluation(args, samples, chain_model_fn, judge, config, chain_dir)

    # Comparison
    print(f"\n{'='*70}\nCOMPUTING COMPARISON\n{'='*70}")
    comparison = compute_comparison_stats(baseline_stats, chain_stats, is_multi_run=is_multi_run)
    print_comparison_table(comparison, is_multi_run=is_multi_run)
    save_comparison_report(comparison, comparison_dir / "comparison_report.json")

    return baseline_stats, chain_stats, comparison


def main():
    parser = argparse.ArgumentParser(description="Run TutorBench evaluation")
    parser.add_argument("--provider", type=str, default="together", choices=["together", "anthropic", "openai"])
    parser.add_argument("--model", type=str, default=None, help="Model identifier (uses provider default if not specified)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--include-multimodal", action="store_true", help="Include multimodal samples")
    parser.add_argument("--csv", type=str, default="evals/test.csv", help="Path to CSV file (default: evals/test.csv)")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace dataset instead of local CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--save-leaderboard", action="store_true", help="Save to leaderboard.json")

    # Concurrency
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--sample-batch-size", type=int, default=10)
    parser.add_argument("--rubric-batch-size", type=int, default=5)
    parser.add_argument("--max-requests-per-second", type=int, default=50)

    # Pipeline options
    parser.add_argument("--use-chain", action="store_true", help="Use chained LLM pipeline")
    parser.add_argument("--chain-safety-check", action="store_true", help="Include safety check stage")
    parser.add_argument("--chain-verbose", action="store_true", help="Print intermediate chain outputs")
    parser.add_argument("--chain-debug", action="store_true", help="Save chain debug log")
    parser.add_argument("--compare", action="store_true", help="Run baseline vs chain comparison")

    # Best-of-N
    parser.add_argument("--use-best-of-n", action="store_true", help="Use best-of-n sampling")
    parser.add_argument("--best-of-n-count", type=int, default=5, help="Number of candidates")
    parser.add_argument("--best-of-n-selector", type=str, default=None, help="Selector provider (format: provider/model)")
    parser.add_argument("--best-of-n-verbose", action="store_true")
    parser.add_argument("--best-of-n-debug", action="store_true")

    args = parser.parse_args()

    # Validation
    if args.use_chain and args.compare:
        parser.error("--use-chain and --compare are mutually exclusive")
    if args.use_best_of_n and (args.use_chain or args.compare):
        parser.error("--use-best-of-n is mutually exclusive with --use-chain and --compare")

    # Print mode
    print("=" * 70)
    print(f"TutorBench Evaluation: {args.provider.upper()}")
    mode = "COMPARISON" if args.compare else (f"MULTI-RUN ({args.n_runs})" if args.n_runs > 1 else "SINGLE RUN")
    pipeline = "CHAIN" if args.use_chain else (f"BEST-OF-{args.best_of_n_count}" if args.use_best_of_n else "BASELINE")
    print(f"Mode: {mode}, Pipeline: {pipeline}")
    print("=" * 70)

    # Load samples
    if args.use_hf:
        print(f"\nLoading samples from HuggingFace...")
        samples = load_samples_from_hf(
            dataset_name="ScaleAI/TutorBench",
            split="train",
            max_samples=args.samples,
            text_only=not args.include_multimodal,
        )
    else:
        print(f"\nLoading samples from {args.csv}...")
        samples = load_samples_from_csv(
            csv_path=args.csv,
            max_samples=args.samples,
            text_only=not args.include_multimodal,
        )
    print(f"Loaded {len(samples)} samples")

    # Initialize provider
    print(f"\nInitializing {args.provider} provider...")
    base_provider = get_provider(args.provider, args.model)

    # Wrap in pipeline if needed
    if args.use_chain:
        chain = create_tutoring_chain(base_provider, args.chain_safety_check, args.chain_verbose)
        model_fn = chain.create_model_fn(log_chain_outputs=args.chain_debug)
    elif args.use_best_of_n:
        from evals import create_best_of_n_pipeline
        selector = get_provider(*args.best_of_n_selector.split('/')) if args.best_of_n_selector else None
        pipeline = create_best_of_n_pipeline(base_provider, args.best_of_n_count, selector, args.best_of_n_verbose)
        model_fn = pipeline.create_model_fn(log_bon_outputs=args.best_of_n_debug)
    else:
        model_fn = base_provider

    # Initialize judge
    print("Initializing judge (Claude Sonnet 4)...")
    judge = TutorBenchJudge(verbose=False)

    # Concurrency config
    config = ConcurrencyConfig(
        max_concurrent=args.max_concurrent,
        sample_batch_size=args.sample_batch_size,
        rubric_batch_size=args.rubric_batch_size,
        max_requests_per_second=args.max_requests_per_second,
    )

    # Run evaluation
    if args.compare:
        run_comparison_evaluation(args, samples, base_provider, judge, config)
    elif args.n_runs == 1:
        run_single_evaluation(args, samples, model_fn, judge, config)
    else:
        run_multi_evaluation(args, samples, model_fn, judge, config)

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
