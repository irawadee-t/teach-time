"""
Main evaluation orchestration script.

This ties everything together: loads samples, runs multiple evaluations,
computes statistics, and generates comprehensive reports.
"""

import json
from pathlib import Path
from typing import Callable, Optional, List

# Import from parent evals package
from ..models import Sample
from ..runner import evaluate_model
from ..judge import TutorBenchJudge
from ..data_loader import load_samples_from_json, load_samples_from_hf

# Import from framework
from .experiment import ExperimentConfig
from .statistics import aggregate_multiple_runs
from .reporting import generate_full_report, print_summary, save_leaderboard_entry


def run_evaluation(
    model_fn: Callable[[str, List[dict]], str],
    config: ExperimentConfig,
    samples_path: Optional[str] = None,
    use_hf: bool = False,
    hf_dataset: str = "ScaleAI/TutorBench",
    hf_split: str = "train",
    n_runs: int = 3,
    output_dir: str = "results",
    save_to_leaderboard: bool = True,
    verbose: bool = True
) -> dict:
    """
    Run complete TutorBench evaluation with statistical rigor.

    This is the main entry point for running evaluations. It:
    1. Loads samples from JSON or HuggingFace
    2. Runs evaluation n_runs times
    3. Computes confidence intervals and aggregates statistics
    4. Saves detailed reports and updates leaderboard
    5. Prints human-readable summary

    Args:
        model_fn: Your model function that takes (system_prompt, messages) -> response
        config: Experiment configuration
        samples_path: Path to samples JSON file (if not using HF)
        use_hf: Whether to load from HuggingFace dataset
        hf_dataset: HuggingFace dataset name (default: "ScaleAI/TutorBench")
        hf_split: Dataset split to use (default: "train")
        n_runs: Number of evaluation runs for computing confidence intervals
        output_dir: Directory to save results
        save_to_leaderboard: Whether to update leaderboard.json
        verbose: Whether to print progress and detailed output

    Returns:
        Aggregated statistics dictionary from all runs

    Example:
        >>> def my_model(system_prompt: str, messages: list) -> str:
        ...     # Your model API call here
        ...     return response
        >>>
        >>> config = ExperimentConfig(
        ...     model_name="gpt-4",
        ...     model_version="gpt-4-2024-01-15",
        ...     temperature=0.0
        ... )
        >>>
        >>> results = run_evaluation(
        ...     model_fn=my_model,
        ...     config=config,
        ...     use_hf=True,
        ...     n_runs=3
        ... )
    """

    # Create output directory structure
    output_path = Path(output_dir) / config.model_name / config.run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(output_path / "config.json")

    # Load samples
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING SAMPLES")
        print(f"{'='*70}\n")

    if use_hf:
        if verbose:
            print(f"Loading from HuggingFace: {hf_dataset} (split: {hf_split})")
        samples = load_samples_from_hf(
            dataset_name=hf_dataset,
            split=hf_split,
            max_samples=config.max_samples
        )
    else:
        if not samples_path:
            raise ValueError("Must provide samples_path if not using HuggingFace")
        if verbose:
            print(f"Loading from file: {samples_path}")
        samples = load_samples_from_json(samples_path)
        if config.max_samples:
            samples = samples[:config.max_samples]

    if verbose:
        print(f"Loaded {len(samples)} samples")
        multimodal_count = sum(1 for s in samples if s.is_multimodal)
        print(f"  Text-only: {len(samples) - multimodal_count}")
        print(f"  Multimodal: {multimodal_count}")

    # Initialize judge once for all runs
    if verbose:
        print(f"\nInitializing TutorBench judge...")
    judge = TutorBenchJudge(verbose=False)

    # Run multiple evaluations
    if verbose:
        print(f"\n{'='*70}")
        print(f"RUNNING {n_runs} EVALUATION RUNS")
        print(f"{'='*70}\n")

    all_reports = []
    for run_idx in range(n_runs):
        if verbose:
            print(f"\n{'-'*70}")
            print(f"RUN {run_idx + 1}/{n_runs}")
            print(f"{'-'*70}\n")

        # Run evaluation using existing runner
        results = evaluate_model(
            samples=samples,
            model_fn=model_fn,
            judge=judge,
            model_name=config.model_name,
            verbose=verbose,
            use_async=config.use_async
        )

        # Generate comprehensive report
        report = generate_full_report(results, samples, config)

        # Save individual run
        run_file = output_path / f"run_{run_idx + 1}.json"
        with open(run_file, 'w') as f:
            json.dump(report, f, indent=2)

        all_reports.append(report)

        if verbose:
            print(f"\nRun {run_idx + 1} Overall Score: {report['overall']['overall_score']:.2%}")
            print(f"Saved to: {run_file}")

    # Aggregate across all runs
    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATING RESULTS")
        print(f"{'='*70}\n")

    final_stats = aggregate_multiple_runs(all_reports)

    # Save final aggregated results
    final_file = output_path / "final_report.json"
    with open(final_file, 'w') as f:
        json.dump(final_stats, f, indent=2)

    if verbose:
        print(f"Final report saved to: {final_file}")

    # Update leaderboard
    if save_to_leaderboard:
        leaderboard_path = Path(output_dir) / "leaderboard.json"
        save_leaderboard_entry(final_stats, leaderboard_path)

    # Print summary
    if verbose:
        print_summary(final_stats, verbose=True)

    return final_stats
