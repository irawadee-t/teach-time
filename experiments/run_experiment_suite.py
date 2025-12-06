#!/usr/bin/env python3
"""
Run the full TeachTime experiment suite (Experiments 1-4).

This script runs all experiments needed for the paper in sequence.

Usage:
    python experiments/run_experiment_suite.py
    python experiments/run_experiment_suite.py --quick  # Run with fewer episodes for testing
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time

# Experiment configurations to run
EXPERIMENTS = [
    "exp1_metrics_match",
    "exp2_learning_gains",
    "exp3_persona_robustness",
    "exp4_react_ablation",
]


def run_experiment(exp_name: str, verbose: bool = False) -> bool:
    """
    Run a single experiment using run_experiment.py.

    Args:
        exp_name: Experiment configuration name
        verbose: Whether to show verbose output

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'#'*70}")
    print(f"# Running: {exp_name}")
    print(f"{'#'*70}\n")

    cmd = ["python", "experiments/run_experiment.py", "--config", exp_name]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {exp_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run full TeachTime experiment suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (fewer episodes)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)"
    )

    args = parser.parse_args()

    # Determine which experiments to run
    experiments_to_run = args.experiments if args.experiments else EXPERIMENTS

    print(f"\n{'='*70}")
    print(f"TeachTime Experiment Suite")
    print(f"{'='*70}\n")
    print(f"Experiments to run: {len(experiments_to_run)}")
    for exp in experiments_to_run:
        print(f"  - {exp}")
    print()

    if args.quick:
        print("⚠️  Running in QUICK mode (fewer episodes for testing)")
        print()

    # Confirm before starting
    response = input("Start experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Run each experiment
    start_time = time.time()
    results = {}

    for exp_name in experiments_to_run:
        exp_start = time.time()
        success = run_experiment(exp_name, verbose=args.verbose)
        exp_duration = time.time() - exp_start

        results[exp_name] = {
            "success": success,
            "duration": exp_duration,
        }

        print(f"\n{'='*70}")
        if success:
            print(f"✓ {exp_name} completed in {exp_duration/60:.1f} minutes")
        else:
            print(f"✗ {exp_name} failed")
        print(f"{'='*70}\n")

        # Brief pause between experiments
        if exp_name != experiments_to_run[-1]:
            print("Pausing 5 seconds before next experiment...\n")
            time.sleep(5)

    # Print summary
    total_duration = time.time() - start_time

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT SUITE SUMMARY")
    print(f"{'#'*70}\n")

    print(f"Total duration: {total_duration/60:.1f} minutes\n")

    successful = sum(1 for r in results.values() if r["success"])
    print(f"Completed: {successful}/{len(results)} experiments\n")

    for exp_name, result in results.items():
        status = "✓" if result["success"] else "✗"
        duration = result["duration"] / 60
        print(f"  {status} {exp_name:30s} ({duration:.1f} min)")

    print()

    if successful == len(results):
        print("✓ All experiments completed successfully!")
        print("\nNext steps:")
        print("  1. Run analysis notebooks in notebooks/")
        print("  2. Check results in results/processed/")
        print("  3. Generate paper figures in results/plots/")
    else:
        print("⚠️  Some experiments failed. Check logs above.")

    print()


if __name__ == "__main__":
    main()
