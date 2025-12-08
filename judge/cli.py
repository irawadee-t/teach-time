"""
Command-line interface for LLM-as-a-judge pedagogical evaluation.

Usage:
    python -m judge.cli evaluate <conversation.json>
    python -m judge.cli batch <directory>
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List

from .evaluator import PedagogicalEvaluator
from .metrics import calculate_pes, get_pes_category
from .report import generate_report, save_evaluation_results, generate_summary_report


def evaluate_single(
    filepath: Path,
    api_key: str,
    output_dir: Path,
    verbose: bool = False,
):
    """Evaluate a single conversation file."""
    print(f"\nEvaluating: {filepath.name}")
    print("-" * 60)

    # Load conversation
    with open(filepath, "r") as f:
        data = json.load(f)

    conversation = data.get("conversation", [])
    metadata = data.get("metadata", {})
    name = metadata.get("name", filepath.stem)

    # Initialize evaluator
    evaluator = PedagogicalEvaluator(
        api_key=api_key,
        judge_model="deepseek-ai/DeepSeek-V3",
        verbose=verbose,
    )

    # Evaluate
    components = evaluator.evaluate_conversation(conversation)
    pes = calculate_pes(components)
    category = get_pes_category(pes)

    # Print summary
    print(f"\nPES Score: {pes}/100")
    print(f"Category: {category}")
    print(f"\nSummary: {components.summary}")

    # Save results
    if output_dir:
        save_evaluation_results(components, conversation, output_dir, name)

    return {
        "name": name,
        "pes_score": pes,
        "pes_category": category,
        "summary": components.summary,
    }


def evaluate_batch(
    directory: Path,
    api_key: str,
    output_dir: Path,
    verbose: bool = False,
):
    """Evaluate all JSON files in a directory."""
    json_files = list(directory.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    print(f"\nFound {len(json_files)} conversation files")
    print("=" * 60)

    results = []
    for i, filepath in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}]")
        result = evaluate_single(filepath, api_key, output_dir, verbose)
        results.append(result)

    # Generate summary report
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)

    summary = generate_summary_report(results)
    print(summary)

    # Save summary
    if output_dir:
        summary_path = output_dir / "batch_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Pedagogical Evaluation System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Evaluate single file
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single conversation")
    eval_parser.add_argument("filepath", type=Path, help="Path to conversation JSON file")
    eval_parser.add_argument(
        "--api-key",
        type=str,
        help="Together AI API key (or set TOGETHER_API_KEY env var)",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("judge/results"),
        help="Output directory for results (default: judge/results)",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation progress",
    )

    # Batch evaluate
    batch_parser = subparsers.add_parser("batch", help="Evaluate multiple conversations")
    batch_parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing conversation JSON files",
    )
    batch_parser.add_argument(
        "--api-key",
        type=str,
        help="Together AI API key (or set TOGETHER_API_KEY env var)",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("judge/results"),
        help="Output directory for results (default: judge/results)",
    )
    batch_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation progress",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Get API key
    import os
    api_key = args.api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: Together AI API key required")
        print("Set TOGETHER_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run command
    if args.command == "evaluate":
        if not args.filepath.exists():
            print(f"Error: File not found: {args.filepath}")
            sys.exit(1)

        evaluate_single(args.filepath, api_key, args.output_dir, args.verbose)

    elif args.command == "batch":
        if not args.directory.exists():
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)

        evaluate_batch(args.directory, api_key, args.output_dir, args.verbose)


if __name__ == "__main__":
    main()
