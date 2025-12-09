"""
Watch grading progress and periodically export passing samples to OpenAI format.

Monitors graded responses, combines:
1. Original samples that passed grading (score >= threshold)
2. Revised samples that passed grading

Exports to OpenAI fine-tuning JSONL format periodically.

Usage:
    # Watch and export every 5 minutes
    python -m evals.synthetic.watch_and_export \
        --graded evals/synthetic/pipeline_outputs/20251208_182025/02_graded.jsonl \
        --revised evals/synthetic/pipeline_outputs/20251208_182025/04_revised_graded.jsonl \
        --output evals/synthetic/pipeline_outputs/20251208_182025/training.jsonl \
        --interval 300

    # One-shot export (no watching)
    python -m evals.synthetic.watch_and_export \
        --graded 02_graded.jsonl \
        --output training.jsonl \
        --once
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set

# Quality threshold
DEFAULT_SCORE_THRESHOLD = 0.70


def load_passing_samples(
    graded_path: Path,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    seen_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """Load samples that pass the quality threshold."""
    if not graded_path.exists():
        return []

    passing = []
    with open(graded_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                sample_id = f"{r['sample_id']}_{r['persona_name']}"

                # Skip if already seen
                if seen_ids and sample_id in seen_ids:
                    continue

                # Check quality
                score = r.get('weighted_score', 0)
                if score >= score_threshold:
                    passing.append(r)

            except (json.JSONDecodeError, KeyError):
                continue

    return passing


def format_for_openai(response: Dict) -> Dict:
    """
    Format a response for OpenAI fine-tuning.

    OpenAI format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    # Build user content from the tutoring context
    user_content = f"""## Question
{response['prompt']}

## Previous Explanation
{response['initial_explanation']}

## Student's Follow-up
{response['follow_up']}"""

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert tutor helping a high school student. Provide clear, adaptive explanations tailored to the student's level of understanding."
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": response['response']
            }
        ]
    }


def export_to_openai_format(
    samples: List[Dict],
    output_path: Path,
    append: bool = False,
):
    """Export samples to OpenAI fine-tuning JSONL format."""
    mode = 'a' if append else 'w'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding='utf-8') as f:
        for sample in samples:
            formatted = format_for_openai(sample)
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')


def get_unique_id(r: Dict) -> str:
    """Get unique identifier for a sample."""
    return f"{r['sample_id']}_{r['persona_name']}"


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    if not path.exists():
        return 0
    with open(path, 'r') as f:
        return sum(1 for _ in f)


def watch_and_export(
    graded_path: Path,
    output_path: Path,
    revised_path: Optional[Path] = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    interval: int = 300,
    once: bool = False,
):
    """Watch grading files and periodically export passing samples."""

    print(f"\n{'='*60}")
    print("WATCH AND EXPORT TO OPENAI FORMAT")
    print(f"{'='*60}")
    print(f"Graded file: {graded_path}")
    if revised_path:
        print(f"Revised file: {revised_path}")
    print(f"Output: {output_path}")
    print(f"Score threshold: {score_threshold:.0%}")
    print(f"Interval: {interval}s" if not once else "Mode: one-shot")
    print(f"{'='*60}\n")

    exported_ids: Set[str] = set()
    total_exported = 0

    # Load any existing exports
    if output_path.exists() and not once:
        # In watch mode, we track what's been exported to avoid duplicates
        total_exported = count_lines(output_path)
        print(f"Found {total_exported} existing exports in {output_path}")

    while True:
        try:
            new_samples = []

            # Load passing samples from graded file
            graded_passing = load_passing_samples(
                graded_path,
                score_threshold,
                seen_ids=exported_ids if not once else None
            )
            new_samples.extend(graded_passing)

            # Load passing samples from revised file
            if revised_path and revised_path.exists():
                revised_passing = load_passing_samples(
                    revised_path,
                    score_threshold,
                    seen_ids=exported_ids if not once else None
                )
                new_samples.extend(revised_passing)

            if new_samples:
                # Export new samples
                if once:
                    # One-shot: overwrite
                    export_to_openai_format(new_samples, output_path, append=False)
                    total_exported = len(new_samples)
                else:
                    # Watch mode: append
                    export_to_openai_format(new_samples, output_path, append=True)
                    total_exported += len(new_samples)

                    # Track exported IDs
                    for sample in new_samples:
                        exported_ids.add(get_unique_id(sample))

                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Exported {len(new_samples)} new samples (total: {total_exported})")

                # Show breakdown
                graded_count = len(graded_passing)
                revised_count = len(new_samples) - graded_count
                if revised_count > 0:
                    print(f"           - Original passing: {graded_count}")
                    print(f"           - Revised passing: {revised_count}")
            else:
                timestamp = datetime.now().strftime('%H:%M:%S')
                graded_total = count_lines(graded_path) if graded_path.exists() else 0
                revised_total = count_lines(revised_path) if revised_path and revised_path.exists() else 0
                print(f"[{timestamp}] No new samples (graded: {graded_total}, revised: {revised_total}, exported: {total_exported})")

            if once:
                break

            # Wait for next check
            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\nStopped. Total exported: {total_exported}")
            break
        except Exception as e:
            print(f"Error: {e}")
            if once:
                raise
            time.sleep(interval)

    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples exported: {total_exported}")
    print(f"Output file: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Watch and export passing samples to OpenAI format")
    parser.add_argument("--graded", type=str, required=True,
                        help="Path to graded responses JSONL")
    parser.add_argument("--revised", type=str, default=None,
                        help="Path to revised+graded responses JSONL (optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for OpenAI format JSONL")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"Quality score threshold (default: {DEFAULT_SCORE_THRESHOLD})")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 300)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no watching)")

    args = parser.parse_args()

    watch_and_export(
        graded_path=Path(args.graded),
        output_path=Path(args.output),
        revised_path=Path(args.revised) if args.revised else None,
        score_threshold=args.threshold,
        interval=args.interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()
