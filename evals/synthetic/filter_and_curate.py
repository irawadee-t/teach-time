"""
Filter and curate graded synthetic responses.

Applies quality filtering:
1. >= 95% pass rate on critical criteria
2. >= 80% weighted overall score

Applies diversity selection:
- Persona-based: keeps max 1 response per persona per sample
- Takes best scoring response from each persona

Usage:
    # Filter graded responses
    python -m evals.synthetic.filter_and_curate --input graded.jsonl

    # See detailed stats
    python -m evals.synthetic.filter_and_curate --input graded.jsonl --verbose

    # Use custom thresholds
    python -m evals.synthetic.filter_and_curate --input graded.jsonl --critical-threshold 0.90
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

# Thresholds (can be overridden via CLI)
CRITICAL_THRESHOLD = 0.70  # 70% critical pass required
OVERALL_THRESHOLD = 0.70   # 70% weighted score required
MAX_PER_PERSONA = 1        # Max responses per persona per sample
MIN_PASS_RATE = 0.25       # Ensure at least 25% of responses pass per sample (via revision)


@dataclass
class FilterStats:
    """Statistics from filtering process."""
    total_input: int = 0
    failed_critical: int = 0
    failed_overall: int = 0
    passed_quality: int = 0
    after_persona_selection: int = 0

    # By subject
    by_subject: Dict[str, Dict] = None

    # By persona
    by_persona: Dict[str, Dict] = None

    def __post_init__(self):
        self.by_subject = defaultdict(lambda: {'input': 0, 'passed': 0})
        self.by_persona = defaultdict(lambda: {'input': 0, 'passed': 0})


def load_graded_responses(input_path: Path) -> List[Dict]:
    """Load graded responses from JSONL."""
    responses = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return responses


def filter_by_quality(
    responses: List[Dict],
    stats: FilterStats,
    critical_threshold: float = CRITICAL_THRESHOLD,
    overall_threshold: float = OVERALL_THRESHOLD,
) -> List[Dict]:
    """
    Filter responses by quality thresholds.

    Two-tier filtering:
    1. Critical: >= critical_threshold pass rate on critical criteria (default 95%)
    2. Overall: >= overall_threshold weighted score (default 80%)
    """
    passed = []

    for r in responses:
        stats.total_input += 1
        stats.by_subject[r['subject']]['input'] += 1
        stats.by_persona[r['persona_name']]['input'] += 1

        # Check critical threshold
        if r['critical_pass_rate'] < critical_threshold:
            stats.failed_critical += 1
            continue

        # Check overall threshold
        if r['weighted_score'] < overall_threshold:
            stats.failed_overall += 1
            continue

        # Passed both thresholds
        stats.passed_quality += 1
        stats.by_subject[r['subject']]['passed'] += 1
        stats.by_persona[r['persona_name']]['passed'] += 1
        passed.append(r)

    return passed


def select_diverse_responses(responses: List[Dict], stats: FilterStats) -> List[Dict]:
    """
    Select diverse responses using persona-based selection.

    For each sample, keeps at most 1 response per persona,
    selecting the highest scoring one.
    """
    # Group by sample_id
    by_sample = defaultdict(list)
    for r in responses:
        by_sample[r['sample_id']].append(r)

    selected = []

    for sample_id, sample_responses in by_sample.items():
        # Group by persona within sample
        by_persona = defaultdict(list)
        for r in sample_responses:
            by_persona[r['persona_name']].append(r)

        # Take best from each persona
        for persona_name, persona_responses in by_persona.items():
            # Sort by weighted score descending
            sorted_responses = sorted(
                persona_responses,
                key=lambda x: x['weighted_score'],
                reverse=True
            )
            # Take up to MAX_PER_PERSONA
            selected.extend(sorted_responses[:MAX_PER_PERSONA])

    stats.after_persona_selection = len(selected)
    return selected


def save_filtered_responses(responses: List[Dict], output_path: Path):
    """Save filtered responses to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def save_rejected_responses(
    all_responses: List[Dict],
    passed_ids: set,
    output_path: Path,
):
    """Save rejected responses with rejection reasons for analysis."""
    rejected = []

    for r in all_responses:
        task_id = f"{r['sample_id']}_{r['persona_name']}"
        if task_id not in passed_ids:
            # Determine rejection reason
            if r['critical_pass_rate'] < CRITICAL_THRESHOLD:
                reason = "failed_critical"
            elif r['weighted_score'] < OVERALL_THRESHOLD:
                reason = "failed_overall"
            else:
                reason = "diversity_filtered"

            r['rejection_reason'] = reason
            rejected.append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return len(rejected)


def print_stats(
    stats: FilterStats,
    verbose: bool = False,
    critical_threshold: float = CRITICAL_THRESHOLD,
    overall_threshold: float = OVERALL_THRESHOLD,
):
    """Print filtering statistics."""
    print(f"\n{'='*60}")
    print("FILTERING STATISTICS")
    print(f"{'='*60}")

    print(f"\n📊 Overall:")
    print(f"  Input responses:          {stats.total_input}")
    print(f"  Failed critical ({100*critical_threshold:.0f}%):   {stats.failed_critical} ({100*stats.failed_critical/stats.total_input:.1f}%)")
    print(f"  Failed overall ({100*overall_threshold:.0f}%):     {stats.failed_overall} ({100*stats.failed_overall/stats.total_input:.1f}%)")
    print(f"  Passed quality filters:   {stats.passed_quality} ({100*stats.passed_quality/stats.total_input:.1f}%)")
    print(f"  After persona selection:  {stats.after_persona_selection}")

    if verbose:
        print(f"\n📚 By Subject:")
        for subject in sorted(stats.by_subject.keys()):
            data = stats.by_subject[subject]
            rate = 100 * data['passed'] / data['input'] if data['input'] > 0 else 0
            print(f"  {subject:20s}: {data['passed']:4d}/{data['input']:4d} ({rate:.1f}%)")

        print(f"\n🎭 By Persona:")
        for persona in sorted(stats.by_persona.keys()):
            data = stats.by_persona[persona]
            rate = 100 * data['passed'] / data['input'] if data['input'] > 0 else 0
            print(f"  {persona:25s}: {data['passed']:4d}/{data['input']:4d} ({rate:.1f}%)")

    print(f"{'='*60}\n")


def run_filtering(
    input_path: Path,
    output_path: Path,
    rejected_path: Optional[Path] = None,
    verbose: bool = False,
    critical_threshold: float = CRITICAL_THRESHOLD,
    overall_threshold: float = OVERALL_THRESHOLD,
):
    """Run the full filtering pipeline."""

    print(f"\n{'='*60}")
    print("FILTERING AND CURATION")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Critical threshold: {100*critical_threshold:.0f}%")
    print(f"Overall threshold: {100*overall_threshold:.0f}%")
    print(f"{'='*60}\n")

    # Load responses
    print("Loading graded responses...")
    all_responses = load_graded_responses(input_path)
    print(f"Loaded {len(all_responses)} responses")

    # Initialize stats
    stats = FilterStats()

    # Step 1: Quality filtering
    print("\n1. Applying quality filters...")
    passed_quality = filter_by_quality(
        all_responses, stats,
        critical_threshold=critical_threshold,
        overall_threshold=overall_threshold,
    )
    print(f"   {len(passed_quality)} passed quality thresholds")

    # Keep all passing responses (no diversity selection - we want more training data)
    final_responses = passed_quality
    stats.after_persona_selection = len(final_responses)

    # Save filtered responses
    save_filtered_responses(final_responses, output_path)
    print(f"\n✅ Saved {len(final_responses)} high-quality responses to {output_path}")

    # Save rejected responses (for analysis)
    if rejected_path:
        passed_ids = {f"{r['sample_id']}_{r['persona_name']}" for r in final_responses}
        n_rejected = save_rejected_responses(all_responses, passed_ids, rejected_path)
        print(f"📝 Saved {n_rejected} rejected responses to {rejected_path}")

    # Print stats
    print_stats(stats, verbose, critical_threshold, overall_threshold)

    return final_responses


def main():
    parser = argparse.ArgumentParser(description="Filter and curate graded responses")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with graded responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: input path with _filtered suffix)")
    parser.add_argument("--save-rejected", action="store_true",
                        help="Also save rejected responses for analysis")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed stats by subject and persona")
    parser.add_argument("--critical-threshold", type=float, default=CRITICAL_THRESHOLD,
                        help=f"Critical criteria pass rate threshold (default: {CRITICAL_THRESHOLD})")
    parser.add_argument("--overall-threshold", type=float, default=OVERALL_THRESHOLD,
                        help=f"Overall weighted score threshold (default: {OVERALL_THRESHOLD})")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_filtered.jsonl"

    rejected_path = None
    if args.save_rejected:
        rejected_path = input_path.parent / f"{input_path.stem}_rejected.jsonl"

    run_filtering(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        verbose=args.verbose,
        critical_threshold=args.critical_threshold,
        overall_threshold=args.overall_threshold,
    )


if __name__ == "__main__":
    main()
