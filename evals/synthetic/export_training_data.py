"""
Export curated synthetic data as training JSONL.

Formats responses for instruction finetuning and generates full visibility
reports on good vs bad samples.

Output formats:
- training.jsonl: Ready for finetuning (messages format)
- quality_report.json: Full statistics and analysis
- good_samples.jsonl: High-quality samples with explanations
- bad_samples.jsonl: Rejected samples with failure reasons

Usage:
    # Export training data
    python -m evals.synthetic.export_training_data --input deduped.jsonl

    # With full visibility reports
    python -m evals.synthetic.export_training_data --input deduped.jsonl --full-report
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Lazy numpy import
np = None


def _ensure_numpy():
    """Lazily import numpy."""
    global np
    if np is None:
        try:
            import numpy as _np
            np = _np
        except ImportError as e:
            raise ImportError(
                "numpy is required for statistics. Install with: pip install numpy"
            ) from e


def load_responses(input_path: Path) -> List[Dict]:
    """Load responses from JSONL."""
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return responses


def format_for_training(response: Dict) -> Dict:
    """
    Format a response for instruction finetuning.

    Uses chat/messages format compatible with most finetuning frameworks.
    """
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
                "content": "You are an expert tutor helping a high school student."
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": response['response']
            }
        ],
        "metadata": {
            "source_sample_id": response['sample_id'],
            "subject": response['subject'],
            "generation_persona": response['persona_name'],
            "quality_score": response['weighted_score'],
            "critical_pass_rate": response['critical_pass_rate'],
            "generation_model": response['model'],
        }
    }


def generate_quality_report(
    good_responses: List[Dict],
    bad_responses: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive quality analysis report."""
    _ensure_numpy()

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {},
        "by_subject": {},
        "by_persona": {},
        "score_distribution": {},
        "rubric_analysis": {},
    }

    # Summary stats
    good_scores = [r['weighted_score'] for r in good_responses]
    report["summary"] = {
        "total_good_samples": len(good_responses),
        "total_bad_samples": len(bad_responses) if bad_responses else 0,
        "pass_rate": len(good_responses) / (len(good_responses) + len(bad_responses or [])) if good_responses or bad_responses else 0,
        "mean_score": float(np.mean(good_scores)) if good_scores else 0,
        "median_score": float(np.median(good_scores)) if good_scores else 0,
        "std_score": float(np.std(good_scores)) if good_scores else 0,
        "min_score": float(min(good_scores)) if good_scores else 0,
        "max_score": float(max(good_scores)) if good_scores else 0,
    }

    # By subject
    by_subject = defaultdict(lambda: {'count': 0, 'scores': []})
    for r in good_responses:
        by_subject[r['subject']]['count'] += 1
        by_subject[r['subject']]['scores'].append(r['weighted_score'])

    for subject, data in by_subject.items():
        report["by_subject"][subject] = {
            "count": data['count'],
            "mean_score": float(np.mean(data['scores'])),
            "std_score": float(np.std(data['scores'])),
        }

    # By persona
    by_persona = defaultdict(lambda: {'count': 0, 'scores': []})
    for r in good_responses:
        by_persona[r['persona_name']]['count'] += 1
        by_persona[r['persona_name']]['scores'].append(r['weighted_score'])

    for persona, data in by_persona.items():
        report["by_persona"][persona] = {
            "count": data['count'],
            "mean_score": float(np.mean(data['scores'])),
            "std_score": float(np.std(data['scores'])),
        }

    # Score distribution (histogram buckets)
    buckets = [0.80, 0.85, 0.90, 0.95, 1.0]
    score_dist = defaultdict(int)
    for score in good_scores:
        for bucket in buckets:
            if score <= bucket:
                score_dist[f"<={bucket:.2f}"] += 1
                break

    report["score_distribution"] = dict(score_dist)

    # Rubric analysis (aggregate pass rates by criterion type)
    if good_responses and 'rubric_results' in good_responses[0]:
        by_skill = defaultdict(lambda: {'total': 0, 'passed': 0})
        by_dimension = defaultdict(lambda: {'total': 0, 'passed': 0})
        by_severity = defaultdict(lambda: {'total': 0, 'passed': 0})

        for r in good_responses:
            for rubric in r.get('rubric_results', []):
                skill = rubric.get('tutoring_skill') or 'unknown'
                dim = rubric.get('eval_dimension', 'unknown')
                sev = rubric.get('severity', 'unknown')

                by_skill[skill]['total'] += 1
                by_dimension[dim]['total'] += 1
                by_severity[sev]['total'] += 1

                if rubric.get('passed'):
                    by_skill[skill]['passed'] += 1
                    by_dimension[dim]['passed'] += 1
                    by_severity[sev]['passed'] += 1

        report["rubric_analysis"]["by_tutoring_skill"] = {
            skill: {
                "total": data['total'],
                "passed": data['passed'],
                "pass_rate": data['passed'] / data['total'] if data['total'] > 0 else 0,
            }
            for skill, data in by_skill.items()
        }

        report["rubric_analysis"]["by_eval_dimension"] = {
            dim: {
                "total": data['total'],
                "passed": data['passed'],
                "pass_rate": data['passed'] / data['total'] if data['total'] > 0 else 0,
            }
            for dim, data in by_dimension.items()
        }

        report["rubric_analysis"]["by_severity"] = {
            sev: {
                "total": data['total'],
                "passed": data['passed'],
                "pass_rate": data['passed'] / data['total'] if data['total'] > 0 else 0,
            }
            for sev, data in by_severity.items()
        }

    # Bad sample analysis
    if bad_responses:
        rejection_reasons = defaultdict(int)
        for r in bad_responses:
            reason = r.get('rejection_reason', 'unknown')
            rejection_reasons[reason] += 1

        report["rejection_analysis"] = {
            "total_rejected": len(bad_responses),
            "by_reason": dict(rejection_reasons),
        }

        # Most common failed criteria
        failed_criteria = defaultdict(int)
        for r in bad_responses:
            for rubric in r.get('rubric_results', []):
                if not rubric.get('passed'):
                    criterion_preview = rubric.get('criterion', '')[:100]
                    failed_criteria[criterion_preview] += 1

        report["rejection_analysis"]["top_failed_criteria"] = dict(
            sorted(failed_criteria.items(), key=lambda x: -x[1])[:20]
        )

    return report


def export_good_samples(responses: List[Dict], output_path: Path):
    """Export good samples with full details for visibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in responses:
            # Include everything for full visibility
            sample = {
                "sample_id": r['sample_id'],
                "persona_name": r['persona_name'],
                "subject": r['subject'],
                "weighted_score": r['weighted_score'],
                "critical_pass_rate": r['critical_pass_rate'],
                "prompt": r['prompt'],
                "follow_up": r['follow_up'],
                "response": r['response'],
                "rubric_results": r.get('rubric_results', []),
                "why_good": _explain_quality(r),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def export_bad_samples(responses: List[Dict], output_path: Path):
    """Export bad samples with failure analysis for visibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in responses:
            # Find failed criteria
            failed_rubrics = [
                rubric for rubric in r.get('rubric_results', [])
                if not rubric.get('passed')
            ]

            sample = {
                "sample_id": r['sample_id'],
                "persona_name": r['persona_name'],
                "subject": r['subject'],
                "weighted_score": r['weighted_score'],
                "critical_pass_rate": r['critical_pass_rate'],
                "rejection_reason": r.get('rejection_reason', 'unknown'),
                "prompt": r['prompt'],
                "follow_up": r['follow_up'],
                "response": r['response'],
                "failed_rubrics": failed_rubrics,
                "why_bad": _explain_failure(r),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def _explain_quality(r: Dict) -> str:
    """Generate human-readable explanation of why sample is good."""
    explanations = []

    score = r['weighted_score']
    if score >= 0.95:
        explanations.append(f"Excellent weighted score ({score:.2%})")
    elif score >= 0.90:
        explanations.append(f"Very good weighted score ({score:.2%})")
    else:
        explanations.append(f"Good weighted score ({score:.2%})")

    if r['critical_pass_rate'] >= 1.0:
        explanations.append("Passed all critical criteria")

    # Count passed rubrics by type
    rubric_results = r.get('rubric_results', [])
    if rubric_results:
        passed_count = sum(1 for rr in rubric_results if rr.get('passed'))
        total_count = len(rubric_results)
        explanations.append(f"Passed {passed_count}/{total_count} rubric criteria")

    return "; ".join(explanations)


def _explain_failure(r: Dict) -> str:
    """Generate human-readable explanation of why sample failed."""
    explanations = []

    reason = r.get('rejection_reason', 'unknown')

    if reason == 'failed_critical':
        explanations.append(f"Failed critical criteria (pass rate: {r['critical_pass_rate']:.2%})")
        # List which critical criteria failed
        failed_critical = [
            rr for rr in r.get('rubric_results', [])
            if not rr.get('passed') and rr.get('severity') == 'critical'
        ]
        for fc in failed_critical[:3]:  # Top 3
            explanations.append(f"  - {fc.get('criterion', '')[:80]}...")
    elif reason == 'failed_overall':
        explanations.append(f"Below 80% threshold (score: {r['weighted_score']:.2%})")
    elif reason == 'diversity_filtered':
        explanations.append("Filtered for diversity (similar to higher-scoring response)")
    else:
        explanations.append(f"Unknown rejection reason: {reason}")

    return "\n".join(explanations)


def run_export(
    input_path: Path,
    output_dir: Path,
    bad_samples_path: Optional[Path] = None,
    full_report: bool = False,
):
    """Run the full export pipeline."""

    print(f"\n{'='*60}")
    print("EXPORT TRAINING DATA")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Load good responses
    print("Loading curated responses...")
    good_responses = load_responses(input_path)
    print(f"Loaded {len(good_responses)} high-quality responses")

    # Load bad responses if available
    bad_responses = None
    if bad_samples_path and bad_samples_path.exists():
        print(f"Loading rejected responses from {bad_samples_path}...")
        bad_responses = load_responses(bad_samples_path)
        print(f"Loaded {len(bad_responses)} rejected responses")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export training data
    training_path = output_dir / "training.jsonl"
    print(f"\n1. Exporting training data...")
    with open(training_path, 'w', encoding='utf-8') as f:
        for r in good_responses:
            training_example = format_for_training(r)
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    print(f"   ✅ Saved {len(good_responses)} training examples to {training_path}")

    if full_report:
        # Generate quality report
        print(f"\n2. Generating quality report...")
        report = generate_quality_report(good_responses, bad_responses)
        report_path = output_dir / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved quality report to {report_path}")

        # Export good samples with explanations
        print(f"\n3. Exporting good samples with visibility...")
        good_path = output_dir / "good_samples.jsonl"
        export_good_samples(good_responses, good_path)
        print(f"   ✅ Saved {len(good_responses)} good samples to {good_path}")

        # Export bad samples with failure analysis
        if bad_responses:
            print(f"\n4. Exporting bad samples with failure analysis...")
            bad_path = output_dir / "bad_samples.jsonl"
            export_bad_samples(bad_responses, bad_path)
            print(f"   ✅ Saved {len(bad_responses)} bad samples to {bad_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Training examples: {len(good_responses)}")
    if bad_responses:
        total = len(good_responses) + len(bad_responses)
        print(f"Quality pass rate: {100*len(good_responses)/total:.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Export training data")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with curated responses")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: input dir / training_data)")
    parser.add_argument("--rejected", type=str, default=None,
                        help="Path to rejected samples JSONL (for analysis)")
    parser.add_argument("--full-report", action="store_true",
                        help="Generate full visibility reports")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "training_data"

    bad_samples_path = Path(args.rejected) if args.rejected else None

    run_export(
        input_path=input_path,
        output_dir=output_dir,
        bad_samples_path=bad_samples_path,
        full_report=args.full_report,
    )


if __name__ == "__main__":
    main()
