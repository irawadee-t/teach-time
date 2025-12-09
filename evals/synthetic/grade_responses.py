"""
Grade synthetic tutoring responses using Claude Sonnet judge.

Grades generated responses from generate_samples_v2.py against sample rubrics.
Uses the existing TutorBenchJudge for evaluation.

Usage:
    # Dry run to see what would be graded
    python -m evals.synthetic.grade_responses --input responses.jsonl --dry-run

    # Grade all responses
    python -m evals.synthetic.grade_responses --input responses.jsonl

    # Resume from checkpoint
    python -m evals.synthetic.grade_responses --input responses.jsonl --resume
"""

import asyncio
import csv
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# Load environment variables from .env.local (preferred) or .env
_project_root = Path(__file__).parent.parent.parent
for env_file in [".env.local", ".env", "evals/.env.local", "evals/.env"]:
    env_path = _project_root / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break

from tqdm.asyncio import tqdm_asyncio, tqdm

from evals.judge import TutorBenchJudge
from evals.models import Rubric, RubricRating, EvaluationDimension, TutoringSkill
from evals.scoring import compute_weighted_score
from evals.config import JUDGE_MODEL, MAX_CONCURRENT, RUBRIC_BATCH_SIZE

# Configuration
CHECKPOINT_INTERVAL = 500  # Save checkpoint every N completions (streaming)

# Paths
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR.parent
TRAIN_CSV = EVALS_DIR / "train.csv"

# Negative criterion patterns (weight -5 when violated)
NEGATIVE_PATTERNS = [
    r"must\s+not",
    r"should\s+not",
    r"must\s+avoid",
    r"should\s+avoid",
    r"must\s+not\s+reveal",
    r"must\s+not\s+give\s+away",
    r"must\s+not\s+state\s+.*\s+answer",
]


@dataclass
class GradedResponse:
    """A graded response with full visibility into judgments."""
    sample_id: str
    persona_name: str
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str
    response: str
    model: str
    generation_timestamp: str

    # Grading results
    weighted_score: float
    pass_rate: float
    critical_pass_rate: float
    rubric_results: List[Dict[str, Any]]  # Full details per criterion

    # Quality flags
    passes_critical_threshold: bool  # 80% critical pass
    passes_overall_threshold: bool   # 80% overall
    is_high_quality: bool            # Both thresholds

    grading_timestamp: str
    judge_model: str

    # Revision tracking
    is_revision: bool = False  # True if this was a revised response


def is_negative_criterion(criterion_text: str) -> bool:
    """Check if criterion is negative (penalizes bad behavior)."""
    return any(re.search(p, criterion_text, re.IGNORECASE) for p in NEGATIVE_PATTERNS)


def parse_rubric_from_csv(rubric_json: str) -> List[Rubric]:
    """Parse rubrics from CSV JSON column."""
    try:
        rubrics_raw = json.loads(rubric_json)
    except json.JSONDecodeError:
        return []

    rubrics = []
    for r in rubrics_raw:
        criterion = r.get('criteria', '')
        severity = r.get('attributes', {}).get('severity', 'not_critical')

        # Determine weight
        if severity == 'critical':
            if is_negative_criterion(criterion):
                weight = -5
            else:
                weight = 5
        else:
            weight = 1

        # Map evaluation dimension
        eval_dim_raw = r.get('attributes', {}).get('eval_dimension', 'instruction_following')
        eval_dim_map = {
            'instruction_following': EvaluationDimension.INSTRUCTION_FOLLOWING,
            'truthfulness': EvaluationDimension.TRUTHFULNESS,
            'conciseness_relevance': EvaluationDimension.CONCISENESS_RELEVANCE,
            'style_tone': EvaluationDimension.STYLE_TONE,
            'visual_perception': EvaluationDimension.VISUAL_PERCEPTION,
            'visual_reasoning': EvaluationDimension.VISUAL_REASONING,
            'student_level_calibration': EvaluationDimension.STUDENT_LEVEL_CALIBRATION,
            'emotional_component': EvaluationDimension.EMOTIONAL_COMPONENT,
        }
        # Handle comma-separated dimensions
        first_dim = eval_dim_raw.split(',')[0].strip().lower().replace(' ', '_')
        eval_dim = eval_dim_map.get(first_dim, EvaluationDimension.INSTRUCTION_FOLLOWING)

        # Map tutoring skill
        skill_raw = r.get('attributes', {}).get('tutoring_skill', '')
        skill_map = {
            'asking guiding questions': TutoringSkill.ASKING_GUIDING_QUESTIONS,
            'asks questions to guide': TutoringSkill.ASKING_GUIDING_QUESTIONS,
            'identifying core difficulty': TutoringSkill.IDENTIFYING_CORE_DIFFICULTY,
            'identifying core difficulty/ misconception attribution': TutoringSkill.IDENTIFYING_CORE_DIFFICULTY,
            'identifying correct steps': TutoringSkill.IDENTIFYING_CORRECT_STEPS,
            'identifying correct steps by student': TutoringSkill.IDENTIFYING_CORRECT_STEPS,
            'identifying incorrect steps': TutoringSkill.IDENTIFYING_INCORRECT_STEPS,
            'identifying incorrect steps by student': TutoringSkill.IDENTIFYING_INCORRECT_STEPS,
            'includes examples/analogy': TutoringSkill.INCLUDING_EXAMPLES,
            'including examples': TutoringSkill.INCLUDING_EXAMPLES,
            'providing alternative solutions': TutoringSkill.PROVIDING_ALTERNATIVE_SOLUTIONS,
            'stating knowledge': TutoringSkill.STATING_KNOWLEDGE,
            'step by step help': TutoringSkill.STEP_BY_STEP_HELP,
            'step by step help/ analysis': TutoringSkill.STEP_BY_STEP_HELP,
            'step by step help/ analysis ': TutoringSkill.STEP_BY_STEP_HELP,
        }
        skill_key = skill_raw.lower().strip()
        skill = skill_map.get(skill_key, None)

        # Objectivity/explicitness
        is_objective = r.get('attributes', {}).get('objectivity', 'objective') == 'objective'
        is_explicit = r.get('attributes', {}).get('explicitness', 'explicit') == 'explicit'

        rubrics.append(Rubric(
            criterion=criterion,
            weight=weight,
            evaluation_dimension=eval_dim,
            tutoring_skill=skill,
            is_objective=is_objective,
            is_explicit=is_explicit,
        ))

    return rubrics


def load_train_csv_rubrics(csv_path: Path) -> Dict[str, Dict]:
    """Load rubrics and context for each sample from train.csv."""
    samples = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sample_id = f"sample_{idx:04d}"
            samples[sample_id] = {
                'rubrics': parse_rubric_from_csv(row.get('RUBRICS', '[]')),
                'subject': row.get('SUBJECT', 'unknown'),
                'prompt': row.get('PROMPT', ''),
                'initial_explanation': row.get('UC1_INITIAL_EXPLANATION', ''),
                'follow_up': row.get('FOLLOW_UP_PROMPT', ''),
            }

    return samples


def load_generated_responses(input_path: Path) -> List[Dict]:
    """Load generated responses from JSONL."""
    responses = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return responses


def load_graded_checkpoint(output_path: Path) -> set:
    """Load already graded task IDs."""
    completed = set()

    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    task_id = f"{record['sample_id']}_{record['persona_name']}"
                    completed.add(task_id)
                except:
                    continue

    return completed


def build_context(sample: Dict, response: Dict) -> str:
    """Build context string for judge evaluation."""
    context = f"""Subject: {sample['subject']}

## Original Question
{sample['prompt']}

## Initial Explanation Provided
{sample['initial_explanation']}

## Student's Follow-up Question
{sample['follow_up']}"""

    return context


async def grade_single_response(
    judge: TutorBenchJudge,
    response: Dict,
    sample: Dict,
    semaphore: asyncio.Semaphore,
) -> Optional[GradedResponse]:
    """Grade a single response against its rubrics."""

    async with semaphore:
        try:
            rubrics = sample['rubrics']
            if not rubrics:
                return None

            context = build_context(sample, response)

            # Evaluate all rubrics for this response
            ratings = await judge.evaluate_rubrics(
                model_response=response['response'],
                rubrics=rubrics,
                context=context,
                batch_size=RUBRIC_BATCH_SIZE,
            )

            # Compute scores
            weighted_score = compute_weighted_score(ratings)
            pass_rate = sum(r.passed for r in ratings) / len(ratings) if ratings else 0.0

            # Critical rubric pass rate
            critical_ratings = [r for r in ratings if abs(r.rubric.weight) == 5]
            if critical_ratings:
                critical_pass_rate = sum(r.passed for r in critical_ratings) / len(critical_ratings)
            else:
                critical_pass_rate = 1.0

            # Build rubric results with full visibility
            rubric_results = []
            for rating in ratings:
                rubric_results.append({
                    'criterion': rating.rubric.criterion,
                    'weight': rating.rubric.weight,
                    'severity': 'critical' if abs(rating.rubric.weight) == 5 else 'not_critical',
                    'is_negative': rating.rubric.weight < 0,
                    'passed': rating.passed,
                    'explanation': rating.explanation,
                    'eval_dimension': rating.rubric.evaluation_dimension.value,
                    'tutoring_skill': rating.rubric.tutoring_skill.value if rating.rubric.tutoring_skill else None,
                })

            # Quality flags (using 70% thresholds for consistency with filtering)
            passes_critical = critical_pass_rate >= 0.70  # 70% critical pass
            passes_overall = weighted_score >= 0.70       # 70% overall

            return GradedResponse(
                sample_id=response['sample_id'],
                persona_name=response['persona_name'],
                subject=response['subject'],
                prompt=response['prompt'],
                initial_explanation=response['initial_explanation'],
                follow_up=response['follow_up'],
                response=response['response'],
                model=response['model'],
                generation_timestamp=response['timestamp'],
                weighted_score=weighted_score,
                pass_rate=pass_rate,
                critical_pass_rate=critical_pass_rate,
                rubric_results=rubric_results,
                passes_critical_threshold=passes_critical,
                passes_overall_threshold=passes_overall,
                is_high_quality=passes_critical and passes_overall,
                grading_timestamp=datetime.now().isoformat(),
                judge_model=judge.model,
                is_revision=response.get('is_revision', False),
            )

        except Exception as e:
            print(f"\nError grading {response['sample_id']}/{response['persona_name']}: {e}")
            return None


def save_graded_responses(responses: List[GradedResponse], output_path: Path, mode: str = 'a'):
    """Save graded responses to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding='utf-8') as f:
        for resp in responses:
            f.write(json.dumps(asdict(resp), ensure_ascii=False) + '\n')


async def run_grading(
    input_path: Path,
    output_path: Path,
    train_csv_path: Path,
    max_samples: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
    concurrency: Optional[int] = None,
):
    """Run the full grading pipeline."""

    # Use provided concurrency or default from config
    max_concurrent = concurrency if concurrency is not None else MAX_CONCURRENT

    print(f"\n{'='*60}")
    print("SYNTHETIC RESPONSE GRADING")
    print(f"{'='*60}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*60}\n")

    # Load rubrics from train.csv
    print("Loading rubrics from train.csv...")
    samples = load_train_csv_rubrics(train_csv_path)
    print(f"Loaded rubrics for {len(samples)} samples")

    # Load generated responses
    print("Loading generated responses...")
    responses = load_generated_responses(input_path)
    print(f"Loaded {len(responses)} generated responses")

    if max_samples:
        responses = responses[:max_samples]
        print(f"Limited to {len(responses)} responses")

    # Filter to responses that have matching rubrics
    valid_responses = []
    for r in responses:
        if r['sample_id'] in samples and samples[r['sample_id']]['rubrics']:
            valid_responses.append(r)

    print(f"{len(valid_responses)} responses have valid rubrics")

    # Handle resume
    if resume:
        completed = load_graded_checkpoint(output_path)
        valid_responses = [r for r in valid_responses
                         if f"{r['sample_id']}_{r['persona_name']}" not in completed]
        print(f"Resuming: {len(completed)} already graded, {len(valid_responses)} remaining")

    if not valid_responses:
        print("No responses to grade!")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would grade {len(valid_responses)} responses")
        print("\nSample responses:")
        for r in valid_responses[:3]:
            sample = samples[r['sample_id']]
            print(f"  - {r['sample_id']} / {r['persona_name']} ({len(sample['rubrics'])} rubrics)")
        return

    # Initialize judge
    judge = TutorBenchJudge(model=JUDGE_MODEL, verbose=True)
    if not judge.client:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    # Grade ALL responses concurrently with immediate streaming to file
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"\nStarting grading of {len(valid_responses)} responses...")
    print(f"Concurrency: {max_concurrent}, Streaming: each result saved immediately")

    # Create all tasks
    all_tasks = []
    for r in valid_responses:
        sample = samples[r['sample_id']]
        all_tasks.append(grade_single_response(judge, r, sample, semaphore))

    # Process with streaming - write each result immediately
    total_graded = 0
    total_high_quality = 0
    mode = 'a' if resume else 'w'

    # Open file for streaming writes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode, encoding='utf-8') as f:
        for coro in tqdm.as_completed(all_tasks, total=len(all_tasks), desc="Grading"):
            result = await coro
            if result is not None:
                # Write immediately to file
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
                f.flush()  # Ensure it's written to disk
                total_graded += 1
                if result.is_high_quality:
                    total_high_quality += 1

    print(f"\n{'='*60}")
    print("GRADING COMPLETE")
    print(f"{'='*60}")
    print(f"Total graded: {total_graded}")
    print(f"High quality (80% critical + 80% overall): {total_high_quality}")
    print(f"High quality rate: {100 * total_high_quality / total_graded:.1f}%")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Grade synthetic tutoring responses")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with generated responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: input path with _graded suffix)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max responses to grade (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be graded without running")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent API calls (default: from config.py)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_graded.jsonl"

    asyncio.run(run_grading(
        input_path=input_path,
        output_path=output_path,
        train_csv_path=TRAIN_CSV,
        max_samples=args.max_samples,
        resume=args.resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
