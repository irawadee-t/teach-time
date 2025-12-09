"""
Revise failed responses using rubric feedback.

When no responses pass quality thresholds for a sample, this step:
1. Selects the top K highest-scoring responses (even though they failed)
2. Shows the model the specific rubric criteria that failed
3. Asks it to generate a revised response addressing those failures
4. Re-grades the revised response

This is essentially a "self-correction" step that uses grading feedback for improvement.

Usage:
    # Revise rejected responses
    python -m evals.synthetic.revise_responses --input rejected.jsonl --output revised.jsonl

    # Dry run
    python -m evals.synthetic.revise_responses --input rejected.jsonl --dry-run
"""

import asyncio
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables
_project_root = Path(__file__).parent.parent.parent
for env_file in [".env.local", ".env"]:
    env_path = _project_root / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break

from tqdm.asyncio import tqdm_asyncio
from together import AsyncTogether

# Configuration
REVISION_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAX_CONCURRENT = 500  # Match generation concurrency
MAX_TOKENS_REVISION = 2500  # Match generation token limit for complex math responses
REVISION_TEMPERATURE = 0.7  # Lower temp for more focused corrections
PER_CALL_TIMEOUT = 60.0  # seconds - cancel any API call taking longer than this

# Rate limiting (Together AI has 1800 RPM = 30 RPS limit on Qwen)
RATE_LIMIT_RPS = 30  # requests per second
_revision_rate_limiter: Optional[asyncio.Semaphore] = None
_revision_rate_limiter_task: Optional[asyncio.Task] = None

# Fallback API key if primary has insufficient balance
FALLBACK_API_KEY = "tgp_v1_OQB_4Mxxn6WflRB18-nLeBKPYj4fsVk_DJXaYS8dmY0"

# How many top responses to try revising per sample (when none pass)
TOP_K_TO_REVISE = 3

# Paths
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR.parent
TRAIN_CSV = EVALS_DIR / "train.csv"


@dataclass
class RevisionTask:
    """A response to revise with its failure feedback."""
    sample_id: str
    persona_name: str
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str
    original_response: str
    weighted_score: float
    critical_pass_rate: float
    failed_criteria: List[Dict[str, Any]]
    # Optional: CoT phases from original generation
    phase0_strategy: Optional[str] = None
    phase1_knowledge: Optional[str] = None
    phase1_5_errors: Optional[str] = None


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


def identify_samples_needing_revision(
    responses: List[Dict],
    critical_threshold: float = 0.70,
    overall_threshold: float = 0.70,
    min_pass_rate: float = 0.25,
) -> Dict[str, List[Dict]]:
    """
    Identify samples that need revision to meet minimum pass rate.

    For each sample:
    - Count how many responses already pass
    - If less than min_pass_rate (25%) pass, select top non-passing responses to revise
    - Returns dict mapping sample_id -> list of non-passing responses to revise (sorted by score desc)
    """
    # Group by sample_id
    by_sample = defaultdict(list)
    for r in responses:
        by_sample[r['sample_id']].append(r)

    needs_revision = {}
    for sample_id, sample_responses in by_sample.items():
        total = len(sample_responses)
        min_needed = max(1, int(total * min_pass_rate))  # At least 1, or 25% of total

        # Split into passing and non-passing
        passing = []
        non_passing = []
        for r in sample_responses:
            if (r.get('critical_pass_rate', 0) >= critical_threshold and
                r.get('weighted_score', 0) >= overall_threshold):
                passing.append(r)
            else:
                non_passing.append(r)

        # How many more do we need to reach min_pass_rate?
        shortfall = min_needed - len(passing)

        if shortfall > 0 and non_passing:
            # Sort non-passing by score (descending) - revise the best ones first
            sorted_non_passing = sorted(
                non_passing,
                key=lambda x: (x.get('weighted_score', 0), x.get('critical_pass_rate', 0)),
                reverse=True
            )
            # Take top K to revise, where K = shortfall
            needs_revision[sample_id] = sorted_non_passing[:shortfall]

    return needs_revision


def extract_failed_criteria(response: Dict) -> List[Dict[str, Any]]:
    """Extract the criteria that failed from rubric_results."""
    failed = []
    for r in response.get('rubric_results', []):
        if not r.get('passed', True):
            failed.append({
                'criterion': r['criterion'],
                'weight': r.get('weight', 1),
                'severity': r.get('severity', 'not_critical'),
                'explanation': r.get('explanation', ''),
            })
    return failed


def create_revision_tasks(
    needs_revision: Dict[str, List[Dict]],
) -> List[RevisionTask]:
    """Create revision tasks from samples needing revision.

    Note: needs_revision already contains exactly the responses to revise
    (selected by identify_samples_needing_revision based on shortfall).
    """
    tasks = []

    for sample_id, responses in needs_revision.items():
        # Revise all responses in the list (already limited by shortfall calculation)
        for r in responses:
            failed_criteria = extract_failed_criteria(r)
            if not failed_criteria:
                continue  # Skip if no failed criteria recorded

            task = RevisionTask(
                sample_id=sample_id,
                persona_name=r['persona_name'],
                subject=r['subject'],
                prompt=r['prompt'],
                initial_explanation=r['initial_explanation'],
                follow_up=r['follow_up'],
                original_response=r['response'],
                weighted_score=r.get('weighted_score', 0),
                critical_pass_rate=r.get('critical_pass_rate', 0),
                failed_criteria=failed_criteria,
                # Include CoT phases if available
                phase0_strategy=r.get('phase0_strategy'),
                phase1_knowledge=r.get('phase1_knowledge'),
                phase1_5_errors=r.get('phase1_5_errors'),
            )
            tasks.append(task)

    return tasks


def build_revision_prompt(task: RevisionTask) -> tuple[str, str]:
    """Build the prompt for revising a failed response."""

    # Format failed criteria
    failed_section = "\n".join([
        f"❌ **{f['severity'].upper()}**: {f['criterion']}\n"
        f"   Evaluator note: {f['explanation']}"
        for f in task.failed_criteria
    ])

    # Include knowledge/error analysis if available
    context_section = ""
    if task.phase1_knowledge:
        context_section += f"\n## Domain Knowledge (from original analysis)\n{task.phase1_knowledge}\n"
    if task.phase1_5_errors:
        context_section += f"\n## Error Analysis (all errors identified)\n{task.phase1_5_errors}\n"

    system = f"""You are an expert {task.subject} tutor revising a response that failed quality checks.

Your task is to write an IMPROVED response that addresses ALL the failed criteria below.

## Failed Criteria (you MUST fix these)
{failed_section}

## Original Response (for reference - improve upon this)
{task.original_response}
{context_section}

## Key Requirements
- Address EVERY failed criterion explicitly
- Keep what was good about the original response
- For fractions, always simplify to lowest terms (e.g., 27/6 = 9/2)
- Show intermediate calculation steps
- Be precise with terminology and mathematical expressions"""

    user = f"""## Original Question
{task.prompt}

## Initial Explanation Provided
{task.initial_explanation}

## Student's Follow-up
{task.follow_up}

---

Write an improved tutoring response that fixes ALL the failed criteria listed above.

Remember:
- You MUST address each failed criterion
- Show your work with explicit intermediate steps
- Simplify all fractions to lowest terms
- Use proper {task.subject} terminology

Write your improved response now:"""

    return system, user


# Retry configuration for revision
import random
import time
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

# Global fallback client for revision (initialized on first balance error)
_revision_fallback_client: Optional[AsyncTogether] = None
_revision_using_fallback = False


async def _revision_token_bucket_refiller():
    """Background task that adds tokens to the rate limiter at exactly RATE_LIMIT_RPS per second."""
    global _revision_rate_limiter
    interval = 1.0 / RATE_LIMIT_RPS  # Time between each token (~33ms for 30 RPS)

    while True:
        await asyncio.sleep(interval)
        # Add a token if below max
        if _revision_rate_limiter._value < RATE_LIMIT_RPS:
            _revision_rate_limiter.release()


async def revision_rate_limit_wait():
    """Acquire a rate limit token. Blocks until one is available."""
    global _revision_rate_limiter, _revision_rate_limiter_task

    if _revision_rate_limiter is None:
        # Initialize with full bucket (30 tokens)
        _revision_rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPS)
        # Start background task to refill tokens
        _revision_rate_limiter_task = asyncio.create_task(_revision_token_bucket_refiller())

    # Acquire a token (blocks if bucket is empty)
    await _revision_rate_limiter.acquire()


async def revise_single_response(
    client: AsyncTogether,
    task: RevisionTask,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict]:
    """Revise a single failed response with retry logic."""
    global _revision_fallback_client, _revision_using_fallback

    current_client = _revision_fallback_client if _revision_using_fallback else client
    task_id = f"{task.sample_id}/{task.persona_name}"

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                # Rate limit before making request
                await revision_rate_limit_wait()

                system, user = build_revision_prompt(task)

                # Wrap API call with timeout to prevent hanging
                resp = await asyncio.wait_for(
                    current_client.chat.completions.create(
                        model=REVISION_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=MAX_TOKENS_REVISION,
                        temperature=REVISION_TEMPERATURE,
                    ),
                    timeout=PER_CALL_TIMEOUT
                )

                revised_response = resp.choices[0].message.content

                return {
                    'sample_id': task.sample_id,
                    'persona_name': task.persona_name,
                    'subject': task.subject,
                    'prompt': task.prompt,
                    'initial_explanation': task.initial_explanation,
                    'follow_up': task.follow_up,
                    'response': revised_response,
                    'original_response': task.original_response,
                    'original_score': task.weighted_score,
                    'original_critical_rate': task.critical_pass_rate,
                    'failed_criteria_count': len(task.failed_criteria),
                    'failed_criteria': task.failed_criteria,
                    'model': REVISION_MODEL,
                    'timestamp': datetime.now().isoformat(),
                    'is_revision': True,
                    # Preserve CoT phases if available
                    'phase0_strategy': task.phase0_strategy,
                    'phase1_knowledge': task.phase1_knowledge,
                    'phase1_5_errors': task.phase1_5_errors,
                }

            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    continue
                print(f"\n[Timeout] {task_id} failed after {MAX_RETRIES} attempts")
                return None

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str
                is_overloaded = "503" in error_str or "502" in error_str or "500" in error_str or "overloaded" in error_str or "not ready" in error_str or "cloudflare" in error_str
                is_balance_error = "balance" in error_str or "insufficient" in error_str or "credit" in error_str
                is_api_key_error = "api key" in error_str or "unauthorized" in error_str or "401" in error_str

                # Check for balance/API key errors - switch to fallback
                if (is_balance_error or is_api_key_error) and not _revision_using_fallback and FALLBACK_API_KEY:
                    print(f"\n[API Key/Balance Error] Switching to fallback API key for revision...")
                    _revision_fallback_client = AsyncTogether(api_key=FALLBACK_API_KEY)
                    _revision_using_fallback = True
                    current_client = _revision_fallback_client
                    continue  # Retry with fallback

                if (is_rate_limit or is_overloaded) and attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    continue

                print(f"\nError revising {task_id}: {e}")
                return None

        return None


def load_completed_revisions(output_path: Path) -> set:
    """Load already revised task IDs from output file."""
    completed = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    task_id = f"{r['sample_id']}_{r['persona_name']}"
                    completed.add(task_id)
                except:
                    continue
    return completed


async def run_revision(
    input_path: Path,
    output_path: Path,
    critical_threshold: float = 0.70,
    overall_threshold: float = 0.70,
    min_pass_rate: float = 0.25,
    dry_run: bool = False,
    resume: bool = False,
) -> List[Dict]:
    """Run the revision pipeline to ensure minimum pass rate per sample."""

    print(f"\n{'='*60}")
    print("RESPONSE REVISION (Ensure {:.0%} Pass Rate)".format(min_pass_rate))
    print(f"{'='*60}")
    print(f"Model: {REVISION_MODEL}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Min pass rate per sample: {100*min_pass_rate:.0f}%")
    print(f"Critical threshold: {100*critical_threshold:.0f}%")
    print(f"Overall threshold: {100*overall_threshold:.0f}%")
    print(f"Resume: {resume}")
    print(f"{'='*60}\n")

    # Load graded/rejected responses
    print("Loading responses...")
    all_responses = load_graded_responses(input_path)
    print(f"Loaded {len(all_responses)} responses")

    # Identify samples needing revision to meet min_pass_rate
    print(f"\nIdentifying samples below {100*min_pass_rate:.0f}% pass rate...")
    needs_revision = identify_samples_needing_revision(
        all_responses, critical_threshold, overall_threshold, min_pass_rate
    )
    total_to_revise = sum(len(v) for v in needs_revision.values())
    print(f"Found {len(needs_revision)} samples needing revision ({total_to_revise} total responses to revise)")

    if not needs_revision:
        print(f"All samples already have >= {100*min_pass_rate:.0f}% pass rate!")
        return []

    # Create revision tasks
    tasks = create_revision_tasks(needs_revision)
    print(f"Created {len(tasks)} revision tasks")

    # Handle resume - filter out already completed
    if resume:
        completed = load_completed_revisions(output_path)
        original_count = len(tasks)
        tasks = [t for t in tasks if f"{t.sample_id}_{t.persona_name}" not in completed]
        print(f"Resume: {len(completed)} already revised, {len(tasks)} remaining (of {original_count})")

    if dry_run:
        print(f"\n[DRY RUN] Would revise {len(tasks)} responses")
        print("\nSample revision tasks:")
        for t in tasks[:5]:
            print(f"  - {t.sample_id} / {t.persona_name}")
            print(f"    Score: {t.weighted_score:.2f}, Critical: {t.critical_pass_rate:.2f}")
            print(f"    Failed criteria: {len(t.failed_criteria)}")
        return []

    # Initialize client
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_AI_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY required")

    client = AsyncTogether(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Run revisions
    print(f"\nRevising {len(tasks)} responses...")
    async_tasks = [
        revise_single_response(client, task, semaphore)
        for task in tasks
    ]
    results = await tqdm_asyncio.gather(*async_tasks, desc="Revising")

    # Filter out None results
    revised = [r for r in results if r is not None]
    print(f"Successfully revised {len(revised)} responses")

    # Save revised responses (append if resuming)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if resume else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for r in revised:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n✅ Saved {len(revised)} revised responses to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("REVISION SUMMARY")
    print(f"{'='*60}")
    print(f"Samples needing revision: {len(needs_revision)}")
    print(f"Revision attempts: {len(tasks)}")
    print(f"Successful revisions: {len(revised)}")
    print(f"\n📝 Next step: Grade the revised responses to see improvement")
    print(f"{'='*60}\n")

    return revised


def main():
    parser = argparse.ArgumentParser(description="Revise failed responses using rubric feedback")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL with graded/rejected responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for revised responses")
    parser.add_argument("--critical-threshold", type=float, default=0.70,
                        help="Critical pass rate threshold (default: 0.70)")
    parser.add_argument("--overall-threshold", type=float, default=0.70,
                        help="Overall score threshold (default: 0.70)")
    parser.add_argument("--min-pass-rate", type=float, default=0.25,
                        help="Minimum pass rate to ensure per sample (default: 0.25)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be revised without running")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already revised)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_revised.jsonl"

    asyncio.run(run_revision(
        input_path=input_path,
        output_path=output_path,
        critical_threshold=args.critical_threshold,
        overall_threshold=args.overall_threshold,
        min_pass_rate=args.min_pass_rate,
        dry_run=args.dry_run,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
