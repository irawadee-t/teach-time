"""
Filter generated tutoring samples using LLM judge evaluation.

This script:
1. Reads samples from synthetic/tutoring_samples_generated.jsonl
2. Evaluates gold_response against rubrics using Claude Opus 4.5
3. Keeps samples where >80% of critical rubrics pass
4. Outputs to synthetic/tutoring_finetune_openai.jsonl (OpenAI format)
5. Generates detailed statistics in synthetic/filtering_statistics.json
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

from anthropic import AsyncAnthropic
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

# Configuration
INPUT_PATH = "synthetic/tutoring_samples_generated.jsonl"
OUTPUT_PATH = "synthetic/tutoring_finetune_openai.jsonl"
STATS_PATH = "synthetic/filtering_statistics.json"
CRITICAL_PASS_THRESHOLD = 0.80  # Keep if >80% critical rubrics pass
MAX_CONCURRENT_REQUESTS = 50

# Initialize Async Anthropic client
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# MODEL: Claude Opus 4.5
MODEL = "claude-opus-4-5-20251101"

# OpenAI fine-tuning system prompt (standardized across all samples)
OPENAI_SYSTEM_PROMPT = (
    "You are an expert tutor. A student asked a question, received an explanation, "
    "and now has a follow-up question showing confusion. Address their specific "
    "misconception while guiding them toward understanding."
)


# ============================================================================
# EVALUATION SCHEMA
# ============================================================================

class RubricEvaluation(BaseModel):
    """Evaluation result for a single rubric."""
    criteria: str
    passes: bool
    reasoning: str


class SampleEvaluation(BaseModel):
    """Complete evaluation of a sample's gold response."""
    rubric_evaluations: list[RubricEvaluation]
    overall_quality_score: Literal[1, 2, 3, 4, 5]  # 1=poor, 5=excellent
    strengths: list[str]
    weaknesses: list[str]


# ============================================================================
# EVALUATION PROMPT
# ============================================================================

def get_evaluation_prompt(sample: dict) -> str:
    """Create evaluation prompt for a sample."""

    # Extract messages
    user_message = sample['messages'][1]['content']
    gold_response = sample['messages'][2]['content']

    # Extract rubrics
    rubrics = sample.get('rubrics', [])

    rubrics_text = "\n\n".join([
        f"**Rubric {i+1}**\n"
        f"- Severity: {r.get('attributes', {}).get('severity', 'unknown')}\n"
        f"- Tutoring Skill: {r.get('attributes', {}).get('tutoring_skill', 'unknown')}\n"
        f"- Eval Dimension: {r.get('attributes', {}).get('eval_dimension', 'unknown')}\n"
        f"- Criteria: {r.get('criteria', 'N/A')}"
        for i, r in enumerate(rubrics)
    ])

    return f"""You are evaluating a tutoring response against specific rubrics.

**STUDENT'S CONTEXT**:
{user_message}

**TUTOR'S RESPONSE**:
{gold_response}

**EVALUATION RUBRICS**:
{rubrics_text}

**YOUR TASK**:
For each rubric, determine if the tutor's response passes the criteria. Be objective and strict.

For the overall quality score:
- 5 = Excellent tutoring (empathetic, clear, accurate, addresses misconception perfectly)
- 4 = Good tutoring (mostly effective with minor issues)
- 3 = Adequate tutoring (addresses question but misses nuances)
- 2 = Poor tutoring (inaccurate, unhelpful, or condescending)
- 1 = Very poor tutoring (incorrect, harmful, or completely off-topic)

Provide:
1. Evaluation for each rubric (passes: true/false, reasoning)
2. Overall quality score (1-5)
3. Strengths of the response (2-4 bullet points)
4. Weaknesses of the response (2-4 bullet points)

Be thorough and objective in your evaluation."""


# ============================================================================
# ASYNC EVALUATION
# ============================================================================

async def evaluate_sample(sample: dict, semaphore: asyncio.Semaphore) -> Tuple[dict, Optional[SampleEvaluation]]:
    """Evaluate a single sample using Claude."""

    async with semaphore:
        try:
            prompt = get_evaluation_prompt(sample)

            response = await client.beta.messages.parse(
                model=MODEL,
                max_tokens=2048,
                temperature=0.3,  # Lower temp for more consistent evaluation
                output_format=SampleEvaluation,
                betas=["structured-outputs-2025-11-13"],
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return (sample, response.parsed_output)

        except Exception as e:
            print(f"\nError evaluating sample {sample.get('id', 'unknown')}: {e}")
            return (sample, None)


async def evaluate_all_samples(samples: list[dict]) -> list[Tuple[dict, SampleEvaluation]]:
    """Evaluate all samples concurrently."""

    print(f"\n{'='*60}")
    print(f"EVALUATING {len(samples)} SAMPLES")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS} evaluations at a time")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [evaluate_sample(sample, semaphore) for sample in samples]

    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating samples")

    # Filter out failed evaluations
    successful = [(s, e) for s, e in results if e is not None]

    print(f"\nSuccessfully evaluated: {len(successful)}/{len(samples)}")

    return successful


# ============================================================================
# FILTERING LOGIC
# ============================================================================

def calculate_critical_pass_rate(sample: dict, evaluation: SampleEvaluation) -> float:
    """Calculate percentage of critical rubrics that pass."""

    rubrics = sample.get('rubrics', [])

    # Find critical rubrics
    critical_indices = [
        i for i, r in enumerate(rubrics)
        if r.get('attributes', {}).get('severity') == 'critical'
    ]

    if not critical_indices:
        # No critical rubrics - consider this a pass
        return 1.0

    # Count how many critical rubrics pass
    critical_passes = sum(
        1 for i in critical_indices
        if i < len(evaluation.rubric_evaluations) and evaluation.rubric_evaluations[i].passes
    )

    return critical_passes / len(critical_indices)


def should_keep_sample(sample: dict, evaluation: SampleEvaluation, threshold: float = CRITICAL_PASS_THRESHOLD) -> bool:
    """Determine if sample should be kept based on evaluation."""

    critical_pass_rate = calculate_critical_pass_rate(sample, evaluation)
    return critical_pass_rate > threshold


# ============================================================================
# OPENAI FORMAT CONVERSION
# ============================================================================

def format_for_openai(sample: dict) -> dict:
    """Convert sample to OpenAI fine-tuning format."""

    user_content = sample['messages'][1]['content']
    assistant_content = sample['messages'][2]['content']

    return {
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


# ============================================================================
# STATISTICS GENERATION
# ============================================================================

def generate_statistics(results: list[Tuple[dict, SampleEvaluation, bool]]) -> dict:
    """Generate detailed filtering statistics."""

    total = len(results)
    kept = sum(1 for _, _, keep in results if keep)
    removed = total - kept

    # Quality score distribution
    quality_scores = [eval.overall_quality_score for _, eval, _ in results]
    quality_distribution = {
        i: quality_scores.count(i) for i in range(1, 6)
    }

    # Critical pass rates
    critical_pass_rates = [
        calculate_critical_pass_rate(sample, eval)
        for sample, eval, _ in results
    ]
    avg_critical_pass_rate = sum(critical_pass_rates) / len(critical_pass_rates) if critical_pass_rates else 0.0

    # Common strengths/weaknesses
    all_strengths = []
    all_weaknesses = []
    for _, eval, _ in results:
        all_strengths.extend(eval.strengths)
        all_weaknesses.extend(eval.weaknesses)

    return {
        "total_samples": total,
        "kept_samples": kept,
        "removed_samples": removed,
        "keep_rate": kept / total if total > 0 else 0,
        "average_critical_pass_rate": avg_critical_pass_rate,
        "quality_score_distribution": quality_distribution,
        "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        "filtering_threshold": CRITICAL_PASS_THRESHOLD,
        "common_strengths": list(set(all_strengths))[:10],
        "common_weaknesses": list(set(all_weaknesses))[:10],
    }


# ============================================================================
# MAIN FILTERING PIPELINE
# ============================================================================

async def filter_samples(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    stats_path: str = STATS_PATH
) -> list[dict]:
    """Main filtering pipeline."""

    print(f"\n{'='*60}")
    print(f"FILTERING PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Stats: {stats_path}")
    print(f"Threshold: >{int(CRITICAL_PASS_THRESHOLD*100)}% critical rubrics pass")
    print(f"{'='*60}\n")

    # Load samples
    print("Loading samples...")
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples\n")

    # Evaluate all samples
    evaluated = await evaluate_all_samples(samples)

    # Filter samples
    print("\nFiltering samples...")
    results = []
    kept_samples = []

    for sample, evaluation in evaluated:
        keep = should_keep_sample(sample, evaluation)
        results.append((sample, evaluation, keep))

        if keep:
            kept_samples.append(format_for_openai(sample))

    print(f"Kept: {len(kept_samples)}/{len(samples)} samples")

    # Save filtered samples
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in kept_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path}")

    # Generate and save statistics
    stats = generate_statistics(results)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {stats_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Kept: {stats['kept_samples']} ({stats['keep_rate']*100:.1f}%)")
    print(f"Removed: {stats['removed_samples']}")
    print(f"Avg critical pass rate: {stats['average_critical_pass_rate']*100:.1f}%")
    print(f"Avg quality score: {stats['average_quality_score']:.2f}/5.0")
    print(f"\nQuality distribution:")
    for score, count in stats['quality_score_distribution'].items():
        print(f"  Score {score}: {count} samples")
    print(f"{'='*60}\n")

    return kept_samples


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(filter_samples())
