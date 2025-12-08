"""
Generate synthetic tutoring training data using Anthropic's structured outputs.

This script:
1. Reads seed examples from "Text Tutoring Prompts.csv"
2. Generates 4 variations per seed using Claude Opus 4.5
3. Outputs to synthetic/tutoring_samples_generated.jsonl
4. Uses async execution with 50 concurrent requests for speed
"""

import asyncio
import csv
import json
import os
import random
from pathlib import Path
from typing import Literal, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

# Configuration
NUM_SEEDS = 250
SAMPLES_PER_SEED = 4
MAX_CONCURRENT_REQUESTS = 50
OUTPUT_PATH = "synthetic/tutoring_samples_generated.jsonl"
CHECKPOINT_INTERVAL = 25  # Save checkpoint every N seeds

# Initialize Async Anthropic client
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# MODEL: Claude Opus 4.5
MODEL = "claude-opus-4-5-20251101"


# ============================================================================
# TUTORBENCH RUBRIC SCHEMA (EXACT MATCH)
# ============================================================================

class RubricAttributes(BaseModel):
    """TutorBench rubric attributes with exact field names and values."""
    explicitness: Literal["explicit", "implicit"]
    objectivity: Literal["objective", "subjective"]
    severity: Literal["critical", "not_critical"]
    tutoring_skill: Literal[
        "Not applicable",
        "Stating definitions/ formulae/ theorems/ laws",
        "Identifying Core difficulty/ misconception attribution ",  # trailing space
        "Identifying incorrect steps by student",
        "Identifying correct steps by student",
        "Step by step help/ analysis ",  # trailing space
        "Includes examples/ analogy ",  # trailing space
        "Asks questions to guide students ",  # trailing space
        "Provides alternative solutions/ paths/",
        "Providing alternative solutions/ paths/",  # variant Claude generates
    ]
    eval_dimension: Literal[
        "instruction_following",
        "instruction_following, student_level_calibration",
        "truthfulness",
        "truthfulness, student_level_calibration",
        "style_tone, emotional_component ",  # trailing space
        "style_tone",
        "conciseness_relevance",
        "conciseness_relevance, student_level_calibration",
        "style_tone, student_level_calibration",
        "student_level_calibration",  # standalone option
    ]


class Rubric(BaseModel):
    """TutorBench rubric with nested attributes."""
    attributes: RubricAttributes
    criteria: str


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class SampleMetadata(BaseModel):
    """Metadata for the generated sample."""
    subject: str
    topic: str
    misconception_type: str


class ScenarioContext(BaseModel):
    """Tutoring scenario context WITHOUT gold response yet."""
    question: str
    initial_explanation: str
    student_followup: str
    metadata: SampleMetadata
    rubrics: list[Rubric]


class GoldResponse(BaseModel):
    """Just the gold response."""
    response: str


class GeneratedSample(BaseModel):
    """Complete generated tutoring sample with rubrics."""
    id: str
    messages: list[Message]
    metadata: SampleMetadata
    rubrics: list[Rubric]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_messages

    @classmethod
    def validate_messages(cls, v):
        if 'messages' in v and len(v['messages']) != 3:
            raise ValueError(f'messages must have exactly 3 elements, got {len(v["messages"])}')
        return v


# ============================================================================
# SUBJECTS (AVOIDING TUTORBENCH OVERLAP)
# ============================================================================

SUBJECTS = [
    # Economics
    "microeconomics", "macroeconomics", "accounting", "finance",

    # Social Sciences
    "psychology", "sociology", "political_science", "philosophy",

    # History
    "world_history", "us_history",

    # Mathematics (non-TutorBench)
    "linear_algebra", "discrete_math", "number_theory", "probability",

    # Chemistry (non-TutorBench)
    "organic_chemistry", "biochemistry",

    # Biology (non-TutorBench)
    "ecology", "genetics",

    # Physics (non-TutorBench)
    "electromagnetism", "thermodynamics", "fluid_dynamics", "quantum_mechanics",

    # Computer Science
    "data_structures", "algorithms", "machine_learning", "databases",

    # Engineering
    "engineering_statics", "engineering_dynamics",
]


# ============================================================================
# MISCONCEPTION TYPES
# ============================================================================

MISCONCEPTION_TYPES = [
    "why_is_step_necessary",
    "how_concepts_relate",
    "counterexample_contradiction",
    "overgeneralization",
    "confusing_similar_terms",
    "incorrect_assumption",
    "procedural_vs_conceptual",
    "misapplying_formula",
]


# ============================================================================
# GENERATION PROMPTS
# ============================================================================

def get_scenario_context_prompt(seed_example: dict, variation_num: int, subject: str, misconception_type: str) -> str:
    """Create prompt for generating scenario context (WITHOUT gold response yet)."""

    return f"""You are generating a NEW tutoring scenario context for training data.

**Context:**
- Subject: {subject}
- Misconception Type: {misconception_type}
- Variation: {variation_num} (make it distinct from other variations)

**Seed Example** (for reference/inspiration only - DO NOT COPY):
Question: {seed_example['question'][:300]}...
Initial Explanation: {seed_example['initial_explanation'][:400]}...
Student Follow-up: {seed_example['student_followup'][:200]}...

**YOUR TASK:**
Generate a COMPLETELY NEW tutoring scenario with:

1. **Question**: A NEW original question about {subject} (2-4 sentences)

2. **Initial Explanation**: A NEW initial explanation by the tutor (4-8 sentences)
   - Should be correct but may be incomplete or not address all student concerns
   - Sets up the student's confusion

3. **Student Follow-up**: A NEW follow-up question showing confusion/{misconception_type} (1-3 sentences)
   - Shows a specific misconception or confusion point
   - This is what the tutor's response will need to address

4. **Metadata**:
   - subject: "{subject}"
   - topic: (specific concept/topic within {subject}, e.g., "equilibrium shifts", "marginal analysis")
   - misconception_type: "{misconception_type}"

5. **Rubrics** (6-8 evaluation criteria): How to evaluate if a tutor's response is good:
   - Mix of critical (severity: "critical") and non-critical (severity: "not_critical")
   - Cover: instruction_following, truthfulness, style_tone, conciseness_relevance, student_level_calibration
   - Nested "attributes" with: explicitness, objectivity, severity, tutoring_skill, eval_dimension
   - Criteria describes what a good response SHOULD do (independent of any specific response)

**Example Rubric:**
{{
  "attributes": {{
    "explicitness": "explicit",
    "objectivity": "objective",
    "severity": "critical",
    "tutoring_skill": "Identifying Core difficulty/ misconception attribution ",
    "eval_dimension": "instruction_following"
  }},
  "criteria": "Response correctly identifies the student's specific misconception about [concept]"
}}

**CRITICAL REQUIREMENTS:**
- Generate ORIGINAL content (don't copy seed example)
- Rubrics should be INDEPENDENT criteria for what makes a good tutoring response
- DO NOT generate the tutor's response yet - that comes in the next step

Return the scenario context structure."""


def get_gold_response_prompt(scenario: ScenarioContext) -> str:
    """Create prompt for generating gold response to meet the scenario rubrics."""

    rubrics_text = "\n\n".join([
        f"**Rubric {i+1}** ({'CRITICAL' if r.attributes.severity == 'critical' else 'non-critical'}):\n"
        f"- Tutoring Skill: {r.attributes.tutoring_skill}\n"
        f"- Eval Dimension: {r.attributes.eval_dimension}\n"
        f"- Criteria: {r.criteria}"
        for i, r in enumerate(scenario.rubrics)
    ])

    return f"""You are an expert {scenario.metadata.subject} tutor. Generate a high-quality tutoring response.

**STUDENT'S CONTEXT:**

## Question
{scenario.question}

## Initial Explanation
{scenario.initial_explanation}

## Student Follow-up
{scenario.student_followup}

**EVALUATION RUBRICS (your response will be judged against these):**
{rubrics_text}

**YOUR TASK:**
Generate a gold-standard tutoring response that:
- Identifies the specific misconception in the student's follow-up
- Provides clear, accurate correction
- Uses encouraging, empathetic tone (never condescending)
- Guides the student toward understanding (Socratic method when appropriate)
- Includes examples or analogies if helpful
- Meets as many of the rubric criteria as possible, especially critical ones

**VARIATION INSTRUCTIONS (CRITICAL - AVOID FORMULAIC PATTERNS):**
- Do NOT always start with "Great question" or similar praise
- Do NOT always end with a reflective question like "Does this help?" or "Can you think of..."
- Vary response length: some should be 3-5 sentences (concise), others 8-12 sentences (detailed)
- Sometimes dive directly into the correction without preamble
- Occasionally use shorter, punchier explanations for simple misconceptions
- Mix up your tutoring approaches: sometimes Socratic, sometimes direct explanation, sometimes analogy-first

**CRITICAL REQUIREMENTS:**
- This is the "gold standard" response we're training models to produce
- Be pedagogically sound and factually accurate
- Address the student's specific confusion about {scenario.metadata.misconception_type}
- AVOID repetitive opening/closing patterns across samples

Return just the response text."""


# ============================================================================
# ASYNC GENERATION
# ============================================================================

async def generate_single_sample(
    seed_example: dict,
    seed_idx: int,
    variation_num: int,
    semaphore: asyncio.Semaphore
) -> Optional[GeneratedSample]:
    """Generate a single sample with concurrency control using two-step process."""

    async with semaphore:
        try:
            # Select subject and misconception type
            subject = SUBJECTS[(seed_idx * SAMPLES_PER_SEED + variation_num) % len(SUBJECTS)]
            misconception_type = MISCONCEPTION_TYPES[(seed_idx * SAMPLES_PER_SEED + variation_num) % len(MISCONCEPTION_TYPES)]

            # ===== STEP 1: Generate scenario context + rubrics (WITHOUT gold response) =====
            scenario_prompt = get_scenario_context_prompt(seed_example, variation_num, subject, misconception_type)

            scenario_response = await client.beta.messages.parse(
                model=MODEL,
                max_tokens=3072,
                temperature=1.0,
                output_format=ScenarioContext,
                betas=["structured-outputs-2025-11-13"],
                messages=[{
                    "role": "user",
                    "content": scenario_prompt
                }]
            )

            scenario = scenario_response.parsed_output

            # ===== STEP 2: Generate gold response to meet those independent rubrics =====
            gold_response_prompt = get_gold_response_prompt(scenario)

            gold_response_response = await client.beta.messages.parse(
                model=MODEL,
                max_tokens=1024,
                temperature=0.7,  # Slightly lower temp for more consistent quality
                output_format=GoldResponse,
                betas=["structured-outputs-2025-11-13"],
                messages=[{
                    "role": "user",
                    "content": gold_response_prompt
                }]
            )

            gold_response = gold_response_response.parsed_output

            # ===== STEP 3: Combine into final GeneratedSample structure =====
            # Build the user message with proper formatting
            user_content = f"""## Question
{scenario.question}

## Initial Explanation
{scenario.initial_explanation}

## Student Follow-up
{scenario.student_followup}"""

            # Create the messages array
            messages = [
                Message(role="system", content=f"You are an expert {scenario.metadata.subject} tutor helping a student who is confused after your initial explanation."),
                Message(role="user", content=user_content),
                Message(role="assistant", content=gold_response.response)
            ]

            # Create the final sample
            sample = GeneratedSample(
                id=f"seed{seed_idx:03d}_var{variation_num}",
                messages=messages,
                metadata=scenario.metadata,
                rubrics=scenario.rubrics
            )

            return sample

        except Exception as e:
            print(f"\nError generating seed{seed_idx:03d}_var{variation_num}: {e}")
            return None


async def generate_batch(seed_examples: list[dict], start_idx: int, batch_size: int) -> list[GeneratedSample]:
    """Generate a batch of samples concurrently."""

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for i in range(start_idx, min(start_idx + batch_size, len(seed_examples))):
        seed_example = seed_examples[i]

        for variation_num in range(SAMPLES_PER_SEED):
            task = generate_single_sample(seed_example, i, variation_num, semaphore)
            tasks.append(task)

    # Run all tasks concurrently with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc=f"Generating batch {start_idx//batch_size + 1}")

    # Filter out None results (errors)
    return [r for r in results if r is not None]


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def load_seed_examples(csv_path: str, num_seeds: int) -> list[dict]:
    """Load seed examples from CSV."""

    examples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                'question': row['PROMPT'],
                'initial_explanation': row['UC1_INITIAL_EXPLANATION'],
                'student_followup': row['FOLLOW_UP_PROMPT'],
            })

    # Shuffle and take num_seeds
    random.seed(42)
    random.shuffle(examples)
    return examples[:num_seeds]


def save_samples(samples: list[GeneratedSample], output_path: str):
    """Save samples to JSONL file."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'a', encoding='utf-8') as f:
        for sample in samples:
            # Convert Pydantic model to dict
            sample_dict = sample.model_dump()
            f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')


def save_checkpoint(samples: list[GeneratedSample], checkpoint_num: int):
    """Save checkpoint file."""

    checkpoint_path = f"{OUTPUT_PATH.rsplit('.', 1)[0]}_checkpoint_{checkpoint_num}.jsonl"
    save_samples(samples, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")


async def generate_all_samples(
    csv_path: str,
    output_path: str,
    num_seeds: int = NUM_SEEDS,
    samples_per_seed: int = SAMPLES_PER_SEED
) -> list[GeneratedSample]:
    """Generate all samples using async/concurrent execution."""

    print(f"\n{'='*60}")
    print(f"GENERATING {num_seeds * samples_per_seed} SAMPLES")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Seeds: {num_seeds}")
    print(f"Variations per seed: {samples_per_seed}")
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS} requests at a time")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Load seed examples
    print("Loading seed examples...")
    seed_examples = load_seed_examples(csv_path, num_seeds)
    print(f"Loaded {len(seed_examples)} seed examples\n")

    # Clear output file if it exists
    if Path(output_path).exists():
        Path(output_path).unlink()

    # Generate in batches with checkpoints
    all_samples = []

    for batch_start in range(0, num_seeds, CHECKPOINT_INTERVAL):
        batch_samples = await generate_batch(seed_examples, batch_start, CHECKPOINT_INTERVAL)
        all_samples.extend(batch_samples)

        # Save batch to main file
        save_samples(batch_samples, output_path)

        # Save checkpoint
        if batch_start + CHECKPOINT_INTERVAL < num_seeds:
            save_checkpoint(batch_samples, batch_start // CHECKPOINT_INTERVAL + 1)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples generated: {len(all_samples)}")
    print(f"Total expected: {num_seeds * samples_per_seed}")
    print(f"Success rate: {100 * len(all_samples) / (num_seeds * samples_per_seed):.1f}%")
    print(f"Output file: {output_path}")
    print(f"{'='*60}\n")

    return all_samples


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(generate_all_samples(
        csv_path="Text Tutoring Prompts.csv",
        output_path=OUTPUT_PATH,
        num_seeds=NUM_SEEDS,
        samples_per_seed=SAMPLES_PER_SEED
    ))
