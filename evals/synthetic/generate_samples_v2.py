"""
Generate synthetic tutoring responses using persona-based diversity.

Based on synthetic_data_plan.md:
- Uses Together AI with Qwen/Qwen3-Next-80B-A3B-Instruct for generation
- Loads from evals/train.csv (556 samples)
- 8 tutor personas for diversity
- Does NOT include rubrics in generation prompt
- Async batching with checkpointing

Usage:
    # Test on a few samples first
    python -m evals.synthetic.generate_samples_v2 --max-samples 5 --dry-run

    # Generate all with checkpointing
    python -m evals.synthetic.generate_samples_v2

    # Resume from checkpoint
    python -m evals.synthetic.generate_samples_v2 --resume
"""

import asyncio
import csv
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env.local (preferred) or .env
_project_root = Path(__file__).parent.parent.parent
for env_file in [".env.local", ".env", "evals/.env.local", "evals/.env"]:
    env_path = _project_root / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break

from tqdm.asyncio import tqdm_asyncio
from together import AsyncTogether

# Configuration
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAX_CONCURRENT_REQUESTS = 50
MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.9  # Slightly higher for more variation
DEFAULT_SAMPLES_PER_PERSONA = 3  # Generate multiple samples per persona for more diversity
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N samples

# Paths
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR.parent
TRAIN_CSV = EVALS_DIR / "train.csv"
OUTPUT_DIR = SCRIPT_DIR / "generated_responses"


# ============================================================================
# TUTOR PERSONAS (from synthetic_data_plan.md)
# ============================================================================

TUTOR_PERSONAS = [
    {
        "name": "empathetic_validator",
        "description": "A patient tutor who always acknowledges the student's feelings and validates their effort before addressing misconceptions. Uses phrases like 'I understand why that's confusing' and 'That's a really thoughtful question.' Gently corrects errors while maintaining the student's confidence."
    },
    {
        "name": "socratic_questioner",
        "description": "A Socratic tutor who guides through questions rather than direct explanation. Rarely gives answers outright - instead asks probing questions like 'What do you think would happen if...' and 'Can you walk me through your reasoning?' Helps students discover insights themselves rather than being told."
    },
    {
        "name": "direct_clarifier",
        "description": "A clear, direct tutor who efficiently identifies the exact misconception and provides targeted correction. Gets to the point quickly while remaining supportive. Names specific errors explicitly: 'The issue here is that you...' Good at pinpointing exactly where reasoning went wrong."
    },
    {
        "name": "analogy_builder",
        "description": "A tutor who excels at relating abstract concepts to concrete, everyday examples. Frequently uses analogies and real-world scenarios to build intuition. Makes the unfamiliar familiar by connecting to things the student already understands."
    },
    {
        "name": "step_by_step_guide",
        "description": "A methodical tutor who breaks complex problems into small, manageable steps. Numbers their explanations, shows intermediate work, and checks understanding at each stage. Walks through calculations explicitly when procedural clarity helps."
    },
    {
        "name": "conceptual_connector",
        "description": "A tutor who emphasizes how concepts relate to the bigger picture. Draws connections between the current topic and previously learned material. Explains underlying principles and laws, names relevant theorems or formulas, and shows why they apply here."
    },
    {
        "name": "error_pattern_expert",
        "description": "A tutor who specializes in identifying common error patterns. Explicitly names the type of mistake ('This is a classic sign error' or 'This is a common confusion between X and Y') and explains why it's a frequent pitfall. Helps students recognize these patterns in the future."
    },
    {
        "name": "alternative_pathfinder",
        "description": "A tutor who presents multiple ways to approach problems. When explaining a concept, offers 'another way to think about this...' and shows different solution strategies or perspectives. Helps students see that there's often more than one valid approach."
    }
]


@dataclass
class GenerationTask:
    """A single generation task."""
    sample_idx: int
    sample_id: str
    persona_idx: int
    persona_name: str
    variation_idx: int  # Which variation within the same persona
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str


@dataclass
class GeneratedResponse:
    """A generated response with metadata."""
    sample_id: str
    persona_name: str
    variation_idx: int
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str
    response: str
    model: str
    timestamp: str


# ============================================================================
# SUBJECT-SPECIFIC GUIDANCE
# ============================================================================

SUBJECT_GUIDANCE = {
    "Biology": """
**Biology-Specific Guidance:**
- Use precise scientific terminology (e.g., "siRNA", "RISC complex", "Argonaute protein") - students need to learn exact terms
- Explain molecular mechanisms with specific details (enzymes, pathways, locations in the cell)
- When correcting misconceptions, clarify the exact biological process that's confused
- Reference specific structures, organelles, or molecules by their proper names
- Connect mechanisms to their biological significance and real-world applications
""",
    "Chemistry": """
**Chemistry-Specific Guidance:**
- Use proper chemical notation, formulas, and equation formats
- Name specific laws, principles, and theories (Law of Mass Action, Le Chatelier's Principle, etc.)
- Explain reaction mechanisms with precise terminology (nucleophile, electrophile, carbocation stability)
- When discussing equilibrium, rates, or thermodynamics, reference the relevant equations
- Connect molecular-level explanations to macroscopic observations
""",
    "Physics": """
**Physics-Specific Guidance:**
- Use precise physical terminology (torque, moment arm, centripetal acceleration, etc.)
- Reference specific laws and principles by name (Newton's Laws, Conservation of Energy, etc.)
- When explaining forces, be explicit about their points of application and directions
- Show how distributed forces can be modeled as resultants at specific points
- Connect mathematical formulations to physical intuition
""",
    "Calculus": """
**Calculus-Specific Guidance:**
- Name relevant theorems and techniques (Chain Rule, Fundamental Theorem, integration by substitution)
- When multiple solution methods exist, acknowledge this but explain the chosen approach
- Be explicit about setup: what's the variable of integration, what are the bounds, why?
- For related rates and optimization, emphasize identifying what's changing and what's constant
- Show intermediate steps in calculations when procedural clarity helps
""",
    "Computer Science": """
**Computer Science-Specific Guidance:**
- Use precise terminology (two's complement, buffer overflow, bitwise operations, etc.)
- When teaching algorithms or representations, guide students through the logic step-by-step
- For binary/number representation, help students discover patterns rather than just giving formulas
- Reference the underlying computer architecture concepts when relevant
- Connect abstract concepts to concrete examples with actual numbers
""",
    "Statistics": """
**Statistics-Specific Guidance:**
- Name specific tests, distributions, and conditions (z-test, t-test, normality assumption, etc.)
- Be explicit about checking conditions/assumptions for statistical inference
- When calculations are involved, show the formula and explain each component
- Distinguish between conceptual understanding and computational procedure
- Connect statistical concepts to real-world interpretation of results
""",
}

# Default guidance for subjects not explicitly listed
DEFAULT_SUBJECT_GUIDANCE = """
**General Guidance:**
- Use precise, domain-appropriate terminology that students need to learn
- Name relevant principles, laws, theorems, or frameworks explicitly
- When correcting errors, be specific about what's wrong and why
- Connect concepts to the broader framework of the subject
"""


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

# Core tutoring principles that encourage good pedagogy without being rubric-specific
TUTORING_PRINCIPLES = """
Before responding, think carefully about:

1. **What does this student actually understand vs. misunderstand?**
   - Identify specific errors or gaps in their reasoning
   - Acknowledge what they got right before addressing mistakes
   - Name the type of error if it's a common pattern

2. **What should I reveal vs. help them discover?**
   - Guide toward insights rather than giving away answers
   - Ask yourself: "Will telling them this rob them of a learning moment?"
   - For procedural questions, showing steps is fine; for conceptual gaps, questions work better
   - Don't solve the entire problem for them if they're close

3. **What prerequisite knowledge or formulas are relevant?**
   - Reference foundational concepts they should connect to
   - If a standard formula/law/theorem applies, name it and explain why
   - Connect to the bigger picture of the subject

4. **What's the clearest way to explain this specific confusion?**
   - Match your explanation to their level of understanding
   - Use concrete examples, analogies, or visualizations when helpful
   - Break complex ideas into digestible pieces
"""


def get_subject_guidance(subject: str) -> str:
    """Get subject-specific guidance, with fallback to default."""
    return SUBJECT_GUIDANCE.get(subject, DEFAULT_SUBJECT_GUIDANCE)


def build_generation_prompt(task: GenerationTask, persona: dict) -> tuple[str, str]:
    """
    Build system and user prompts for generation.

    Includes tutoring principles and subject-specific guidance to encourage
    good pedagogy without encoding specific rubric criteria.

    Returns:
        (system_prompt, user_prompt)
    """
    subject_guidance = get_subject_guidance(task.subject)

    system_prompt = f"""You are an expert {task.subject} tutor working with a high school student.

## Your Tutoring Style
{persona['description']}

## Tutoring Principles
{TUTORING_PRINCIPLES}
{subject_guidance}

Respond naturally as this tutor would, applying these principles thoughtfully."""

    user_prompt = f"""## Original Question
{task.prompt}

## Initial Explanation Provided
{task.initial_explanation}

## Student's Follow-up Question
{task.follow_up}

---

Respond to the student's follow-up question. Think about what they understand, what they're missing, and how to guide them effectively without just giving away the answer."""

    return system_prompt, user_prompt


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_samples(csv_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load training samples from CSV."""
    samples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            samples.append({
                'idx': idx,
                'sample_id': f"sample_{idx:04d}",
                'subject': row.get('SUBJECT', 'unknown'),
                'prompt': row.get('PROMPT', ''),
                'initial_explanation': row.get('UC1_INITIAL_EXPLANATION', ''),
                'follow_up': row.get('FOLLOW_UP_PROMPT', ''),
            })

            if max_samples and len(samples) >= max_samples:
                break

    return samples


def create_generation_tasks(
    samples: List[Dict],
    personas: List[Dict],
    samples_per_persona: int = DEFAULT_SAMPLES_PER_PERSONA,
) -> List[GenerationTask]:
    """Create generation tasks for all sample-persona-variation combinations."""
    tasks = []

    for sample in samples:
        for persona_idx, persona in enumerate(personas):
            for variation_idx in range(samples_per_persona):
                task = GenerationTask(
                    sample_idx=sample['idx'],
                    sample_id=sample['sample_id'],
                    persona_idx=persona_idx,
                    persona_name=persona['name'],
                    variation_idx=variation_idx,
                    subject=sample['subject'],
                    prompt=sample['prompt'],
                    initial_explanation=sample['initial_explanation'],
                    follow_up=sample['follow_up'],
                )
                tasks.append(task)

    return tasks


# ============================================================================
# ASYNC GENERATION
# ============================================================================

async def generate_single_response(
    client: AsyncTogether,
    task: GenerationTask,
    persona: dict,
    semaphore: asyncio.Semaphore,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Optional[GeneratedResponse]:
    """Generate a single response with concurrency control."""

    async with semaphore:
        try:
            system_prompt, user_prompt = build_generation_prompt(task, persona)

            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )

            response_text = response.choices[0].message.content

            return GeneratedResponse(
                sample_id=task.sample_id,
                persona_name=task.persona_name,
                variation_idx=task.variation_idx,
                subject=task.subject,
                prompt=task.prompt,
                initial_explanation=task.initial_explanation,
                follow_up=task.follow_up,
                response=response_text,
                model=MODEL,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            print(f"\nError generating {task.sample_id}/{task.persona_name}/v{task.variation_idx}: {e}")
            return None


async def generate_batch(
    client: AsyncTogether,
    tasks: List[GenerationTask],
    personas: List[Dict],
    desc: str = "Generating",
    temperature: float = DEFAULT_TEMPERATURE,
) -> List[GeneratedResponse]:
    """Generate a batch of responses concurrently."""

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async_tasks = []
    for task in tasks:
        persona = personas[task.persona_idx]
        async_tasks.append(
            generate_single_response(client, task, persona, semaphore, temperature)
        )

    results = await tqdm_asyncio.gather(*async_tasks, desc=desc)

    # Filter out None results (errors)
    return [r for r in results if r is not None]


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_responses(responses: List[GeneratedResponse], output_path: Path, mode: str = 'a'):
    """Save responses to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding='utf-8') as f:
        for resp in responses:
            record = {
                'sample_id': resp.sample_id,
                'persona_name': resp.persona_name,
                'variation_idx': resp.variation_idx,
                'subject': resp.subject,
                'prompt': resp.prompt,
                'initial_explanation': resp.initial_explanation,
                'follow_up': resp.follow_up,
                'response': resp.response,
                'model': resp.model,
                'timestamp': resp.timestamp,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_checkpoint(output_path: Path) -> set:
    """Load completed task IDs from existing output file."""
    completed = set()

    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Include variation_idx in task ID for proper resume
                    variation_idx = record.get('variation_idx', 0)
                    task_id = f"{record['sample_id']}_{record['persona_name']}_v{variation_idx}"
                    completed.add(task_id)
                except:
                    continue

    return completed


# ============================================================================
# MAIN PIPELINE
# ============================================================================

async def run_generation(
    csv_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    samples_per_persona: int = DEFAULT_SAMPLES_PER_PERSONA,
):
    """Run the full generation pipeline."""

    total_per_sample = len(TUTOR_PERSONAS) * samples_per_persona

    print(f"\n{'='*60}")
    print("SYNTHETIC RESPONSE GENERATION")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {temperature}")
    print(f"Samples per persona: {samples_per_persona}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Personas: {len(TUTOR_PERSONAS)}")
    print(f"Generations per sample: {total_per_sample} ({len(TUTOR_PERSONAS)} personas x {samples_per_persona} variations)")
    print(f"Max concurrent: {MAX_CONCURRENT_REQUESTS}")
    print(f"{'='*60}\n")

    # Load samples
    print("Loading training samples...")
    samples = load_train_samples(csv_path, max_samples)
    print(f"Loaded {len(samples)} samples")

    # Create tasks
    tasks = create_generation_tasks(samples, TUTOR_PERSONAS, samples_per_persona)
    print(f"Created {len(tasks)} generation tasks ({len(samples)} samples x {len(TUTOR_PERSONAS)} personas x {samples_per_persona} variations)")

    # Handle resume
    if resume:
        completed = load_checkpoint(output_path)
        tasks = [t for t in tasks if f"{t.sample_id}_{t.persona_name}_v{t.variation_idx}" not in completed]
        print(f"Resuming: {len(completed)} already completed, {len(tasks)} remaining")

    if not tasks:
        print("No tasks to process!")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would generate {len(tasks)} responses")
        print("\nSample tasks:")
        for task in tasks[:5]:
            print(f"  - {task.sample_id} / {task.persona_name} / v{task.variation_idx} / {task.subject}")
        return

    # Initialize client
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_AI_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY or TOGETHER_AI_API_KEY environment variable required")

    client = AsyncTogether(api_key=api_key)

    # Generate in batches with checkpointing
    total_generated = 0

    for batch_start in range(0, len(tasks), CHECKPOINT_INTERVAL):
        batch_tasks = tasks[batch_start:batch_start + CHECKPOINT_INTERVAL]
        batch_num = batch_start // CHECKPOINT_INTERVAL + 1
        total_batches = (len(tasks) + CHECKPOINT_INTERVAL - 1) // CHECKPOINT_INTERVAL

        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch_tasks)} tasks) ---")

        responses = await generate_batch(
            client, batch_tasks, TUTOR_PERSONAS,
            desc=f"Batch {batch_num}/{total_batches}",
            temperature=temperature,
        )

        # Save batch
        mode = 'a' if (batch_start > 0 or resume) else 'w'
        save_responses(responses, output_path, mode)

        total_generated += len(responses)
        print(f"Saved {len(responses)} responses (total: {total_generated})")

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total generated: {total_generated}")
    print(f"Expected: {len(tasks)}")
    print(f"Success rate: {100 * total_generated / len(tasks):.1f}%")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tutoring responses")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: generated_responses/responses_{timestamp}.jsonl)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature for generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--samples-per-persona", type=int, default=DEFAULT_SAMPLES_PER_PERSONA,
                        help=f"Number of variations to generate per persona (default: {DEFAULT_SAMPLES_PER_PERSONA})")

    args = parser.parse_args()

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"responses_{timestamp}.jsonl"

    asyncio.run(run_generation(
        csv_path=TRAIN_CSV,
        output_path=output_path,
        max_samples=args.max_samples,
        resume=args.resume,
        dry_run=args.dry_run,
        temperature=args.temperature,
        samples_per_persona=args.samples_per_persona,
    ))


if __name__ == "__main__":
    main()
