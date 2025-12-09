"""
Chain-of-Thought generation for synthetic tutoring responses.

Generation phases:
0. Metacognitive Strategy: Determine subject-appropriate teaching approach
1. Knowledge Retrieval: Surface relevant frameworks, equations, formulas
1.5. Error Enumeration: Explicitly list ALL errors in student's work
2. Pedagogical Planning: Decide what to include vs. withhold
3. Response Generation: Write the final tutoring response

Method Branching (for Calculus, Physics):
- Analyzes student's solution method vs alternatives
- Constrains some personas to fix student's method (not switch approaches)
- Helps pass rubrics that demand specific solution methods

This approach aims to improve rubric pass rates by:
- Tailoring metacognitive strategy to subject-specific needs
- Ensuring domain knowledge is surfaced (helps "must mention X" rubrics)
- Error enumeration ensures ALL mistakes are identified (helps "must identify X" rubrics)
- Forcing explicit reasoning about revelation (helps "must NOT reveal Y" rubrics)
- Method branching ensures coverage of student's approach (helps method-specific rubrics)

Usage:
    python -m evals.synthetic.generate_samples_cot --max-samples 5 --dry-run
    python -m evals.synthetic.generate_samples_cot --max-samples 5
    python -m evals.synthetic.generate_samples_cot --max-samples 5 --temperature 1.0
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
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAX_CONCURRENT_REQUESTS = 500  # Maximum concurrency, retries handle rate limits

# Fallback API key if primary has insufficient balance
FALLBACK_API_KEY = "tgp_v1_OQB_4Mxxn6WflRB18-nLeBKPYj4fsVk_DJXaYS8dmY0"
MAX_TOKENS_PHASE0 = 500   # Metacognitive strategy
MAX_TOKENS_PHASE1 = 800   # Knowledge retrieval
MAX_TOKENS_PHASE1_5 = 600 # Error enumeration
MAX_TOKENS_PHASE2 = 600   # Pedagogical planning
MAX_TOKENS_PHASE3 = 2500  # Final response - needs to be long for complex math with all intermediate steps
DEFAULT_TEMPERATURE = 0.9
DEFAULT_SAMPLES_PER_PERSONA = 3
CHECKPOINT_INTERVAL = 500  # Checkpoint every N completions (streaming, not batching)

# Retry configuration
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 60.0  # seconds
PER_CALL_TIMEOUT = 60.0  # seconds - cancel any API call taking longer than this

# Rate limiting (Together AI has 1800 RPM = 30 RPS limit on Qwen)
RATE_LIMIT_RPS = 30  # requests per second
_rate_limiter: Optional[asyncio.Semaphore] = None
_rate_limiter_task: Optional[asyncio.Task] = None

# Paths
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR.parent
TRAIN_CSV = EVALS_DIR / "train.csv"
OUTPUT_DIR = SCRIPT_DIR / "generated_responses_cot"


# ============================================================================
# TUTOR PERSONAS (8 personas like v2)
# ============================================================================

TUTOR_PERSONAS = [
    {
        "name": "empathetic_validator",
        "style": "Patient and validating. Acknowledge feelings, use phrases like 'I understand why that's confusing.' Gently correct while maintaining confidence. IMPORTANT: Work within the student's chosen method - fix their approach rather than switching to a different one."
    },
    {
        "name": "socratic_questioner",
        "style": "Socratic and inquiry-based. Guide through questions rather than direct answers. Help students discover insights themselves."
    },
    {
        "name": "direct_clarifier",
        "style": "Clear and direct. Efficiently identify the exact misconception and provide targeted correction. Name specific errors explicitly. IMPORTANT: Stay with the student's approach and show them exactly where each error occurred and how to fix it."
    },
    {
        "name": "analogy_builder",
        "style": "Excel at relating abstract concepts to concrete, everyday examples. Use analogies and real-world scenarios to build intuition."
    },
    {
        "name": "step_by_step_guide",
        "style": "Methodical and structured. Break into numbered steps, show intermediate work, check understanding at each stage. IMPORTANT: Use the student's method and show step-by-step how to correct their work, computing each intermediate value explicitly."
    },
    {
        "name": "conceptual_connector",
        "style": "Emphasize how concepts relate to the bigger picture. Draw connections to previously learned material. Name relevant theorems or formulas."
    },
    {
        "name": "error_pattern_expert",
        "style": "Specialize in identifying common error patterns. Explicitly name the type of mistake and explain why it's a frequent pitfall. IMPORTANT: Diagnose ALL errors in the student's work - every sign error, every missing case, every algebraic mistake - and show the correct values."
    },
    {
        "name": "alternative_pathfinder",
        "style": "Present multiple ways to approach problems. Offer 'another way to think about this...' and show different solution strategies."
    },
]


# ============================================================================
# SUBJECT-SPECIFIC METACOGNITIVE STRATEGIES
# ============================================================================

SUBJECT_STRATEGIES = {
    "Biology": """
**Biology Metacognitive Strategy:**

Biology requires precise terminology because molecular mechanisms have specific names that students must learn.

**Example of good tutoring (glycolysis confusion):**
Student: "So the cell just breaks down glucose to make energy?"
Tutor: "You're on the right track! Let's be more precise about what 'breaking down' means. In glycolysis, glucose (a 6-carbon molecule) is split into two 3-carbon pyruvate molecules. This happens in the cytoplasm and produces 2 ATP directly. But notice I said 'pyruvate' not 'energy' - can you tell me what happens to pyruvate next?"

**Key principles:**
- Always use the correct molecular names (don't say "energy molecule" - say "ATP")
- When a process has multiple mechanisms, distinguish them clearly
- Include WHERE in the cell something happens
- Include specific enzyme names when relevant
- If a student conflates two different processes, explicitly name both and explain the difference
- Guide students to discover connections rather than listing facts
""",
    "Chemistry": """
**Chemistry Metacognitive Strategy:**

Chemistry tutoring balances precision with guided discovery. Students should derive relationships, not just memorize formulas.

**Example of good tutoring (acid-base equilibrium):**
Student: "I know Ka = [H+][A-]/[HA], but I don't understand why a higher Ka means stronger acid."
Tutor: "Great question! Look at your Ka expression - it's a ratio. If Ka is large, what does that tell you about the numerator compared to the denominator? Think about what's in the numerator... those are the products of dissociation. So if Ka is large, there must be more ___ than ___. What does that mean about how much the acid dissociates?"

**Key principles:**
- Name laws and principles explicitly (Le Chatelier's Principle, Hess's Law, etc.)
- For quantitative problems: guide students through the logic, don't just give the formula
- Don't reveal final numerical answers - let students calculate
- When students are close to a discovery, ask guiding questions instead of telling
- Use proper notation (equilibrium arrows, concentration brackets, oxidation states)
- Connect molecular behavior to macroscopic observations
""",
    "Physics": """
**Physics Metacognitive Strategy:**

Physics requires connecting mathematical formalism to physical intuition. Students often know formulas but misapply them.

**Example of good tutoring (projectile motion):**
Student: "I used v = v₀ + at to find the velocity, but my answer is negative. Did I mess up?"
Tutor: "Not necessarily! Let's think about what negative means here. You chose a coordinate system - which direction did you define as positive? If you chose 'up' as positive, then what sign should the acceleration due to gravity have? And if the ball is moving downward at some point, what sign should its velocity have?"

**Key principles:**
- Always clarify coordinate systems and sign conventions
- Distinguish vector and scalar quantities explicitly
- When discussing forces, specify: magnitude, direction, and point of application
- Reference physical laws by name (Newton's Second Law, Conservation of Momentum)
- Connect equations to physical meaning ("F=ma tells us that more mass means...")
- For distributed quantities (pressure, density), explain the concept of "spread over an area/volume"
""",
    "Calculus": """
**Calculus Metacognitive Strategy:**

Calculus tutoring requires identifying ALL errors, not just the conceptual one. Students often make multiple mistakes.

**Example of good tutoring (integration error):**
Student: "I got ∫(x² - 3x)dx = x³ - 3x², but the answer key says (x³/3) - (3x²/2) + C"
Tutor: "I see two issues here. First, let's check your antiderivatives: when you integrate x², you need to add 1 to the exponent AND divide by the new exponent. So x² becomes x³/3, not x³. Second, you forgot the constant of integration C. Let's verify: if we differentiate x³/3, we get... what?"

**Key principles:**
- Identify EVERY error, not just the main one (sign errors, algebra mistakes, missing regions)
- When students set up integrals, verify: correct bounds, correct integrand, correct variable
- For area/volume problems: check if ALL regions are accounted for
- If multiple solution methods exist, acknowledge them but work with the student's approach

**CRITICAL - Be explicit about these common issues:**
- If a formula is written incorrectly (e.g., sides inverted, wrong function), explicitly point it out: "You wrote y = √x, but x = y² actually gives y = ±√x - you're missing the negative branch"
- If the student makes a sign error (plus where it should be minus, or vice versa), call it out specifically: "You wrote √x - x - 2, but it should be √x - (x - 2) = √x - x + 2"
- When showing calculations with intermediate steps (like simplifying fractions), show ALL the work: "4/3 + 19/6 = 8/6 + 19/6 = 27/6 = 9/2"
- If a region or case is missing from the student's analysis, state exactly what was missed: "You missed the area from x = 0 to x = 1"
- When computing integrals, show the numeric result: "∫₀¹ 2√x dx = [4x^(3/2)/3]₀¹ = 4/3"
""",
    "Computer Science": """
**Computer Science Metacognitive Strategy:**

CS tutoring emphasizes logical reasoning and discovering patterns through concrete examples.

**Example of good tutoring (recursion):**
Student: "My recursive function returns the wrong answer. I have: if n==0 return 0, else return n + func(n-1)"
Tutor: "Let's trace through this together. If you call func(3), what happens? It returns 3 + func(2). And func(2) returns 2 + func(1). Keep going... what does func(1) return? And func(0)? Now add them all up. What do you get? Is that what you expected for the sum 1+2+3?"

**Key principles:**
- Trace through code with concrete examples, step by step
- Help students discover bugs themselves rather than pointing them out directly
- For number representation: work through actual bit patterns
- Explain the range of representable values and why overflow happens
- Use precise terminology (stack frame, heap allocation, pointer dereference)
- When students are close to the answer, ask "what if..." questions to guide them
""",
    "Statistics": """
**Statistics Metacognitive Strategy:**

Statistics requires distinguishing when to use which test and checking assumptions carefully.

**Example of good tutoring (hypothesis testing):**
Student: "I want to compare the means of two groups. Should I use a z-test or t-test?"
Tutor: "Good question! The choice depends on what you know. Ask yourself: do you know the population standard deviation, or only the sample standard deviation? If you only have sample data (which is almost always the case in practice), you'd use a t-test. But there's another consideration: how large are your samples? Why might sample size matter for choosing between these tests?"

**Key principles:**
- Always check assumptions before recommending a test
- Distinguish between population parameters and sample statistics
- Show formulas AND explain what each term means
- Connect statistical conclusions to real-world interpretation
- When students choose the wrong test, explain WHY it's wrong, not just which is right
- Emphasize the logic of hypothesis testing, not just the procedure
""",
}

DEFAULT_STRATEGY = """
**General Metacognitive Strategy:**
- Use precise, domain-appropriate terminology
- Name relevant principles, laws, theorems, or frameworks explicitly
- Identify common misconceptions and address them directly
- Show work and explain reasoning
- Connect concepts to the broader framework of the subject
"""


@dataclass
class GenerationTask:
    sample_idx: int
    sample_id: str
    persona_idx: int
    persona_name: str
    variation_idx: int
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str
    # Method branching (optional)
    method_constraint: Optional[str] = None  # e.g., "student_method", "alternative_method"
    method_description: Optional[str] = None  # e.g., "x-integration", "y-integration"


@dataclass
class CoTResponse:
    sample_id: str
    persona_name: str
    variation_idx: int
    subject: str
    prompt: str
    initial_explanation: str
    follow_up: str
    # CoT phases
    phase0_strategy: str
    phase1_knowledge: str
    phase1_5_errors: str  # Error enumeration phase
    phase2_plan: str
    phase3_response: str
    # Final response for grading
    response: str
    model: str
    timestamp: str
    # Method info (optional)
    method_constraint: Optional[str] = None
    method_description: Optional[str] = None


# ============================================================================
# PHASE 0: METACOGNITIVE STRATEGY
# ============================================================================

def build_phase0_prompt(task: GenerationTask) -> tuple[str, str]:
    """Phase 0: Determine subject-appropriate metacognitive strategy."""

    base_strategy = SUBJECT_STRATEGIES.get(task.subject, DEFAULT_STRATEGY)

    system = f"""You are an expert {task.subject} educator preparing to help a student.

Your task is to determine the best metacognitive approach for THIS specific question and student confusion.

{base_strategy}"""

    user = f"""## Original Question
{task.prompt}

## Initial Explanation Given
{task.initial_explanation}

## Student's Follow-up Question
{task.follow_up}

---

Based on this specific situation, determine:

1. **Question Type**: What kind of question is this? (conceptual, procedural, debugging an error, etc.)

2. **Student's Level**: What does the follow-up reveal about the student's understanding level?

3. **Key Challenge**: What is the main pedagogical challenge here?

4. **Strategy Selection**: Which approach will be most effective?
   - Should we guide through questions or explain directly?
   - Should we show worked examples or let them discover?
   - What terminology MUST be included for this subject?
   - What common pitfalls should we address?

5. **Subject-Specific Requirements**: For {task.subject}, what specific concepts, formulas, or terms should definitely be mentioned?

Be specific to THIS question - don't give generic advice."""

    return system, user


# ============================================================================
# METHOD ANALYSIS (for branching on solution approaches)
# ============================================================================

# Subjects that benefit from method branching
METHOD_BRANCH_SUBJECTS = {"Calculus", "Physics"}

# Maximum number of alternative methods to consider (keeps branching manageable)
MAX_ALTERNATIVE_METHODS = 3

METHOD_ANALYSIS_PROMPT = """Analyze the student's work to identify their solution method and possible alternatives.

## Student's Work
{follow_up}

---

Respond in this EXACT format (keep the labels exactly as shown):

STUDENT_METHOD: [One short phrase describing the student's approach, e.g., "x-integration", "substitution method", "force decomposition"]

ALTERNATIVE_METHODS: [Up to 3 other valid approaches (comma-separated), or "none" if only one approach works]

SHOULD_BRANCH: [yes/no - "yes" if the student's method can be corrected to get the right answer, "no" if their method fundamentally cannot work]

STUDENT_METHOD_DESCRIPTION: [One sentence explaining how to fix the student's approach while staying within their chosen method]

Be specific about the mathematical/scientific method, not just "wrong" vs "right"."""


def parse_method_analysis(response: str) -> dict:
    """Parse the structured method analysis response."""
    result = {
        "student_method": None,
        "alternative_methods": [],
        "should_branch": False,
        "student_method_description": None,
    }

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("STUDENT_METHOD:"):
            result["student_method"] = line.replace("STUDENT_METHOD:", "").strip()
        elif line.startswith("ALTERNATIVE_METHODS:"):
            alts = line.replace("ALTERNATIVE_METHODS:", "").strip()
            if alts.lower() != "none":
                all_alts = [a.strip() for a in alts.split(",") if a.strip()]
                # Limit to MAX_ALTERNATIVE_METHODS
                result["alternative_methods"] = all_alts[:MAX_ALTERNATIVE_METHODS]
        elif line.startswith("SHOULD_BRANCH:"):
            result["should_branch"] = "yes" in line.lower()
        elif line.startswith("STUDENT_METHOD_DESCRIPTION:"):
            result["student_method_description"] = line.replace("STUDENT_METHOD_DESCRIPTION:", "").strip()

    return result


def build_method_analysis_prompt(task: GenerationTask) -> tuple[str, str]:
    """Build prompt for method analysis phase."""
    system = f"""You are an expert {task.subject} educator analyzing a student's problem-solving approach.

Your job is to identify:
1. What method/approach the student is using
2. Whether alternative valid methods exist
3. Whether the student's method CAN work (even if they made errors)

For example, in Calculus area problems:
- A student using "x-integration" (integrating with respect to x) vs "y-integration" (integrating with respect to y)
- Both methods can work, but require different setups
- If a student chose x-integration, a good tutor might fix their x-integration rather than switching to y

In Physics:
- A student using "energy conservation" vs "force/kinematics equations"
- Both can work for many problems"""

    user = METHOD_ANALYSIS_PROMPT.format(follow_up=task.follow_up)
    return system, user


# ============================================================================
# PHASE 1: KNOWLEDGE RETRIEVAL
# ============================================================================

def build_phase1_prompt(task: GenerationTask, phase0_strategy: str) -> tuple[str, str]:
    """Phase 1: Surface relevant domain knowledge."""

    system = f"""You are an expert {task.subject} teacher preparing to help a student.

You have already determined a metacognitive strategy. Now retrieve ALL relevant knowledge.

## Your Strategy
{phase0_strategy}"""

    user = f"""## Original Question
{task.prompt}

## Initial Explanation Given
{task.initial_explanation}

## Student's Follow-up Question
{task.follow_up}

---

List the relevant knowledge for addressing this student's confusion:

1. **Key Concepts**: What fundamental concepts are involved? Name them precisely.

2. **Formulas/Equations/Laws**: What mathematical formulas, equations, or laws apply? Write them out explicitly - not just names.

3. **Terminology**: What specific terms should be used? (Use proper {task.subject} terminology)

4. **Common Misconceptions**: What errors or misconceptions might this student have?

5. **Correct Answer/Approach**: What is the correct way to think about or solve this?

6. **Prerequisites**: What should the student already know?

Be specific and thorough. Include actual formulas and precise terminology."""

    return system, user


# ============================================================================
# PHASE 1.5: ERROR ENUMERATION
# ============================================================================

def build_phase1_5_prompt(task: GenerationTask, phase0_strategy: str, phase1_knowledge: str) -> tuple[str, str]:
    """Phase 1.5: Enumerate ALL errors in the student's work."""

    # Add method constraint context if present
    method_context = ""
    if task.method_constraint == "student_method" and task.method_description:
        method_context = f"""
Note: You will be working within the student's method ({task.method_description}).
Find ALL errors in their approach that prevent them from getting the correct answer."""

    system = f"""You are an expert {task.subject} educator performing a thorough error analysis.

Your task is to find EVERY single error in the student's work - no matter how small.
Missing even ONE error is a critical failure.{method_context}

## Your Strategy
{phase0_strategy}

## Domain Knowledge
{phase1_knowledge}"""

    user = f"""## Original Question
{task.prompt}

## Initial Explanation Given
{task.initial_explanation}

## Student's Work to Analyze
{task.follow_up}

---

Perform a COMPLETE error analysis. List EVERY error, mistake, or missing element:

**ERROR #1**: [Describe the first error]
- What they wrote: [exact quote or formula]
- What it should be: [correct version]
- Why it's wrong: [explanation]
- Impact: [how this affects their answer]

**ERROR #2**: [Describe the second error]
- What they wrote: [exact quote or formula]
- What it should be: [correct version]
- Why it's wrong: [explanation]
- Impact: [how this affects their answer]

[Continue for ALL errors...]

**MISSING ELEMENTS** (things they forgot or didn't account for):
- [List any missing regions, cases, terms, steps, etc.]

**CORRECT FINAL ANSWER**: [State the correct answer, simplified to lowest terms for fractions]

---

IMPORTANT:
- List errors in the ORDER they appear in the student's work
- Be specific: quote exactly what they wrote
- Show exact values (e.g., "should be 4/3, not just 'a different integral'")
- For fractions: always simplify to lowest terms (e.g., 27/6 = 9/2)
- Include sign errors, algebraic mistakes, missing regions/cases, incorrect bounds - EVERYTHING
- If there are multiple errors, you MUST list ALL of them, not just the "main" one"""

    return system, user


# ============================================================================
# PHASE 2: PEDAGOGICAL PLANNING
# ============================================================================

def build_phase2_prompt(task: GenerationTask, phase0_strategy: str, phase1_knowledge: str, phase1_5_errors: str) -> tuple[str, str]:
    """Phase 2: Plan what to include vs. withhold."""

    # Add method constraint to system prompt if present
    method_section = ""
    if task.method_constraint == "student_method" and task.method_description:
        method_section = f"""

## CRITICAL: Method Constraint
You MUST work within the student's chosen method: **{task.method_description}**
Do NOT switch to a different approach. Fix their errors while staying in their framework.
Show how to do it correctly USING THEIR METHOD, even if another method might be "easier"."""
    elif task.method_constraint == "alternative_method" and task.method_description:
        method_section = f"""

## Method Approach
You may suggest an alternative approach: **{task.method_description}**
Explain why this method might be cleaner or more natural for this problem."""

    system = f"""You are an expert {task.subject} tutor planning your response strategy.

You have determined a metacognitive approach, retrieved relevant knowledge, and enumerated ALL errors in the student's work. Now plan what to INCLUDE and WITHHOLD.

## Your Strategy
{phase0_strategy}{method_section}"""

    user = f"""## Student's Question
{task.follow_up}

## Retrieved Knowledge
{phase1_knowledge}

## Error Analysis (ALL errors found in student's work)
{phase1_5_errors}

---

Plan your tutoring response:

1. **Student's Current Understanding**: What does the student seem to understand correctly?

2. **ALL Errors to Address** (from error analysis above):
   - You MUST address EVERY error listed in the error analysis
   - List each error and how you'll explain it

3. **MUST INCLUDE in Response** (be specific):
   - What terminology to use (exact terms)
   - What formulas/concepts to explain
   - What errors to point out (ALL of them!)
   - What steps to show
   - What intermediate values to compute explicitly

4. **WITHHOLD from Response** (be specific):
   - Final answers they should discover
   - Steps they're close to figuring out
   - Things that would rob them of "aha" moments

5. **Teaching Strategy**: How will you guide them?
   - Questions to ask?
   - Examples to use?
   - Step-by-step walkthrough?

Be explicit about your reasoning for including vs. withholding each element.

CRITICAL: You must address ALL errors from the error analysis. Missing any error is unacceptable."""

    return system, user


# ============================================================================
# PHASE 3: RESPONSE GENERATION
# ============================================================================

def build_phase3_prompt(task: GenerationTask, phase0_strategy: str, phase1_knowledge: str, phase1_5_errors: str, phase2_plan: str, persona: dict) -> tuple[str, str]:
    """Phase 3: Generate the final tutoring response."""

    # Add method constraint section if present
    method_section = ""
    if task.method_constraint == "student_method" and task.method_description:
        method_section = f"""

## CRITICAL: Method Constraint
You MUST use the student's method: **{task.method_description}**
Do NOT switch to a different approach. Show them how to fix THEIR approach.
The evaluator will check that you stayed within the student's chosen method."""
    elif task.method_constraint == "alternative_method" and task.method_description:
        method_section = f"""

## Method Approach
You are showing an alternative approach: **{task.method_description}**"""

    system = f"""You are an expert {task.subject} tutor.

## Your Style
{persona['style']}

You have analyzed all errors and planned your response. Now write it naturally, following your plan.

IMPORTANT: Use the terminology and include the concepts you decided on in your plan. Follow your include/withhold decisions.{method_section}

## Evaluation Criteria
Your response will be evaluated by a master tutor who will check:
- Did you identify ALL errors the student made (not just some)?
- Did you show calculations with intermediate steps and explicit values?
- Did you explain WHY each error is wrong and what the correct approach is?
- Did you use the student's method where possible, rather than switching approaches?
- Did you simplify all fractions to lowest terms?
- Did you avoid giving away answers the student should discover themselves?

A perfect score requires addressing EVERY error in the student's work.

## Numerical Answer Format
- ALWAYS simplify fractions to their lowest terms (e.g., 27/6 = 9/2, not left as 27/6)
- Show intermediate calculation steps (e.g., "4/3 + 19/6 = 8/6 + 19/6 = 27/6 = 9/2")
- When computing integrals, state the numeric result explicitly"""

    method_reminder = ""
    if task.method_constraint == "student_method" and task.method_description:
        method_reminder = f"\n- STAY WITHIN the student's method ({task.method_description}) - do NOT switch approaches"

    user = f"""## Original Question
{task.prompt}

## Initial Explanation
{task.initial_explanation}

## Student's Follow-up
{task.follow_up}

---

## Your Teaching Strategy
{phase0_strategy}

## Your Knowledge (for reference)
{phase1_knowledge}

## Error Analysis (ALL errors you MUST address)
{phase1_5_errors}

## Your Plan (FOLLOW THIS)
{phase2_plan}

---

Now write your tutoring response.

IMPORTANT REMINDERS:
- Address ALL errors from your error analysis - missing any is unacceptable
- Include what you decided to INCLUDE (terminology, formulas, concepts)
- Withhold what you decided to WITHHOLD
- Use your planned teaching strategy
- Be natural and conversational, not robotic
- Use precise {task.subject} terminology
- Simplify all fractions to lowest terms (27/6 → 9/2)
- Show intermediate steps when computing values
- Remember: An evaluator will check if you addressed ALL the student's errors{method_reminder}"""

    return system, user


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_samples(csv_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
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
# ASYNC GENERATION (4-phase) WITH RETRIES
# ============================================================================

import random

# Global fallback client (initialized on first balance error)
_fallback_client: Optional[AsyncTogether] = None
_using_fallback = False

async def _token_bucket_refiller():
    """Background task that adds tokens to the rate limiter at exactly RATE_LIMIT_RPS per second."""
    global _rate_limiter
    interval = 1.0 / RATE_LIMIT_RPS  # Time between each token (~33ms for 30 RPS)

    while True:
        await asyncio.sleep(interval)
        # Add a token if below max
        if _rate_limiter._value < RATE_LIMIT_RPS:
            _rate_limiter.release()


async def rate_limit_wait():
    """Acquire a rate limit token. Blocks until one is available."""
    global _rate_limiter, _rate_limiter_task

    if _rate_limiter is None:
        # Initialize with full bucket (30 tokens)
        _rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPS)
        # Start background task to refill tokens
        _rate_limiter_task = asyncio.create_task(_token_bucket_refiller())

    # Acquire a token (blocks if bucket is empty)
    await _rate_limiter.acquire()


async def call_with_retry(
    client: AsyncTogether,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    task_id: str = "",
) -> Optional[str]:
    """Make an API call with exponential backoff retry on rate limits and timeouts.

    Falls back to secondary API key if primary has insufficient balance.
    Respects rate limit of RATE_LIMIT_QPS requests per second.
    """
    global _fallback_client, _using_fallback

    current_client = _fallback_client if _using_fallback else client

    for attempt in range(MAX_RETRIES):
        try:
            # Rate limit before making request
            await rate_limit_wait()

            # Wrap API call with timeout to prevent hanging
            resp = await asyncio.wait_for(
                current_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=PER_CALL_TIMEOUT
            )
            return resp.choices[0].message.content
        except asyncio.TimeoutError:
            # Timeout - treat as retriable
            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), RETRY_MAX_DELAY)
                if task_id:
                    print(f"\n[Timeout] {task_id} attempt {attempt+1}/{MAX_RETRIES}, retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            else:
                if task_id:
                    print(f"\n[Timeout] {task_id} failed after {MAX_RETRIES} attempts")
                return None  # Return None instead of raising - don't crash the batch
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate" in error_str
            is_overloaded = "503" in error_str or "502" in error_str or "500" in error_str or "overloaded" in error_str or "not ready" in error_str or "cloudflare" in error_str
            is_balance_error = "balance" in error_str or "insufficient" in error_str or "credit" in error_str
            is_api_key_error = "api key" in error_str or "unauthorized" in error_str or "401" in error_str

            # Check for balance/API key errors - switch to fallback
            if (is_balance_error or is_api_key_error) and not _using_fallback and FALLBACK_API_KEY:
                print(f"\n[API Key/Balance Error] Switching to fallback API key...")
                _fallback_client = AsyncTogether(api_key=FALLBACK_API_KEY)
                _using_fallback = True
                current_client = _fallback_client
                continue  # Retry with fallback

            if (is_rate_limit or is_overloaded) and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                delay = min(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)
                continue
            else:
                if task_id:
                    print(f"\nError {task_id}: {e}")
                return None  # Return None instead of raising - don't crash the batch
    return None


async def generate_cot_response(
    client: AsyncTogether,
    task: GenerationTask,
    persona: dict,
    semaphore: asyncio.Semaphore,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Optional[CoTResponse]:
    """Generate a response using 5-phase CoT with retry logic."""

    task_id = f"{task.sample_id}/{task.persona_name}/v{task.variation_idx}"

    async with semaphore:
        try:
            # Phase 0: Metacognitive Strategy
            sys0, user0 = build_phase0_prompt(task)
            phase0_strategy = await call_with_retry(
                client,
                [{"role": "system", "content": sys0}, {"role": "user", "content": user0}],
                MAX_TOKENS_PHASE0,
                temperature,
                f"{task_id}/phase0",
            )
            if not phase0_strategy:
                return None

            # Phase 1: Knowledge Retrieval
            sys1, user1 = build_phase1_prompt(task, phase0_strategy)
            phase1_knowledge = await call_with_retry(
                client,
                [{"role": "system", "content": sys1}, {"role": "user", "content": user1}],
                MAX_TOKENS_PHASE1,
                temperature,
                f"{task_id}/phase1",
            )
            if not phase1_knowledge:
                return None

            # Phase 1.5: Error Enumeration
            sys1_5, user1_5 = build_phase1_5_prompt(task, phase0_strategy, phase1_knowledge)
            phase1_5_errors = await call_with_retry(
                client,
                [{"role": "system", "content": sys1_5}, {"role": "user", "content": user1_5}],
                MAX_TOKENS_PHASE1_5,
                temperature,
                f"{task_id}/phase1_5",
            )
            if not phase1_5_errors:
                return None

            # Phase 2: Pedagogical Planning
            sys2, user2 = build_phase2_prompt(task, phase0_strategy, phase1_knowledge, phase1_5_errors)
            phase2_plan = await call_with_retry(
                client,
                [{"role": "system", "content": sys2}, {"role": "user", "content": user2}],
                MAX_TOKENS_PHASE2,
                temperature,
                f"{task_id}/phase2",
            )
            if not phase2_plan:
                return None

            # Phase 3: Response Generation
            sys3, user3 = build_phase3_prompt(task, phase0_strategy, phase1_knowledge, phase1_5_errors, phase2_plan, persona)
            phase3_response = await call_with_retry(
                client,
                [{"role": "system", "content": sys3}, {"role": "user", "content": user3}],
                MAX_TOKENS_PHASE3,
                temperature,
                f"{task_id}/phase3",
            )
            if not phase3_response:
                return None

            return CoTResponse(
                sample_id=task.sample_id,
                persona_name=task.persona_name,
                variation_idx=task.variation_idx,
                subject=task.subject,
                prompt=task.prompt,
                initial_explanation=task.initial_explanation,
                follow_up=task.follow_up,
                phase0_strategy=phase0_strategy,
                phase1_knowledge=phase1_knowledge,
                phase1_5_errors=phase1_5_errors,
                phase2_plan=phase2_plan,
                phase3_response=phase3_response,
                response=phase3_response,  # Final response for grading
                model=MODEL,
                timestamp=datetime.now().isoformat(),
                method_constraint=task.method_constraint,
                method_description=task.method_description,
            )

        except Exception as e:
            print(f"\nError generating {task_id}: {e}")
            return None


async def generate_all_streaming(
    client: AsyncTogether,
    tasks: List[GenerationTask],
    personas: List[Dict],
    output_path: Path,
    resume: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
) -> int:
    """Generate all responses with streaming checkpoints.

    Instead of batching, this runs ALL tasks concurrently (limited by semaphore)
    and saves results as they complete, checkpointing every CHECKPOINT_INTERVAL.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create all async tasks
    async_tasks = []
    for task in tasks:
        persona = personas[task.persona_idx]
        async_tasks.append(
            generate_cot_response(client, task, persona, semaphore, temperature)
        )

    # Process with streaming progress and periodic saves
    completed = []
    total_saved = 0
    mode = 'a' if resume else 'w'

    # Use tqdm for progress tracking
    from tqdm.asyncio import tqdm

    for coro in tqdm.as_completed(async_tasks, total=len(async_tasks), desc="Generating"):
        result = await coro
        if result is not None:
            completed.append(result)

            # Checkpoint every CHECKPOINT_INTERVAL completions
            if len(completed) >= CHECKPOINT_INTERVAL:
                save_responses(completed, output_path, mode)
                total_saved += len(completed)
                completed = []
                mode = 'a'  # Always append after first save

    # Save any remaining
    if completed:
        save_responses(completed, output_path, mode)
        total_saved += len(completed)

    return total_saved


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_responses(responses: List[CoTResponse], output_path: Path, mode: str = 'a'):
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
                'phase0_strategy': resp.phase0_strategy,
                'phase1_knowledge': resp.phase1_knowledge,
                'phase1_5_errors': resp.phase1_5_errors,
                'phase2_plan': resp.phase2_plan,
                'phase3_response': resp.phase3_response,
                'response': resp.response,
                'model': resp.model,
                'timestamp': resp.timestamp,
                'method_constraint': resp.method_constraint,
                'method_description': resp.method_description,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_checkpoint(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    variation_idx = record.get('variation_idx', 0)
                    method = record.get('method_constraint', 'none') or 'none'
                    task_id = f"{record['sample_id']}_{record['persona_name']}_v{variation_idx}_{method}"
                    completed.add(task_id)
                except:
                    continue
    return completed


def get_task_id(task: GenerationTask) -> str:
    """Get unique task ID including method constraint."""
    method = task.method_constraint or 'none'
    return f"{task.sample_id}_{task.persona_name}_v{task.variation_idx}_{method}"


# ============================================================================
# METHOD ANALYSIS FOR BRANCHING
# ============================================================================

MAX_TOKENS_METHOD_ANALYSIS = 300

async def analyze_sample_methods(
    client: AsyncTogether,
    samples: List[Dict],
    semaphore: asyncio.Semaphore,
    temperature: float = 0.3,  # Lower temp for more consistent analysis
) -> Dict[str, dict]:
    """Analyze solution methods for samples that could benefit from branching.

    Returns a dict mapping sample_id -> method analysis result.
    """
    results = {}

    # Only analyze samples in branching subjects
    samples_to_analyze = [s for s in samples if s['subject'] in METHOD_BRANCH_SUBJECTS]

    if not samples_to_analyze:
        return results

    print(f"\nAnalyzing methods for {len(samples_to_analyze)} samples in {METHOD_BRANCH_SUBJECTS}...")

    async def analyze_one(sample: Dict) -> tuple[str, dict]:
        task = GenerationTask(
            sample_idx=sample['idx'],
            sample_id=sample['sample_id'],
            persona_idx=0,
            persona_name="",
            variation_idx=0,
            subject=sample['subject'],
            prompt=sample['prompt'],
            initial_explanation=sample['initial_explanation'],
            follow_up=sample['follow_up'],
        )

        async with semaphore:
            sys_prompt, user_prompt = build_method_analysis_prompt(task)
            response = await call_with_retry(
                client,
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                MAX_TOKENS_METHOD_ANALYSIS,
                temperature,
                f"{sample['sample_id']}/method_analysis",
            )

            if response:
                parsed = parse_method_analysis(response)
                return sample['sample_id'], parsed
            return sample['sample_id'], {"student_method": None, "should_branch": False}

    tasks = [analyze_one(s) for s in samples_to_analyze]
    analysis_results = await tqdm_asyncio.gather(*tasks, desc="Method analysis")

    for sample_id, analysis in analysis_results:
        results[sample_id] = analysis

    # Print summary
    branching_count = sum(1 for a in results.values() if a.get('should_branch'))
    print(f"Method analysis complete: {branching_count}/{len(results)} samples will branch on methods")

    return results


def create_branched_tasks(
    samples: List[Dict],
    personas: List[Dict],
    method_analyses: Dict[str, dict],
    samples_per_persona: int = DEFAULT_SAMPLES_PER_PERSONA,
) -> List[GenerationTask]:
    """Create generation tasks with method branching.

    For samples with branching enabled:
    - Half the personas get "student_method" constraint
    - Half get "alternative_method" constraint (if alternatives exist)

    For samples without branching:
    - All personas get no constraint (original behavior)
    """
    tasks = []

    # Personas that should stick with student's method
    student_method_personas = {
        "empathetic_validator", "direct_clarifier",
        "step_by_step_guide", "error_pattern_expert"
    }

    for sample in samples:
        analysis = method_analyses.get(sample['sample_id'], {})
        should_branch = analysis.get('should_branch', False)
        student_method = analysis.get('student_method')
        alternatives = analysis.get('alternative_methods', [])
        student_method_desc = analysis.get('student_method_description')

        for persona_idx, persona in enumerate(personas):
            for variation_idx in range(samples_per_persona):
                # Determine method constraint
                method_constraint = None
                method_description = None

                if should_branch and student_method:
                    if persona['name'] in student_method_personas:
                        method_constraint = "student_method"
                        method_description = student_method_desc or student_method
                    elif alternatives and persona['name'] == "alternative_pathfinder":
                        method_constraint = "alternative_method"
                        method_description = alternatives[0] if alternatives else None

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
                    method_constraint=method_constraint,
                    method_description=method_description,
                )
                tasks.append(task)

    return tasks


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
    total_per_sample = len(TUTOR_PERSONAS) * samples_per_persona

    print(f"\n{'='*60}")
    print("CHAIN-OF-THOUGHT RESPONSE GENERATION (5-Phase)")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {temperature}")
    print(f"Samples per persona: {samples_per_persona}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Personas: {len(TUTOR_PERSONAS)}")
    print(f"Generations per sample: {total_per_sample} ({len(TUTOR_PERSONAS)} personas x {samples_per_persona} variations)")
    print(f"Phases: 5 (Strategy → Knowledge → Errors → Plan → Response)")
    print(f"API calls per generation: 5")
    print(f"Max concurrent: {MAX_CONCURRENT_REQUESTS}")
    print(f"Method branching subjects: {METHOD_BRANCH_SUBJECTS}")
    print(f"{'='*60}\n")

    # Load samples
    print("Loading training samples...")
    samples = load_train_samples(csv_path, max_samples)
    print(f"Loaded {len(samples)} samples")

    # Initialize client (needed for method analysis)
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_AI_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY required")

    client = AsyncTogether(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Run method analysis for branching subjects
    method_analyses = {}
    if not dry_run:
        method_analyses = await analyze_sample_methods(client, samples, semaphore)

    # Create tasks with method branching
    tasks = create_branched_tasks(samples, TUTOR_PERSONAS, method_analyses, samples_per_persona)

    # Count tasks with method constraints
    constrained_tasks = sum(1 for t in tasks if t.method_constraint)
    print(f"Created {len(tasks)} generation tasks ({constrained_tasks} with method constraints)")
    print(f"Total API calls: {len(tasks) * 5}")

    # Handle resume
    if resume:
        completed = load_checkpoint(output_path)
        tasks = [t for t in tasks if get_task_id(t) not in completed]
        print(f"Resuming: {len(completed)} completed, {len(tasks)} remaining")

    if not tasks:
        print("No tasks to process!")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would generate {len(tasks)} responses ({len(tasks)*5} API calls)")
        print("\nSample tasks:")
        for task in tasks[:10]:
            method_info = f" [{task.method_constraint}: {task.method_description}]" if task.method_constraint else ""
            print(f"  - {task.sample_id} / {task.persona_name} / v{task.variation_idx} / {task.subject}{method_info}")

        # Show sample prompts
        print("\n--- Sample Phase 0 (Strategy) Prompt ---")
        sys0, user0 = build_phase0_prompt(tasks[0])
        print(f"System: {sys0[:300]}...")
        print(f"User: {user0[:400]}...")
        return

    # Generate ALL tasks concurrently with streaming checkpoints
    print(f"\nStarting generation of {len(tasks)} tasks...")
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS}, Checkpoint every: {CHECKPOINT_INTERVAL}")

    total_generated = await generate_all_streaming(
        client, tasks, TUTOR_PERSONAS,
        output_path, resume, temperature,
    )

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total generated: {total_generated}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate CoT tutoring responses (4-phase)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--samples-per-persona", type=int, default=DEFAULT_SAMPLES_PER_PERSONA,
                        help=f"Variations per persona (default: {DEFAULT_SAMPLES_PER_PERSONA})")

    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"cot_responses_{timestamp}.jsonl"

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
