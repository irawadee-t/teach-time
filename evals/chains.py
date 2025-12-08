"""
Chained LLM pipeline for multi-stage tutoring.

Instead of a single model call, breaks down tutoring into explicit cognitive stages:
1. Perception & Extraction - What did the student do?
2. Diagnosis - What's wrong and why?
3. Student Modeling - What do they know/not know?
4. Response Planning - What to say vs. withhold?
5. Generation - Write the actual response
6. Safety Check - Did we give away too much? (optional)

This addresses key tutoring skills that models often skip or conflate.
"""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

from .models import UseCase


@dataclass
class StageTemplate:
    """
    Template for a single stage in the chain.

    Prompt can include placeholders:
    - {question}: The original question/problem
    - {student_work}: Student's work or response
    - {use_case}: The tutoring use case
    - {stage_1_output}, {stage_2_output}, etc.: Previous stage outputs
    """
    name: str
    prompt_template: str
    system_prompt: str = "You are an AI tutor analyzing student work."

    def format(self, **kwargs) -> str:
        """Format the prompt template with provided variables."""
        return self.prompt_template.format(**kwargs)


@dataclass
class ChainedPipelineResult:
    """Result from running a chained pipeline."""
    final_response: str  # What gets evaluated
    stage_outputs: Dict[str, str] = field(default_factory=dict)  # For debugging

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "final_response": self.final_response,
            "stage_outputs": self.stage_outputs,
            "num_stages": len(self.stage_outputs),
        }


class ChainedPipeline:
    """
    Multi-stage LLM pipeline that chains reasoning steps.

    Each stage:
    1. Receives context + outputs from previous stages
    2. Calls the base model provider
    3. Passes output to next stage

    The final stage output is what gets evaluated by TutorBench.
    """

    def __init__(
        self,
        base_provider: Callable[[str, List[Dict]], str],  # Async model function
        stages: List[StageTemplate],
        verbose: bool = False,
    ):
        """
        Args:
            base_provider: Async model function from providers.py
            stages: List of stage templates to execute in order
            verbose: Print intermediate outputs
        """
        self.base_provider = base_provider
        self.stages = stages
        self.verbose = verbose

        # Check if provider is async
        self.is_async = asyncio.iscoroutinefunction(base_provider)

    async def run_async(
        self,
        system_prompt: str,  # Original system prompt (for context)
        messages: List[Dict],  # Original messages
        use_case: Optional[UseCase] = None,
    ) -> ChainedPipelineResult:
        """
        Run the chained pipeline asynchronously.

        Args:
            system_prompt: Original system prompt from TutorBench
            messages: Original messages (question + student work)
            use_case: Tutoring use case (for context)

        Returns:
            ChainedPipelineResult with final response and intermediate outputs
        """
        # Extract question and student work from messages
        context = self._extract_context(messages)

        # Store outputs from each stage
        stage_outputs = {}

        # Run each stage sequentially
        for i, stage in enumerate(self.stages, 1):
            stage_name = f"stage_{i}"

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running Stage {i}: {stage.name}")
                print(f"{'='*60}")

            # Format the prompt with available context
            prompt_vars = {
                "question": context.get("question", ""),
                "student_work": context.get("student_work", ""),
                "use_case": use_case.value if use_case else "tutoring",
                **{f"stage_{j}_output": stage_outputs.get(f"stage_{j}", "")
                   for j in range(1, i)},
            }

            stage_prompt = stage.format(**prompt_vars)

            # Call the base provider
            messages_for_stage = [{"role": "user", "content": stage_prompt}]

            if self.is_async:
                output = await self.base_provider(stage.system_prompt, messages_for_stage)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    self.base_provider,
                    stage.system_prompt,
                    messages_for_stage,
                )

            # Store output
            stage_outputs[stage_name] = output

            if self.verbose:
                print(f"Output: {output[:200]}...")

        # The final stage output is what gets evaluated
        final_stage = f"stage_{len(self.stages)}"
        final_response = stage_outputs[final_stage]

        return ChainedPipelineResult(
            final_response=final_response,
            stage_outputs=stage_outputs,
        )

    def _extract_context(self, messages: List[Dict]) -> Dict[str, str]:
        """
        Extract question and student work from messages.

        Args:
            messages: Chat messages from TutorBench sample

        Returns:
            Dict with 'question' and 'student_work' keys
        """
        context = {"question": "", "student_work": ""}

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                if role == "user":
                    # First user message is usually the question
                    if not context["question"]:
                        context["question"] = content
                    else:
                        # Subsequent user messages are student work
                        context["student_work"] += "\n" + content

        return context

    def create_model_fn(self, log_chain_outputs: bool = False) -> Callable:
        """
        Create a model function compatible with TutorBench runner.

        Args:
            log_chain_outputs: If True, attach chain metadata to response
                              for debugging (saved in evaluation results)

        Returns:
            Async function with signature: (system_prompt, messages) -> response
        """
        async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
            """Wrapped model function that runs the chain."""
            result = await self.run_async(system_prompt, messages)

            # Attach chain outputs as metadata if logging enabled
            if log_chain_outputs:
                # Store in a way that can be accessed by the runner
                if not hasattr(model_fn, '_chain_outputs'):
                    model_fn._chain_outputs = {}

                # Use a simple counter as key (will be matched to sample later)
                sample_key = len(model_fn._chain_outputs)
                model_fn._chain_outputs[sample_key] = result.to_dict()

            return result.final_response

        # Add method to retrieve chain outputs
        def get_chain_outputs():
            return getattr(model_fn, '_chain_outputs', {})

        model_fn.get_chain_outputs = get_chain_outputs

        return model_fn


# ============================================================================
# Stage Templates for Tutoring Pipeline
# ============================================================================

TUTORING_STAGES = {
    "perception": StageTemplate(
        name="Perception & Extraction",
        system_prompt="You are an AI tutor analyzing student work with precision.",
        prompt_template="""You are analyzing a student's work on a problem.

Problem: {question}

Student's work: {student_work}

Extract the following information:
1. What steps did the student take?
2. What intermediate values did they calculate?
3. What final answer (if any) did they give?
4. What formulas or methods did they attempt to use?

Be precise. Quote their work directly where relevant. Do not make judgments yet.""",
    ),

    "diagnosis": StageTemplate(
        name="Diagnosis",
        system_prompt="You are an AI tutor identifying errors and misconceptions. You must verify all factual claims.",
        prompt_template="""Given this analysis of a student's work:

{stage_1_output}

Identify:
1. What specific errors did they make? (arithmetic, conceptual, procedural)
2. What is the root cause or misconception behind each error?
3. What did they do correctly?

CRITICAL: Verify the factual accuracy of your diagnosis. Double-check:
- Mathematical formulas and their correct forms
- Scientific facts (e.g., which acids are stronger, which reactions are faster)
- Physical laws and their applications
- Statistical methods and when they apply

Be specific. "They made a math error" is not enough—say exactly what the error was and verify it's actually an error.""",
    ),

    "student_modeling": StageTemplate(
        name="Student Modeling",
        system_prompt="You are an AI tutor inferring student knowledge and emotional state.",
        prompt_template="""Based on this student's work and errors:

Student's work:
{stage_1_output}

Diagnosis:
{stage_2_output}

Infer:
1. What concepts does this student seem to understand?
2. What prerequisite knowledge might they be missing?
3. What is their apparent skill level (beginner/intermediate/advanced)?
4. Are there emotional cues (frustration, confusion, confidence)?

This will help calibrate the response appropriately.""",
    ),

    "response_planning": StageTemplate(
        name="Response Planning",
        system_prompt="You are an AI tutor planning an effective tutoring response.",
        prompt_template="""You are planning your tutoring response.

Use case: {use_case}

Student situation:
{stage_1_output}

{stage_2_output}

{stage_3_output}

Plan your response:
1. What is the ONE most important thing to address?
2. What should you explicitly teach? (Formulas, definitions, procedures are OKAY to teach directly)
3. What should you withhold? (Only withhold final numerical answers or complete solutions)
4. Should you ask a guiding question? If so, what?
5. Would a brief example help? What kind?

Balance: It's GOOD to teach formulas, concepts, and methods. Only preserve agency by withholding final answers.
For HINTS (active_learning): Do NOT reveal the final answer, but DO teach the methods needed.""",
    ),

    "response": StageTemplate(
        name="Response",
        system_prompt="You are an expert AI tutor providing clear, concise tutoring responses.",
        prompt_template="""Write a tutoring response to the student.

Problem: {question}
Student's work: {student_work}

Your diagnosis:
{stage_2_output}

CRITICAL REQUIREMENTS:
1. **Be concise** - Keep under 200 words unless problem requires detail
2. **Include required formulas** - Show any formulas mentioned in the problem
3. **Provide numerical answers** - Give specific values when asked
4. **Acknowledge student** - Start by recognizing what they did/asked
5. **Be step-by-step** - Break down solutions into clear numbered steps when appropriate
6. **No formatting fluff** - No headers (###), no labels (**Tutor:**), no signatures
7. **Natural tone** - Write like a helpful tutor, not a template

For {use_case}:
- ADAPTIVE (explanation): Explain concepts clearly, include formulas and examples
- ASSESSMENT (feedback): Identify errors specifically, explain what's correct
- ACTIVE_LEARNING (hints): Guide without revealing final answer, teach methods

Write your response:""",
    ),

    "safety_check": StageTemplate(
        name="Safety Check",
        system_prompt="You are reviewing a tutoring response for quality and appropriateness.",
        prompt_template="""Review this tutoring response:

{stage_3_output}

The task was: {use_case}

Check:
1. If this was a HINT request (active_learning), does the response reveal the final answer?
2. Is anything factually incorrect?
3. Is it too long or off-topic?

If there are problems, rewrite the response to fix them.
Otherwise, output the original response unchanged.

Final response:""",
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def create_tutoring_chain(
    base_provider: Callable,
    include_safety_check: bool = False,
    verbose: bool = False,
) -> ChainedPipeline:
    """
    Create a simplified 3-stage tutoring chain.

    Args:
        base_provider: Async model function from providers.py
        include_safety_check: Whether to include the final safety check stage (default: False)
        verbose: Print intermediate outputs

    Returns:
        ChainedPipeline configured for tutoring

    Example:
        >>> from evals.providers import get_async_provider
        >>> provider = get_async_provider("anthropic", "claude-sonnet-4-20250514")
        >>> chain = create_tutoring_chain(provider, verbose=True)
        >>> model_fn = chain.create_model_fn()
        >>> # Use model_fn in evaluate_model_async
    """
    # Simplified 3-stage chain: Perception -> Diagnosis -> Response
    stages = [
        TUTORING_STAGES["perception"],
        TUTORING_STAGES["diagnosis"],
        TUTORING_STAGES["response"],
    ]

    if include_safety_check:
        stages.append(TUTORING_STAGES["safety_check"])

    return ChainedPipeline(
        base_provider=base_provider,
        stages=stages,
        verbose=verbose,
    )


def create_custom_chain(
    base_provider: Callable,
    stage_names: List[str],
    verbose: bool = False,
) -> ChainedPipeline:
    """
    Create a custom chain with specific stages.

    Args:
        base_provider: Async model function from providers.py
        stage_names: List of stage names to include (from TUTORING_STAGES)
        verbose: Print intermediate outputs

    Returns:
        ChainedPipeline with selected stages

    Example:
        >>> # Just diagnosis + generation
        >>> chain = create_custom_chain(
        ...     provider,
        ...     ["diagnosis", "generation"],
        ... )
    """
    stages = [TUTORING_STAGES[name] for name in stage_names]
    return ChainedPipeline(
        base_provider=base_provider,
        stages=stages,
        verbose=verbose,
    )


# ============================================================================
# Comparison Utilities
# ============================================================================

async def compare_single_vs_chain(
    sample,
    single_provider: Callable,
    chain: ChainedPipeline,
) -> Dict[str, Any]:
    """
    Compare single-call vs chain performance on a sample.

    Args:
        sample: TutorBench sample
        single_provider: Single-call async provider
        chain: ChainedPipeline instance

    Returns:
        Dict with both responses and intermediate outputs
    """
    from .prompts import get_system_prompt

    # Get system prompt
    system_prompt = get_system_prompt(sample.use_case, sample.is_multimodal)

    # Single call
    single_response = await single_provider(system_prompt, sample.messages)

    # Chained call
    chain_result = await chain.run_async(
        system_prompt,
        sample.messages,
        use_case=sample.use_case,
    )

    return {
        "sample_id": sample.sample_id,
        "use_case": sample.use_case.value,
        "single_response": single_response,
        "chain_response": chain_result.final_response,
        "chain_stages": chain_result.stage_outputs,
    }


def save_chain_debug_log(
    chain_outputs: Dict,
    samples: List,
    results: List,
    output_path: str,
) -> None:
    """
    Save detailed chain debug log with all intermediate outputs.

    Args:
        chain_outputs: Dict from model_fn.get_chain_outputs()
        samples: List of Sample objects
        results: List of EvaluationResult objects
        output_path: Path to save debug log
    """
    import json
    from pathlib import Path

    debug_data = {
        "metadata": {
            "total_samples": len(samples),
            "total_chain_outputs": len(chain_outputs),
        },
        "samples": []
    }

    # Match chain outputs to samples and results
    for idx, (sample, result) in enumerate(zip(samples, results)):
        chain_output = chain_outputs.get(idx, {})

        sample_debug = {
            "sample_id": sample.sample_id,
            "use_case": sample.use_case.value,
            "subject": sample.subject,
            "score": result.weighted_score,
            "pass_rate": result.pass_rate,
            "num_rubrics": len(result.rubric_ratings),
            "chain_data": chain_output,
        }

        debug_data["samples"].append(sample_debug)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=2)

    print(f"📝 Chain debug log saved to: {output_path}")
