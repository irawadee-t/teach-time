"""Chained LLM pipeline for multi-stage tutoring."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any

from ..models import UseCase
from .base import BasePipeline, PipelineResult, save_pipeline_debug_log


@dataclass
class StageTemplate:
    """Template for a single stage in the chain."""
    name: str
    prompt_template: str
    system_prompt: str = "You are an AI tutor analyzing student work."

    def format(self, **kwargs) -> str:
        return self.prompt_template.format(**kwargs)


@dataclass
class ChainedPipelineResult(PipelineResult):
    """Result from running a chained pipeline."""
    stage_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_response": self.final_response,
            "stage_outputs": self.stage_outputs,
            "num_stages": len(self.stage_outputs),
        }


class ChainedPipeline(BasePipeline):
    """Multi-stage LLM pipeline that chains reasoning steps."""

    _output_key = "_chain_outputs"

    def __init__(
        self,
        base_provider: Callable[[str, List[Dict]], str],
        stages: List[StageTemplate],
        verbose: bool = False,
    ):
        super().__init__(base_provider, verbose)
        self.stages = stages

    async def run_async(
        self,
        system_prompt: str,
        messages: List[Dict],
        use_case: Optional[UseCase] = None,
    ) -> ChainedPipelineResult:
        context = self.extract_context(messages)
        stage_outputs = {}

        for i, stage in enumerate(self.stages, 1):
            stage_name = f"stage_{i}"
            self.debug_print(f"Running Stage {i}: {stage.name}")

            prompt_vars = {
                "question": context.get("question", ""),
                "student_work": context.get("student_work", ""),
                "use_case": use_case.value if use_case else "tutoring",
                **{f"stage_{j}_output": stage_outputs.get(f"stage_{j}", "") for j in range(1, i)},
            }

            stage_prompt = stage.format(**prompt_vars)
            messages_for_stage = [{"role": "user", "content": stage_prompt}]
            output = await self.call_provider(stage.system_prompt, messages_for_stage)
            stage_outputs[stage_name] = output

            if self.verbose:
                print(f"Output: {output[:200]}...")

        final_stage = f"stage_{len(self.stages)}"
        return ChainedPipelineResult(
            final_response=stage_outputs[final_stage],
            stage_outputs=stage_outputs,
        )

    def create_model_fn(self, log_chain_outputs: bool = False) -> Callable:
        """Create model function with chain-specific output method."""
        model_fn = super().create_model_fn(log_outputs=log_chain_outputs)
        model_fn.get_chain_outputs = model_fn.get_outputs
        return model_fn


# Stage Templates for Tutoring Pipeline
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


def create_tutoring_chain(
    base_provider: Callable,
    include_safety_check: bool = False,
    verbose: bool = False,
) -> ChainedPipeline:
    """Create a simplified 3-stage tutoring chain (Perception -> Diagnosis -> Response)."""
    stages = [
        TUTORING_STAGES["perception"],
        TUTORING_STAGES["diagnosis"],
        TUTORING_STAGES["response"],
    ]
    if include_safety_check:
        stages.append(TUTORING_STAGES["safety_check"])
    return ChainedPipeline(base_provider=base_provider, stages=stages, verbose=verbose)


def create_custom_chain(
    base_provider: Callable,
    stage_names: List[str],
    verbose: bool = False,
) -> ChainedPipeline:
    """Create a custom chain with specific stages from TUTORING_STAGES."""
    stages = [TUTORING_STAGES[name] for name in stage_names]
    return ChainedPipeline(base_provider=base_provider, stages=stages, verbose=verbose)


async def compare_single_vs_chain(sample, single_provider: Callable, chain: ChainedPipeline) -> Dict[str, Any]:
    """Compare single-call vs chain performance on a sample."""
    from ..prompts import get_system_prompt

    system_prompt = get_system_prompt(sample.use_case, sample.is_multimodal)
    single_response = await single_provider(system_prompt, sample.messages)
    chain_result = await chain.run_async(system_prompt, sample.messages, use_case=sample.use_case)

    return {
        "sample_id": sample.sample_id,
        "use_case": sample.use_case.value,
        "single_response": single_response,
        "chain_response": chain_result.final_response,
        "chain_stages": chain_result.stage_outputs,
    }


def save_chain_debug_log(chain_outputs: Dict, samples: List, results: List, output_path: str) -> None:
    """Save detailed chain debug log with all intermediate outputs."""
    save_pipeline_debug_log(chain_outputs, samples, results, output_path, data_key="chain_data")
