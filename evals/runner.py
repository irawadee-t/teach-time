"""
Evaluation runner for TutorBench.

Orchestrates:
1. Generating model responses with appropriate system prompts
2. Evaluating responses against sample-specific rubrics
3. Computing scores and aggregating results
"""

from typing import List, Optional, Callable
from tqdm import tqdm

from .models import Sample, EvaluationResult, RubricRating
from .judge import TutorBenchJudge
from .scoring import compute_weighted_score
from .prompts import get_system_prompt


def evaluate_model(
    samples: List[Sample],
    model_fn: Callable[[str, List[dict]], str],
    judge: Optional[TutorBenchJudge] = None,
    model_name: str = "unknown",
    verbose: bool = False,
) -> List[EvaluationResult]:
    """
    Evaluate a model on TutorBench samples.

    Args:
        samples: List of TutorBench samples
        model_fn: Function that takes (system_prompt, messages) and returns response
        judge: TutorBench judge instance (creates default if None)
        model_name: Name of the model being evaluated
        verbose: Whether to show progress

    Returns:
        List of evaluation results
    """
    # Initialize judge if not provided
    if judge is None:
        judge = TutorBenchJudge(verbose=verbose)

    results = []
    iterator = tqdm(samples, desc=f"Evaluating {model_name}") if verbose else samples

    for sample in iterator:
        # Get system prompt for this use case
        system_prompt = get_system_prompt(sample.use_case, sample.is_multimodal)

        # Generate model response
        model_response = model_fn(system_prompt, sample.messages)

        # Prepare context for judge
        context = _prepare_context(sample)

        # Evaluate against rubrics
        rubric_ratings = judge.evaluate_rubrics(
            model_response=model_response,
            rubrics=sample.rubrics,
            context=context,
        )

        # Compute weighted score
        weighted_score = compute_weighted_score(rubric_ratings)

        # Create evaluation result
        result = EvaluationResult(
            sample_id=sample.sample_id,
            model_name=model_name,
            model_response=model_response,
            rubric_ratings=rubric_ratings,
            weighted_score=weighted_score,
        )
        results.append(result)

    return results


def _prepare_context(sample: Sample) -> str:
    """
    Prepare context string for the judge.

    Includes the question and student work/context from the conversation.

    Args:
        sample: TutorBench sample

    Returns:
        Formatted context string
    """
    context_parts = []

    # Add subject and use case
    context_parts.append(f"Subject: {sample.subject}")
    context_parts.append(f"Use Case: {sample.use_case.value}")
    context_parts.append("")

    # Add conversation messages
    for msg in sample.messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            context_parts.append(f"Student: {content}")
        elif role == "assistant":
            context_parts.append(f"Initial Response: {content}")

    return "\n".join(context_parts)


def generate_response(
    model_fn: Callable[[str, List[dict]], str],
    sample: Sample,
) -> str:
    """
    Generate a single model response for a sample.

    Args:
        model_fn: Function that takes (system_prompt, messages) and returns response
        sample: TutorBench sample

    Returns:
        Model response string
    """
    system_prompt = get_system_prompt(sample.use_case, sample.is_multimodal)
    return model_fn(system_prompt, sample.messages)
