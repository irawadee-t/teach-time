"""
Evaluation runner for TutorBench.

Provides concurrent evaluation with two-level concurrency:
1. Process multiple samples concurrently
2. Evaluate multiple rubrics per sample concurrently

Performance: 5-7x faster than sequential evaluation.
"""

import asyncio
from typing import List, Optional, Callable
from tqdm.asyncio import tqdm as async_tqdm

from .models import Sample, EvaluationResult
from .judge import TutorBenchJudge
from .scoring import compute_weighted_score
from .prompts import get_system_prompt
from .providers import ConcurrencyConfig


async def evaluate_model_async(
    samples: List[Sample],
    model_fn: Callable,
    judge: Optional[TutorBenchJudge] = None,
    model_name: str = "unknown",
    config: Optional[ConcurrencyConfig] = None,
    verbose: bool = False,
) -> List[EvaluationResult]:
    """
    Async version of evaluate_model with concurrent processing.

    Two-level concurrency:
    1. Process multiple samples concurrently (controlled by max_concurrent)
    2. For each sample, evaluate rubrics concurrently (controlled by rubric_batch_size)

    Args:
        samples: List of TutorBench samples
        model_fn: Function (sync or async) that generates responses
        judge: TutorBench judge instance (creates default if None)
        model_name: Name of model being evaluated
        config: Concurrency configuration
        verbose: Show progress

    Returns:
        List of evaluation results

    Example:
        >>> config = ConcurrencyConfig(max_concurrent=20, rubric_batch_size=5)
        >>> results = await evaluate_model_async(samples, model_fn, config=config)
    """
    # Initialize config and judge
    config = config or ConcurrencyConfig()

    if judge is None:
        judge = TutorBenchJudge(verbose=verbose)

    # Detect if model_fn is async
    is_async_model = asyncio.iscoroutinefunction(model_fn)

    # Global semaphore for max concurrent API calls
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def process_sample(sample: Sample, pbar: Optional[async_tqdm] = None) -> EvaluationResult:
        """Process a single sample with concurrency control."""
        async with semaphore:
            try:
                # Generate model response
                system_prompt = get_system_prompt(sample.use_case, sample.is_multimodal)

                if is_async_model:
                    model_response = await model_fn(system_prompt, sample.messages)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    model_response = await loop.run_in_executor(
                        None, model_fn, system_prompt, sample.messages
                    )

                # Prepare context for judge
                context = _prepare_context(sample)

                # Evaluate rubrics concurrently
                rubric_ratings = await judge.evaluate_rubrics(
                    model_response=model_response,
                    rubrics=sample.rubrics,
                    context=context,
                    batch_size=config.rubric_batch_size,
                )

                # Compute score
                weighted_score = compute_weighted_score(rubric_ratings)

                # Update progress
                if pbar:
                    pbar.update(1)

                return EvaluationResult(
                    sample_id=sample.sample_id,
                    model_name=model_name,
                    model_response=model_response,
                    rubric_ratings=rubric_ratings,
                    weighted_score=weighted_score,
                )

            except Exception as e:
                # Handle errors gracefully
                if verbose:
                    print(f"Error processing sample {sample.sample_id}: {e}")

                if pbar:
                    pbar.update(1)

                # Return failed result
                return EvaluationResult(
                    sample_id=sample.sample_id,
                    model_name=model_name,
                    model_response=f"[Error during evaluation: {str(e)}]",
                    rubric_ratings=[],
                    weighted_score=0.0,
                )

    # Create progress bar if verbose
    pbar = None
    if verbose:
        pbar = async_tqdm(total=len(samples), desc=f"Evaluating {model_name}")

    # Create tasks for all samples
    tasks = [process_sample(sample, pbar) for sample in samples]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Close progress bar
    if pbar:
        pbar.close()

    # Process results and handle any exceptions
    final_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            if verbose:
                print(f"Unexpected error for sample {samples[idx].sample_id}: {result}")
            # Create failed result
            final_results.append(EvaluationResult(
                sample_id=samples[idx].sample_id,
                model_name=model_name,
                model_response="[Unexpected error]",
                rubric_ratings=[],
                weighted_score=0.0,
            ))
        else:
            final_results.append(result)

    return final_results


def evaluate_model_concurrent(
    samples: List[Sample],
    model_fn: Callable,
    judge: Optional[TutorBenchJudge] = None,
    model_name: str = "unknown",
    config: Optional[ConcurrencyConfig] = None,
    verbose: bool = False,
) -> List[EvaluationResult]:
    """
    Synchronous wrapper for evaluate_model_async.

    Allows calling async evaluation from sync code using asyncio.run().

    Args:
        samples: List of TutorBench samples
        model_fn: Function (sync or async) that generates responses
        judge: TutorBench judge instance (creates default if None)
        model_name: Name of model being evaluated
        config: Concurrency configuration
        verbose: Show progress

    Returns:
        List of evaluation results

    Example:
        >>> config = ConcurrencyConfig(max_concurrent=25)
        >>> results = evaluate_model_concurrent(samples, model_fn, config=config)
    """
    return asyncio.run(evaluate_model_async(
        samples=samples,
        model_fn=model_fn,
        judge=judge,
        model_name=model_name,
        config=config,
        verbose=verbose,
    ))


def evaluate_model(
    samples: List[Sample],
    model_fn: Callable,
    judge: Optional[TutorBenchJudge] = None,
    model_name: str = "unknown",
    verbose: bool = False,
    concurrency_config: Optional[ConcurrencyConfig] = None,
) -> List[EvaluationResult]:
    """
    Main evaluation function - wrapper around evaluate_model_concurrent.

    Args:
        samples: List of TutorBench samples
        model_fn: Function (sync or async) that generates responses
        judge: TutorBench judge instance (creates default if None)
        model_name: Name of model being evaluated
        verbose: Show progress
        concurrency_config: Concurrency configuration

    Returns:
        List of evaluation results
    """
    return evaluate_model_concurrent(
        samples=samples,
        model_fn=model_fn,
        judge=judge,
        model_name=model_name,
        config=concurrency_config,
        verbose=verbose,
    )


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
