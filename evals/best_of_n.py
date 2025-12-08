"""
Best-of-N sampling pipeline for model evaluation.

Instead of a single model call, generates N responses in parallel and selects
the best one using a judge model. This can improve response quality by:
1. Generating diverse candidate responses
2. Selecting the most pedagogically sound one
3. Filtering out poor responses before evaluation

The selected response is then evaluated normally by TutorBench.
"""

import asyncio
import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

from .models import UseCase


@dataclass
class BestOfNResult:
    """Result from running best-of-n selection."""
    final_response: str  # Selected best response (what gets evaluated)
    all_responses: List[str] = field(default_factory=list)  # All N candidates
    selection_reasoning: str = ""  # Why this response was selected
    selected_index: int = -1  # Which response was selected (0-indexed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "final_response": self.final_response,
            "all_responses": self.all_responses,
            "selection_reasoning": self.selection_reasoning,
            "selected_index": self.selected_index,
            "num_candidates": len(self.all_responses),
        }


# Selection prompt template
SELECTION_PROMPT_TEMPLATE = """You are evaluating multiple tutoring responses to select the best one.

Context:
{context}

Here are {n} different tutoring responses to the same student:

{responses}

Your task: Select the SINGLE BEST response for effective tutoring.

Consider:
1. Pedagogical quality - Does it guide without over-explaining?
2. Accuracy - Is the content factually correct?
3. Clarity - Is it well-organized and understandable?
4. Appropriateness - Does it match the student's level and the use case?
5. Engagement - Is it encouraging and natural?

Respond in this exact format:
SELECTED: <number 1-{n}>
REASONING: <brief explanation of why this response is best>

Your selection:"""


class BestOfNPipeline:
    """
    Generate N responses in parallel and select the best one.

    Workflow:
    1. Call base_provider N times in parallel using asyncio.gather()
    2. Format all responses for comparison
    3. Call selection_provider once to pick the best response
    4. Parse the selection and return the chosen response

    The selected response is what gets evaluated by TutorBench.
    """

    def __init__(
        self,
        base_provider: Callable[[str, List[Dict]], str],  # Async model function
        n: int = 5,
        selection_provider: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """
        Args:
            base_provider: Async model function from providers.py (generates candidates)
            n: Number of responses to generate
            selection_provider: Async model function for selection (defaults to base_provider)
            verbose: Print intermediate outputs
        """
        self.base_provider = base_provider
        self.n = n
        self.selection_provider = selection_provider or base_provider
        self.verbose = verbose

        # Check if providers are async
        self.is_async = asyncio.iscoroutinefunction(base_provider)
        self.selector_is_async = asyncio.iscoroutinefunction(self.selection_provider)

    async def run_async(
        self,
        system_prompt: str,
        messages: List[Dict],
        use_case: Optional[UseCase] = None,
    ) -> BestOfNResult:
        """
        Generate N responses and select the best one.

        Args:
            system_prompt: Original system prompt from TutorBench
            messages: Original messages (question + student work)
            use_case: Tutoring use case (for context)

        Returns:
            BestOfNResult with selected response and metadata
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Best of {self.n}: Generating {self.n} candidate responses")
            print(f"{'='*60}")

        # Step 1: Generate N responses in parallel
        if self.is_async:
            # Async provider - use gather
            tasks = [
                self.base_provider(system_prompt, messages)
                for _ in range(self.n)
            ]
            responses = await asyncio.gather(*tasks)
        else:
            # Sync provider - run in executor
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    self.base_provider,
                    system_prompt,
                    messages
                )
                for _ in range(self.n)
            ]
            responses = await asyncio.gather(*tasks)

        if self.verbose:
            for i, resp in enumerate(responses, 1):
                print(f"\nCandidate {i} preview: {resp[:150]}...")

        # Step 2: Format context for selection
        context = self._extract_context(messages, use_case)

        # Step 3: Select the best response
        selected_index, reasoning = await self._select_best_response(
            responses, context
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Selected response {selected_index + 1}/{self.n}")
            print(f"Reasoning: {reasoning}")
            print(f"{'='*60}")

        return BestOfNResult(
            final_response=responses[selected_index],
            all_responses=list(responses),
            selection_reasoning=reasoning,
            selected_index=selected_index,
        )

    async def _select_best_response(
        self,
        responses: List[str],
        context: str,
    ) -> tuple[int, str]:
        """
        Use selection_provider to pick the best response.

        Args:
            responses: List of N candidate responses
            context: Context string (question, student work, use case)

        Returns:
            Tuple of (selected_index, reasoning)
        """
        # Format responses for comparison
        formatted_responses = "\n\n".join([
            f"Response {i+1}:\n{resp}"
            for i, resp in enumerate(responses)
        ])

        # Create selection prompt
        selection_prompt = SELECTION_PROMPT_TEMPLATE.format(
            context=context,
            n=self.n,
            responses=formatted_responses,
        )

        # Call selection provider
        messages = [{"role": "user", "content": selection_prompt}]
        system_prompt = "You are an expert tutor evaluating tutoring responses."

        if self.selector_is_async:
            selection_output = await self.selection_provider(
                system_prompt, messages
            )
        else:
            loop = asyncio.get_event_loop()
            selection_output = await loop.run_in_executor(
                None,
                self.selection_provider,
                system_prompt,
                messages,
            )

        # Parse the selection
        selected_index, reasoning = self._parse_selection(selection_output)

        return selected_index, reasoning

    def _parse_selection(self, output: str) -> tuple[int, str]:
        """
        Parse selection output to extract index and reasoning.

        Expected format:
        SELECTED: 2
        REASONING: This response is clearer and more pedagogically sound...

        Args:
            output: Raw LLM output

        Returns:
            Tuple of (selected_index, reasoning)
        """
        selected_index = 0  # Default to first
        reasoning = "No reasoning provided"

        lines = output.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("SELECTED:"):
                try:
                    # Extract number and convert to 0-indexed
                    num_str = line.split("SELECTED:")[1].strip()
                    # Handle cases like "SELECTED: 2" or "SELECTED: Response 2"
                    match = re.search(r'\d+', num_str)
                    if match:
                        selected_num = int(match.group())
                        # Clamp to valid range
                        selected_index = max(0, min(selected_num - 1, self.n - 1))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                # Extract reasoning and include subsequent lines
                reasoning = line.split("REASONING:")[1].strip()
                # Append subsequent non-header lines
                for next_line in lines[i+1:]:
                    next_line = next_line.strip()
                    if not next_line.startswith("SELECTED:") and next_line:
                        reasoning += " " + next_line
                break

        return selected_index, reasoning

    def _extract_context(
        self,
        messages: List[Dict],
        use_case: Optional[UseCase] = None
    ) -> str:
        """
        Extract context for selection prompt.

        Args:
            messages: Chat messages from TutorBench sample
            use_case: Tutoring use case

        Returns:
            Formatted context string
        """
        context_parts = []

        if use_case:
            context_parts.append(f"Use case: {use_case.value}")

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                if role == "user":
                    context_parts.append(f"Student: {content}")
                elif role == "assistant":
                    context_parts.append(f"Previous response: {content}")

        return "\n".join(context_parts)

    def create_model_fn(self, log_bon_outputs: bool = False) -> Callable:
        """
        Create a model function compatible with TutorBench runner.

        Args:
            log_bon_outputs: If True, attach best-of-n metadata to response
                            for debugging (saved in evaluation results)

        Returns:
            Async function with signature: (system_prompt, messages) -> response
        """
        async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
            """Wrapped model function that runs best-of-n selection."""
            result = await self.run_async(system_prompt, messages)

            # Attach best-of-n outputs as metadata if logging enabled
            if log_bon_outputs:
                if not hasattr(model_fn, '_bon_outputs'):
                    model_fn._bon_outputs = {}

                # Use a simple counter as key (matched to sample later)
                sample_key = len(model_fn._bon_outputs)
                model_fn._bon_outputs[sample_key] = result.to_dict()

            return result.final_response

        # Add method to retrieve best-of-n outputs
        def get_bon_outputs():
            return getattr(model_fn, '_bon_outputs', {})

        model_fn.get_bon_outputs = get_bon_outputs

        return model_fn


# ============================================================================
# Helper Functions
# ============================================================================

def create_best_of_n_pipeline(
    base_provider: Callable,
    n: int = 5,
    selection_provider: Optional[Callable] = None,
    verbose: bool = False,
) -> BestOfNPipeline:
    """
    Create a best-of-n pipeline.

    Args:
        base_provider: Async model function from providers.py
        n: Number of candidate responses to generate
        selection_provider: Async model function for selection (defaults to base_provider)
        verbose: Print intermediate outputs

    Returns:
        BestOfNPipeline configured for best-of-n selection

    Example:
        >>> from evals.providers import get_async_provider
        >>> provider = get_async_provider("anthropic", "claude-sonnet-4-20250514")
        >>> pipeline = create_best_of_n_pipeline(provider, n=5, verbose=True)
        >>> model_fn = pipeline.create_model_fn()
        >>> # Use model_fn in evaluate_model_async
    """
    return BestOfNPipeline(
        base_provider=base_provider,
        n=n,
        selection_provider=selection_provider,
        verbose=verbose,
    )


def save_bon_debug_log(
    bon_outputs: Dict,
    samples: List,
    results: List,
    output_path: str,
) -> None:
    """
    Save detailed best-of-n debug log with all candidate responses.

    Args:
        bon_outputs: Dict from model_fn.get_bon_outputs()
        samples: List of Sample objects
        results: List of EvaluationResult objects
        output_path: Path to save debug log
    """
    import json
    from pathlib import Path

    debug_data = {
        "metadata": {
            "total_samples": len(samples),
            "total_bon_outputs": len(bon_outputs),
        },
        "samples": []
    }

    # Match bon outputs to samples and results
    for idx, (sample, result) in enumerate(zip(samples, results)):
        bon_output = bon_outputs.get(idx, {})

        sample_debug = {
            "sample_id": sample.sample_id,
            "use_case": sample.use_case.value,
            "subject": sample.subject,
            "score": result.weighted_score,
            "pass_rate": result.pass_rate,
            "num_rubrics": len(result.rubric_ratings),
            "bon_data": bon_output,
        }

        debug_data["samples"].append(sample_debug)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=2)

    print(f"📝 Best-of-N debug log saved to: {output_path}")
