"""Best-of-N sampling pipeline for model evaluation."""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any

from ..models import UseCase
from .base import BasePipeline, PipelineResult, save_pipeline_debug_log


@dataclass
class BestOfNResult(PipelineResult):
    """Result from running best-of-n selection."""
    all_responses: List[str] = field(default_factory=list)
    selection_reasoning: str = ""
    selected_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_response": self.final_response,
            "all_responses": self.all_responses,
            "selection_reasoning": self.selection_reasoning,
            "selected_index": self.selected_index,
            "num_candidates": len(self.all_responses),
        }


SELECTION_PROMPT = """You are evaluating multiple tutoring responses to select the best one.

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


class BestOfNPipeline(BasePipeline):
    """Generate N responses in parallel and select the best one."""

    _output_key = "_bon_outputs"

    def __init__(
        self,
        base_provider: Callable[[str, List[Dict]], str],
        n: int = 5,
        selection_provider: Optional[Callable] = None,
        verbose: bool = False,
    ):
        super().__init__(base_provider, verbose)
        self.n = n
        self.selection_provider = selection_provider or base_provider
        self.selector_is_async = asyncio.iscoroutinefunction(self.selection_provider)

    async def run_async(
        self,
        system_prompt: str,
        messages: List[Dict],
        use_case: Optional[UseCase] = None,
    ) -> BestOfNResult:
        self.debug_print(f"Best of {self.n}: Generating {self.n} candidate responses")

        # Generate N responses in parallel
        if self.is_async:
            tasks = [self.base_provider(system_prompt, messages) for _ in range(self.n)]
            responses = await asyncio.gather(*tasks)
        else:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, self.base_provider, system_prompt, messages)
                for _ in range(self.n)
            ]
            responses = await asyncio.gather(*tasks)

        if self.verbose:
            for i, resp in enumerate(responses, 1):
                print(f"\nCandidate {i} preview: {resp[:150]}...")

        # Select the best response
        context = self._format_context(messages, use_case)
        selected_index, reasoning = await self._select_best(responses, context)

        self.debug_print(f"Selected response {selected_index + 1}/{self.n}", reasoning)

        return BestOfNResult(
            final_response=responses[selected_index],
            all_responses=list(responses),
            selection_reasoning=reasoning,
            selected_index=selected_index,
        )

    async def _select_best(self, responses: List[str], context: str) -> tuple[int, str]:
        """Use selection_provider to pick the best response."""
        formatted_responses = "\n\n".join([f"Response {i+1}:\n{resp}" for i, resp in enumerate(responses)])
        prompt = SELECTION_PROMPT.format(context=context, n=self.n, responses=formatted_responses)

        messages = [{"role": "user", "content": prompt}]
        system = "You are an expert tutor evaluating tutoring responses."

        if self.selector_is_async:
            output = await self.selection_provider(system, messages)
        else:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, self.selection_provider, system, messages)

        return self._parse_selection(output)

    def _parse_selection(self, output: str) -> tuple[int, str]:
        """Parse selection output to extract index and reasoning."""
        selected_index = 0
        reasoning = "No reasoning provided"
        lines = output.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("SELECTED:"):
                match = re.search(r'\d+', line.split("SELECTED:")[1])
                if match:
                    selected_index = max(0, min(int(match.group()) - 1, self.n - 1))
            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:")[1].strip()
                for next_line in lines[i+1:]:
                    next_line = next_line.strip()
                    if not next_line.startswith("SELECTED:") and next_line:
                        reasoning += " " + next_line
                break

        return selected_index, reasoning

    def _format_context(self, messages: List[Dict], use_case: Optional[UseCase] = None) -> str:
        """Format context for selection prompt."""
        parts = [f"Use case: {use_case.value}"] if use_case else []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                if role == "user":
                    parts.append(f"Student: {content}")
                elif role == "assistant":
                    parts.append(f"Previous response: {content}")
        return "\n".join(parts)

    def create_model_fn(self, log_bon_outputs: bool = False) -> Callable:
        """Create model function with best-of-n-specific output method."""
        model_fn = super().create_model_fn(log_outputs=log_bon_outputs)
        model_fn.get_bon_outputs = model_fn.get_outputs
        return model_fn


def create_best_of_n_pipeline(
    base_provider: Callable,
    n: int = 5,
    selection_provider: Optional[Callable] = None,
    verbose: bool = False,
) -> BestOfNPipeline:
    """Create a best-of-n pipeline."""
    return BestOfNPipeline(
        base_provider=base_provider,
        n=n,
        selection_provider=selection_provider,
        verbose=verbose,
    )


def save_bon_debug_log(bon_outputs: Dict, samples: List, results: List, output_path: str) -> None:
    """Save detailed best-of-n debug log with all candidate responses."""
    save_pipeline_debug_log(bon_outputs, samples, results, output_path, data_key="bon_data")
