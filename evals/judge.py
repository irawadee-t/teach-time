"""
LLM judge for TutorBench evaluation.

Uses Claude Sonnet 4 (Anthropic) as the judge model, achieving:
- F1 score of 0.82 vs human majority vote
- Better agreement than median human expert (0.78 vs 0.75)
"""

import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from .models import Rubric, RubricRating
from .providers import _init_provider


class RubricJudgment(BaseModel):
    """Structured output for a single rubric judgment."""
    criterion: str = Field(description="The rubric criterion being evaluated")
    passed: bool = Field(description="Whether the criterion was met (true/false)")
    explanation: str = Field(description="Brief explanation of the judgment")


class BatchJudgment(BaseModel):
    """Structured output for batch rubric evaluation."""
    judgments: List[RubricJudgment] = Field(description="List of judgments for each rubric")


class TutorBenchJudge:
    """
    LLM judge for evaluating model responses against rubric criteria.

    Uses Claude Sonnet 4 as per TutorBench paper.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = False,
    ):
        """
        Args:
            api_key: Anthropic API key (reads from ANTHROPIC_API_KEY if not provided)
            model: Judge model name (default: Claude Sonnet 4)
            verbose: Whether to print debug information
        """
        self.model = model
        self.verbose = verbose

        # Initialize async Anthropic client using provider init system
        try:
            self.client = _init_provider(
                library_name="anthropic",
                client_class_path="anthropic.AsyncAnthropic",
                api_key_vars=["ANTHROPIC_API_KEY"],
            )
        except (ImportError, ValueError) as e:
            self.client = None
            if self.verbose:
                print(f"Warning: {e}")

    async def evaluate_rubric(
        self,
        model_response: str,
        rubric: Rubric,
        context: Optional[str] = None,
    ) -> RubricRating:
        """
        Evaluate a single rubric criterion asynchronously.

        Args:
            model_response: The model's tutoring response to evaluate
            rubric: The rubric criterion to check
            context: Optional context (question, student work, etc.)

        Returns:
            RubricRating with pass/fail decision and explanation
        """
        judge_prompt = self._construct_judge_prompt(model_response, rubric, context)
        passed, explanation = await self._get_judgment(judge_prompt)

        return RubricRating(
            rubric=rubric,
            passed=passed,
            explanation=explanation,
        )

    async def evaluate_rubrics(
        self,
        model_response: str,
        rubrics: List[Rubric],
        context: Optional[str] = None,
        batch_size: int = 5,
    ) -> List[RubricRating]:
        """
        Evaluate rubrics concurrently with batching.

        Evaluates rubrics in batches for better throughput while
        respecting rate limits.

        Args:
            model_response: The model's tutoring response
            rubrics: List of rubric criteria
            context: Optional context
            batch_size: Number of rubrics to evaluate concurrently

        Returns:
            List of rubric ratings (same order as input rubrics)
        """
        if not rubrics:
            return []

        # Create tasks for all rubrics
        tasks = [
            self.evaluate_rubric(model_response, rubric, context)
            for rubric in rubrics
        ]

        # Execute in batches
        ratings = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            # Handle exceptions
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Fallback to failed rating
                    rubric_idx = i + idx
                    ratings.append(RubricRating(
                        rubric=rubrics[rubric_idx],
                        passed=False,
                        explanation=f"[Error: {str(result)}]",
                    ))
                else:
                    ratings.append(result)

        return ratings

    def _construct_judge_prompt(
        self,
        model_response: str,
        rubric: Rubric,
        context: Optional[str] = None,
    ) -> str:
        """
        Construct the prompt for the LLM judge.

        Format follows TutorBench methodology:
        - Present the context (if any)
        - Present the model response
        - Present the rubric criterion
        - Ask for binary pass/fail judgment
        """
        prompt_parts = []

        if context:
            prompt_parts.append(f"### Context\n{context}\n")

        prompt_parts.append(f"### Model Response to Evaluate\n{model_response}\n")

        prompt_parts.append(f"### Evaluation Criterion\n{rubric.criterion}\n")

        prompt_parts.append(
            "### Task\n"
            "Evaluate whether the model response satisfies the criterion above.\n\n"
            "Respond with EXACTLY this format:\n"
            "JUDGMENT: [PASS or FAIL]\n"
            "EXPLANATION: [Brief explanation of why it passes or fails]\n"
        )

        return "\n".join(prompt_parts)

    async def _get_judgment(self, judge_prompt: str) -> tuple[bool, str]:
        """
        Get pass/fail judgment from LLM judge asynchronously.

        Args:
            judge_prompt: The constructed prompt

        Returns:
            Tuple of (passed, explanation)
        """
        if not self.client:
            # Mock response for testing
            if self.verbose:
                print("Warning: Using mock judgment (no API client)")
            return True, "[Mock judgment - API not configured]"

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            content = response.content[0].text
            passed, explanation = self._parse_judgment(content)
            return passed, explanation

        except Exception as e:
            if self.verbose:
                print(f"Error in async judgment: {e}")
            return False, f"[Error: {str(e)}]"

    def _parse_judgment(self, response_text: str) -> tuple[bool, str]:
        """
        Parse the LLM judge's response to extract pass/fail and explanation.

        Expected format:
            JUDGMENT: PASS
            EXPLANATION: The response correctly identifies...

        Args:
            response_text: Raw response from judge

        Returns:
            Tuple of (passed, explanation)
        """
        lines = response_text.strip().split('\n')

        judgment = None
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.startswith("JUDGMENT:"):
                judgment_text = line.split("JUDGMENT:", 1)[1].strip().upper()
                judgment = "PASS" in judgment_text
            elif line.startswith("EXPLANATION:"):
                explanation = line.split("EXPLANATION:", 1)[1].strip()

        # If parsing fails, default to fail for safety
        if judgment is None:
            if self.verbose:
                print(f"Warning: Could not parse judgment from: {response_text[:100]}")
            judgment = False
            explanation = response_text[:200]  # First 200 chars as explanation

        return judgment, explanation
