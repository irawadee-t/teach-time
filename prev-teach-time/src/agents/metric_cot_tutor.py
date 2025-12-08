"""
Metric-aware Chain-of-Thought (Metric-CoT) tutor agent.

Uses CoT reasoning with explicit teaching metrics in the prompt,
but no discrete action space.
"""

from typing import Optional
from .base_agent import BaseTutorAgent
from ..env.teaching_env import Observation
from ..env.actions import PedagogicalAction, ActionType
from ..llm.prompt_utils import build_metric_cot_tutor_prompt, parse_cot_response


class MetricCoTTutorAgent(BaseTutorAgent):
    """
    Metric-aware CoT tutor.

    - Uses chain-of-thought reasoning
    - Teaching metrics explicitly included in prompt
    - No discrete pedagogical action space (still free-form)
    - Agent can reason about metrics when generating responses
    """

    def __init__(
        self,
        agent_id: str = "metric_cot_tutor",
        llm_client=None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        super().__init__(agent_id, llm_client, temperature, max_tokens)

    def act(self, observation: Observation) -> PedagogicalAction:
        """
        Generate tutor response using metric-aware CoT reasoning.

        Args:
            observation: Current observation from environment (includes metrics)

        Returns:
            PedagogicalAction (mapped to a generic action type)
        """
        # Build prompt with metrics
        prompt = build_metric_cot_tutor_prompt(
            topic=self.current_task.topic,
            learning_objectives=self.current_task.learning_objectives,
            conversation_history=self.dialogue_history,
            student_utterance=observation.student_utterance,
            metrics=observation.metrics,
        )

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.call(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse response
        thought, tutor_response = parse_cot_response(response)

        # Record thought for analysis
        self.thoughts.append({
            "step": observation.step_index,
            "thought": thought,
            "response": tutor_response,
            "metrics": observation.metrics.copy(),
        })

        # Record this turn
        self.record_turn("tutor", tutor_response)

        # Infer action type (same heuristic as baseline CoT)
        action_type = self._infer_action_type(tutor_response)

        return PedagogicalAction(
            action_type=action_type,
            content=tutor_response,
            metadata={"thought": thought, "metrics": observation.metrics}
        )

    def _infer_action_type(self, response: str) -> ActionType:
        """
        Infer action type from free-form response content.

        Same heuristic as baseline CoT.
        """
        response_lower = response.lower()

        # Check for questions
        if '?' in response:
            # Distinguish between open questions and checks
            check_indicators = [
                "do you understand",
                "does that make sense",
                "can you explain",
                "let me check",
            ]
            if any(indicator in response_lower for indicator in check_indicators):
                return ActionType.ASK_CHECK_UNDERSTANDING

            # Check for background probing
            background_indicators = [
                "have you seen",
                "do you know about",
                "are you familiar with",
                "what do you already know",
                "have you learned",
            ]
            if any(indicator in response_lower for indicator in background_indicators):
                return ActionType.ASK_BACKGROUND

            # Default to open question
            return ActionType.ASK_OPEN_QUESTION

        # Check for explanations
        explanation_indicators = [
            "let me explain",
            "here's how",
            "the way to",
            "step by step",
            "first",
            "second",
            "then",
        ]
        if any(indicator in response_lower for indicator in explanation_indicators):
            return ActionType.GIVE_STEP_BY_STEP_EXPLANATION

        # Check for practice problems
        practice_indicators = [
            "try this",
            "practice problem",
            "work through",
            "solve this",
        ]
        if any(indicator in response_lower for indicator in practice_indicators):
            return ActionType.ASSIGN_PRACTICE_PROBLEM

        # Check for wrap-up
        wrapup_indicators = [
            "to summarize",
            "in summary",
            "we've covered",
            "that's all",
            "great work today",
        ]
        if any(indicator in response_lower for indicator in wrapup_indicators):
            return ActionType.SUMMARIZE_AND_WRAP_UP

        # Default: classify as explanation
        return ActionType.GIVE_STEP_BY_STEP_EXPLANATION
