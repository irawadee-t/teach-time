"""
ReAct-Teacher agent.

Uses ReAct (Reason + Act) framework with discrete pedagogical action space
and explicit teaching metrics in observations.
"""

from typing import List, Optional
from .base_agent import BaseTutorAgent
from ..env.teaching_env import Observation
from ..env.actions import PedagogicalAction, ActionType, get_action_space
from ..llm.prompt_utils import build_react_teacher_prompt, parse_react_response


class ReActTeacherAgent(BaseTutorAgent):
    """
    ReAct-Teacher agent.

    - Uses ReAct-style reasoning (Thought → Action → Observation loop)
    - Discrete pedagogical action space
    - Teaching metrics explicitly in observations
    - Agent must choose one action per step
    """

    def __init__(
        self,
        agent_id: str = "react_teacher",
        llm_client=None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        action_space_config: str = "default",
        include_metrics: bool = True,
    ):
        """
        Args:
            agent_id: Agent identifier
            llm_client: LLM client
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            action_space_config: Action space size ("small", "default", "large")
            include_metrics: Whether to include metrics in prompt (for ablation)
        """
        super().__init__(agent_id, llm_client, temperature, max_tokens)
        self.action_space_config = action_space_config
        self.include_metrics = include_metrics
        self.available_actions = get_action_space(action_space_config)

    def act(self, observation: Observation) -> PedagogicalAction:
        """
        Choose a pedagogical action using ReAct reasoning.

        Args:
            observation: Current observation from environment

        Returns:
            PedagogicalAction with explicit action type
        """
        # Build prompt
        prompt = build_react_teacher_prompt(
            topic=self.current_task.topic,
            learning_objectives=self.current_task.learning_objectives,
            dialogue_summary=observation.dialogue_summary,
            student_utterance=observation.student_utterance,
            metrics=observation.metrics if self.include_metrics else {},
            available_actions=self.available_actions,
        )

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.call(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse ReAct response
        thought, action_type_str, action_content = parse_react_response(response)

        # Validate and map action type
        action_type = self._validate_action_type(action_type_str)

        # Record thought for analysis
        self.thoughts.append({
            "step": observation.step_index,
            "thought": thought,
            "action_type": action_type.value,
            "action_content": action_content,
            "metrics": observation.metrics.copy() if self.include_metrics else {},
            "raw_response": response,
        })

        # Record this turn
        self.record_turn("tutor", action_content)

        return PedagogicalAction(
            action_type=action_type,
            content=action_content,
            metadata={
                "thought": thought,
                "metrics": observation.metrics if self.include_metrics else {},
            }
        )

    def _validate_action_type(self, action_type_str: str) -> ActionType:
        """
        Validate and map action type string to ActionType enum.

        If invalid or not in available actions, returns a default.
        """
        # Try to match to ActionType
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            # Try fuzzy matching
            action_type = self._fuzzy_match_action(action_type_str)

        # Check if action is in available action space
        if action_type not in self.available_actions:
            # Fall back to first available action
            action_type = self.available_actions[0]

        return action_type

    def _fuzzy_match_action(self, action_str: str) -> ActionType:
        """
        Attempt to match action string to ActionType using fuzzy matching.

        Args:
            action_str: Action string from LLM (may have variations)

        Returns:
            Best matching ActionType, or default
        """
        action_str_lower = action_str.lower().replace("_", " ").replace("-", " ")

        # Define matching keywords for each action
        action_keywords = {
            ActionType.ASK_OPEN_QUESTION: ["ask", "open", "question"],
            ActionType.ASK_CHECK_UNDERSTANDING: ["check", "understanding", "understand"],
            ActionType.GIVE_STEP_BY_STEP_EXPLANATION: ["explain", "explanation", "step"],
            ActionType.ASK_BACKGROUND: ["background", "prior", "knowledge"],
            ActionType.ASSIGN_PRACTICE_PROBLEM: ["practice", "problem", "assign"],
            ActionType.SUMMARIZE_AND_WRAP_UP: ["summarize", "summary", "wrap"],
        }

        # Score each action type by keyword matches
        best_match = None
        best_score = 0

        for action_type, keywords in action_keywords.items():
            score = sum(1 for kw in keywords if kw in action_str_lower)
            if score > best_score:
                best_score = score
                best_match = action_type

        # Return best match or default
        return best_match if best_match else self.available_actions[0]


class ReActTeacherNoMetrics(ReActTeacherAgent):
    """
    Ablation: ReAct-Teacher without metrics in observation.

    Same ReAct structure but metrics are not provided to the agent.
    """

    def __init__(
        self,
        agent_id: str = "react_teacher_no_metrics",
        llm_client=None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        action_space_config: str = "default",
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
            action_space_config=action_space_config,
            include_metrics=False,  # Key difference
        )


class ReActTeacherSmallActionSpace(ReActTeacherAgent):
    """
    Ablation: ReAct-Teacher with small action space (3 actions).
    """

    def __init__(
        self,
        agent_id: str = "react_teacher_small_actions",
        llm_client=None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
            action_space_config="small",  # Small action space
            include_metrics=True,
        )


class ReActTeacherLargeActionSpace(ReActTeacherAgent):
    """
    Ablation: ReAct-Teacher with large action space.

    For now, "large" is the same as "default" but could be extended.
    """

    def __init__(
        self,
        agent_id: str = "react_teacher_large_actions",
        llm_client=None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
            action_space_config="large",  # Large action space
            include_metrics=True,
        )
