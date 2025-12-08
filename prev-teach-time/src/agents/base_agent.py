"""
Base agent interface for TeachTime tutors.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..env.teaching_env import Observation
from ..env.actions import PedagogicalAction
from ..env.tasks.base import TaskSpec


class BaseTutorAgent(ABC):
    """
    Abstract base class for tutor agents.

    All agents must implement the act() method which takes an observation
    and returns a pedagogical action.
    """

    def __init__(
        self,
        agent_id: str,
        llm_client,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent
            llm_client: LLM client for generating responses
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens to generate
        """
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Task context (set when episode starts)
        self.current_task: Optional[TaskSpec] = None

        # Episode state tracking
        self.dialogue_history = []  # List of (speaker, utterance) tuples
        self.thoughts = []  # List of reasoning traces (for analysis)

    @abstractmethod
    def act(self, observation: Observation) -> PedagogicalAction:
        """
        Choose a pedagogical action based on the observation.

        Args:
            observation: Current observation from environment

        Returns:
            PedagogicalAction to take
        """
        pass

    def reset(self, task: TaskSpec):
        """
        Reset agent state for a new episode.

        Args:
            task: The task for this episode
        """
        self.current_task = task
        self.dialogue_history = []
        self.thoughts = []

    def record_turn(self, speaker: str, utterance: str):
        """Record a dialogue turn for context tracking."""
        self.dialogue_history.append((speaker, utterance))

    def get_agent_info(self) -> dict:
        """Get agent metadata for logging."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def get_reasoning_trace(self) -> list:
        """Get list of thoughts/reasoning for analysis."""
        return self.thoughts.copy()
