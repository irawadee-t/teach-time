"""
Pedagogical actions for the TeachTime framework.

Defines the discrete action space for teaching agents.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ActionType(Enum):
    """Discrete pedagogical action primitives."""

    ASK_OPEN_QUESTION = "Ask_Open_Question"
    ASK_CHECK_UNDERSTANDING = "Ask_Check_Understanding"
    GIVE_STEP_BY_STEP_EXPLANATION = "Give_Step_By_Step_Explanation"
    ASK_BACKGROUND = "Ask_Background"
    ASSIGN_PRACTICE_PROBLEM = "Assign_Practice_Problem"
    SUMMARIZE_AND_WRAP_UP = "Summarize_And_Wrap_Up"

    @classmethod
    def get_action_description(cls, action_type: 'ActionType') -> str:
        """Get a description of what each action type does."""
        descriptions = {
            cls.ASK_OPEN_QUESTION: (
                "Ask an open-ended question to encourage student reasoning and elaboration. "
                "Use this to probe student understanding without giving away answers."
            ),
            cls.ASK_CHECK_UNDERSTANDING: (
                "Ask a targeted question to check if the student understands a specific concept. "
                "Use this to verify comprehension before moving forward."
            ),
            cls.GIVE_STEP_BY_STEP_EXPLANATION: (
                "Provide a clear, step-by-step explanation of a concept or solution. "
                "Use this when the student needs explicit guidance."
            ),
            cls.ASK_BACKGROUND: (
                "Probe the student's prior knowledge and experience with related concepts. "
                "Use this early in the session to gauge where to start."
            ),
            cls.ASSIGN_PRACTICE_PROBLEM: (
                "Give the student a practice problem to work through. "
                "Use this to help solidify understanding through application."
            ),
            cls.SUMMARIZE_AND_WRAP_UP: (
                "Summarize the key points covered and wrap up the session. "
                "Use this near the end to reinforce learning."
            ),
        }
        return descriptions[action_type]

    @classmethod
    def get_all_descriptions(cls) -> dict:
        """Get descriptions for all action types."""
        return {action.value: cls.get_action_description(action) for action in cls}


@dataclass
class PedagogicalAction:
    """
    A discrete teaching action with its natural language content.

    Attributes:
        action_type: The type of pedagogical action
        content: The specific content/argument for this action (e.g., the question text)
        metadata: Optional additional information about the action
    """
    action_type: ActionType
    content: str
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "content": self.content,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PedagogicalAction':
        """Create from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            content=data["content"],
            metadata=data.get("metadata")
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f'{self.action_type.value}("{self.content[:50]}...")'


# Action space configurations for ablations
ACTION_SPACE_CONFIGS = {
    "small": [
        ActionType.ASK_OPEN_QUESTION,
        ActionType.GIVE_STEP_BY_STEP_EXPLANATION,
        ActionType.SUMMARIZE_AND_WRAP_UP,
    ],
    "default": list(ActionType),  # All 6 actions
    "large": list(ActionType) + [
        # Note: Large action space would require extending the enum
        # For now, "large" is the same as "default"
        # Could be extended with: Reflect_On_Solution, Ask_Student_To_Teach_Back, etc.
    ],
}


def get_action_space(config_name: str = "default") -> list[ActionType]:
    """
    Get an action space configuration by name.

    Args:
        config_name: One of "small", "default", or "large"

    Returns:
        List of ActionType values for the specified configuration
    """
    if config_name not in ACTION_SPACE_CONFIGS:
        raise ValueError(
            f"Unknown action space config: {config_name}. "
            f"Choose from: {list(ACTION_SPACE_CONFIGS.keys())}"
        )
    return ACTION_SPACE_CONFIGS[config_name]
