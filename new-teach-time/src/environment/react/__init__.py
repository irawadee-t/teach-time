"""
ReAct pedagogical reasoning module.

Based on: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

Provides:
- Action space for Socratic tutoring
- Parser for ReAct-format responses (Thought → Action → Utterance)
"""

from .actions import (
    Action,
    ACTIONS,
    ACTION_ALIASES,
    get_action,
    get_action_names,
    get_terminal_actions,
    is_valid_action,
    is_terminal_action,
    resolve_action_alias,
)
from .parser import (
    TeacherResponse,
    parse_teacher_response,
    attempt_repair,
)

__all__ = [
    # Actions
    "Action",
    "ACTIONS",
    "ACTION_ALIASES",
    "get_action",
    "get_action_names",
    "get_terminal_actions",
    "is_valid_action",
    "is_terminal_action",
    "resolve_action_alias",
    # Parser
    "TeacherResponse",
    "parse_teacher_response",
    "attempt_repair",
]
