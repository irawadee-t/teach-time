"""
Pedagogical Action Space for ReAct Tutoring

Based on:
- ReAct (Yao et al., 2022): Synergizing Reasoning and Acting
- Cognitive tutoring literature (VanLehn, 2011)

7 actions: 5 pedagogical + 2 terminal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Action:
    """A pedagogical action."""
    name: str
    description: str
    is_terminal: bool = False


# =============================================================================
# ACTION SPACE (7 actions - clean, distinct, no overlap)
# =============================================================================

ACTIONS: dict[str, Action] = {
    
    # === SCAFFOLDING ===
    
    "GIVE_HINT": Action(
        name="GIVE_HINT",
        description="Provide a targeted clue toward the solution.",
    ),
    
    "ASK_QUESTION": Action(
        name="ASK_QUESTION",
        description="Ask a guiding question to lead student toward insight.",
    ),
    
    # === CORRECTIVE ===
    
    "CORRECT_ERROR": Action(
        name="CORRECT_ERROR",
        description="Point out and address a specific mistake.",
    ),
    
    "EXPLAIN_CONCEPT": Action(
        name="EXPLAIN_CONCEPT",
        description="Explain a concept the student is missing.",
    ),
    
    # === AFFIRMATIVE ===
    
    "CONFIRM_PROGRESS": Action(
        name="CONFIRM_PROGRESS",
        description="Acknowledge correct reasoning and guide to next step.",
    ),
    
    # === TERMINAL ===
    
    "END_SUCCESS": Action(
        name="END_SUCCESS",
        description="End session - student reached correct answer (judge confirmed).",
        is_terminal=True,
    ),
    
    "END_STUCK": Action(
        name="END_STUCK",
        description="End session - student cannot reach the answer.",
        is_terminal=True,
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_action(name: str) -> Optional[Action]:
    """Get action by name (case-insensitive)."""
    return ACTIONS.get(name.upper())


def get_action_names() -> list[str]:
    """Get all action names."""
    return list(ACTIONS.keys())


def get_terminal_actions() -> list[str]:
    """Get names of terminal actions."""
    return [name for name, action in ACTIONS.items() if action.is_terminal]


def is_valid_action(name: str) -> bool:
    """Check if action name is valid."""
    return name.upper() in ACTIONS


def is_terminal_action(name: str) -> bool:
    """Check if action terminates the session."""
    action = ACTIONS.get(name.upper())
    return action.is_terminal if action else False


# =============================================================================
# Action Aliases (for parser fuzzy matching)
# =============================================================================

ACTION_ALIASES: dict[str, str] = {
    # Common variations
    "HINT": "GIVE_HINT",
    "QUESTION": "ASK_QUESTION",
    "ASK": "ASK_QUESTION",
    "CORRECT": "CORRECT_ERROR",
    "ERROR": "CORRECT_ERROR",
    "EXPLAIN": "EXPLAIN_CONCEPT",
    "CONCEPT": "EXPLAIN_CONCEPT",
    "CONFIRM": "CONFIRM_PROGRESS",
    "PROGRESS": "CONFIRM_PROGRESS",
    "SUCCESS": "END_SUCCESS",
    "STUCK": "END_STUCK",
    
    # Longer variations that models sometimes produce
    "CORRECT_MISCONCEPTION": "CORRECT_ERROR",
    "ASK_GUIDING_QUESTION": "ASK_QUESTION",
    "PROBE_UNDERSTANDING": "ASK_QUESTION",
    "SCAFFOLD_STEP": "EXPLAIN_CONCEPT",
    "ENCOURAGE_REFLECTION": "ASK_QUESTION",
}


def resolve_action_alias(name: str) -> Optional[str]:
    """Resolve action name or alias to canonical form."""
    normalized = name.upper().replace(" ", "_").replace("-", "_")
    
    # Direct match
    if normalized in ACTIONS:
        return normalized
    
    # Alias match
    if normalized in ACTION_ALIASES:
        return ACTION_ALIASES[normalized]
    
    # Partial match (fallback)
    for action_name in ACTIONS:
        if normalized in action_name or action_name in normalized:
            return action_name
    
    return None
