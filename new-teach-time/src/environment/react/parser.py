"""
ReAct Response Parser

Parses teacher responses in ReAct format:
```
Thought: [reasoning about student state and strategy]
Action: [ACTION_NAME]
Utterance: [what to say to student]
```

Based on ReAct (Yao et al., 2022): Synergizing Reasoning and Acting
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from .actions import (
    is_valid_action,
    is_terminal_action,
    get_action_names,
    resolve_action_alias,
    ACTIONS,
)


@dataclass
class TeacherResponse:
    """Parsed teacher response in ReAct format."""
    thought: str
    action: str
    utterance: str
    is_terminal: bool = False
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    raw_output: str = ""


def parse_teacher_response(raw_output: str) -> TeacherResponse:
    """
    Parse teacher ReAct output.
    
    Expected format:
    ```
    Thought: [reasoning]
    Action: [ACTION_NAME]
    Utterance: [text]
    ```
    """
    response = TeacherResponse(
        thought="",
        action="",
        utterance="",
        raw_output=raw_output,
    )
    
    text = raw_output.strip()
    
    # === Extract Thought ===
    thought_match = re.search(
        r'Thought:\s*(.+?)(?=\n\s*Action:|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        response.thought = thought_match.group(1).strip().rstrip('`')
    
    # === Extract Action ===
    action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
    if action_match:
        raw_action = action_match.group(1).strip().upper()
        resolved = resolve_action_alias(raw_action)
        response.action = resolved if resolved else raw_action
    
    # === Extract Utterance ===
    utterance_match = re.search(
        r'Utterance:\s*(.+?)(?:```|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if utterance_match:
        response.utterance = utterance_match.group(1).strip()
        # Clean quotes
        response.utterance = re.sub(r'^["\']|["\']$', '', response.utterance).strip()
    
    # === Validate ===
    errors = []
    
    if not response.action:
        errors.append("No action found")
        response.is_valid = False
    elif not is_valid_action(response.action):
        errors.append(f"Invalid action: {response.action}")
        response.is_valid = False
    
    if not response.utterance:
        errors.append("No utterance found")
        response.is_valid = False
    
    response.validation_errors = errors
    response.is_terminal = is_terminal_action(response.action)
    
    return response


def attempt_repair(response: TeacherResponse) -> TeacherResponse:
    """
    Attempt to repair common parsing failures.
    
    Handles cases where model uses slightly different formatting.
    """
    # Try to find action in raw output
    if not response.action or not is_valid_action(response.action):
        for action_name in ACTIONS.keys():
            if action_name in response.raw_output.upper():
                response.action = action_name
                response.is_valid = True
                response.is_terminal = is_terminal_action(action_name)
                response.validation_errors = [
                    e for e in response.validation_errors
                    if "action" not in e.lower()
                ]
                break
    
    # Try to extract utterance from remaining text
    if not response.utterance:
        lines = response.raw_output.split('\n')
        utterance_lines = []
        found_action = False
        
        for line in lines:
            line_lower = line.lower()
            if 'action:' in line_lower:
                found_action = True
                continue
            if found_action and line.strip() and 'thought:' not in line_lower:
                utterance_lines.append(line.strip())
        
        if utterance_lines:
            response.utterance = ' '.join(utterance_lines)
            response.is_valid = bool(response.action and is_valid_action(response.action))
            response.validation_errors = [
                e for e in response.validation_errors
                if "utterance" not in e.lower()
            ]
    
    return response
