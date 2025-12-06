"""
Prompt utilities for building agent prompts from templates.
"""

from pathlib import Path
from typing import List, Dict, Optional
from ..env.actions import ActionType
from ..env.metrics import format_metrics_for_prompt


# Get path to prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template file.

    Args:
        template_name: Name of template file (e.g., "cot_tutor.txt")

    Returns:
        Template content as string
    """
    template_path = PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        return f.read()


def format_conversation_history(
    dialogue_history: List[tuple],
    max_turns: int = 10
) -> str:
    """
    Format dialogue history for prompt inclusion.

    Args:
        dialogue_history: List of (speaker, utterance) tuples
        max_turns: Maximum number of recent turns to include

    Returns:
        Formatted conversation string
    """
    if not dialogue_history:
        return "[Session just started]"

    recent = dialogue_history[-max_turns:]
    formatted = []
    for speaker, utterance in recent:
        formatted.append(f"{speaker.capitalize()}: {utterance}")

    return "\n".join(formatted)


def format_learning_objectives(objectives: List[str]) -> str:
    """Format learning objectives as bulleted list."""
    return "\n".join(f"- {obj}" for obj in objectives)


def format_available_actions(action_types: List[ActionType]) -> str:
    """
    Format available actions with descriptions.

    Args:
        action_types: List of ActionType enums

    Returns:
        Formatted string describing available actions
    """
    formatted = []
    for action_type in action_types:
        desc = ActionType.get_action_description(action_type)
        formatted.append(f"\n**{action_type.value}**\n{desc}")

    return "\n".join(formatted)


def build_cot_tutor_prompt(
    topic: str,
    learning_objectives: List[str],
    conversation_history: List[tuple],
    student_utterance: str,
) -> str:
    """
    Build prompt for Baseline-CoT tutor.

    Args:
        topic: The topic being taught
        learning_objectives: List of learning objectives
        conversation_history: Dialogue history
        student_utterance: Latest student message

    Returns:
        Complete prompt string
    """
    template = load_prompt_template("cot_tutor.txt")

    prompt = template.format(
        topic=topic,
        learning_objectives=format_learning_objectives(learning_objectives),
        conversation_history=format_conversation_history(conversation_history),
        student_utterance=student_utterance,
    )

    return prompt


def build_metric_cot_tutor_prompt(
    topic: str,
    learning_objectives: List[str],
    conversation_history: List[tuple],
    student_utterance: str,
    metrics: Dict,
) -> str:
    """
    Build prompt for Metric-CoT tutor.

    Args:
        topic: The topic being taught
        learning_objectives: List of learning objectives
        conversation_history: Dialogue history
        student_utterance: Latest student message
        metrics: Current teaching metrics

    Returns:
        Complete prompt string
    """
    template = load_prompt_template("metric_cot_tutor.txt")

    prompt = template.format(
        topic=topic,
        learning_objectives=format_learning_objectives(learning_objectives),
        metrics=format_metrics_for_prompt(metrics),
        conversation_history=format_conversation_history(conversation_history),
        student_utterance=student_utterance,
    )

    return prompt


def build_react_teacher_prompt(
    topic: str,
    learning_objectives: List[str],
    dialogue_summary: str,
    student_utterance: str,
    metrics: Dict,
    available_actions: List[ActionType],
) -> str:
    """
    Build prompt for ReAct-Teacher.

    Args:
        topic: The topic being taught
        learning_objectives: List of learning objectives
        dialogue_summary: Summary of recent dialogue
        student_utterance: Latest student message
        metrics: Current teaching metrics
        available_actions: List of available action types

    Returns:
        Complete prompt string
    """
    template = load_prompt_template("react_teacher.txt")

    prompt = template.format(
        topic=topic,
        learning_objectives=format_learning_objectives(learning_objectives),
        metrics=format_metrics_for_prompt(metrics),
        available_actions=format_available_actions(available_actions),
        dialogue_summary=dialogue_summary,
        student_utterance=student_utterance,
    )

    return prompt


def parse_react_response(response: str) -> tuple[str, str, str]:
    """
    Parse ReAct-style response into thought, action_type, and action_content.

    Expected format:
        Thought: ...
        Action: ACTION_NAME("content")

    Args:
        response: Raw LLM response

    Returns:
        Tuple of (thought, action_type, action_content)
        Returns ("", "", "") if parsing fails
    """
    lines = response.strip().split('\n')

    thought = ""
    action_type = ""
    action_content = ""

    # Extract Thought
    for i, line in enumerate(lines):
        if line.strip().startswith("Thought:"):
            # Collect thought (may span multiple lines until Action)
            thought_lines = [line.replace("Thought:", "").strip()]
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("Action:"):
                    break
                thought_lines.append(lines[j].strip())
            thought = " ".join(thought_lines)
            break

    # Extract Action
    for line in lines:
        if line.strip().startswith("Action:"):
            action_str = line.replace("Action:", "").strip()

            # Parse ACTION_NAME("content")
            if '(' in action_str and ')' in action_str:
                action_type = action_str[:action_str.index('(')].strip()
                # Extract content between quotes
                content_start = action_str.index('(') + 1
                content_end = action_str.rindex(')')
                content_with_quotes = action_str[content_start:content_end].strip()

                # Remove surrounding quotes if present
                if content_with_quotes.startswith('"') and content_with_quotes.endswith('"'):
                    action_content = content_with_quotes[1:-1]
                elif content_with_quotes.startswith("'") and content_with_quotes.endswith("'"):
                    action_content = content_with_quotes[1:-1]
                else:
                    action_content = content_with_quotes

            break

    return thought, action_type, action_content


def parse_cot_response(response: str) -> tuple[str, str]:
    """
    Parse CoT-style response into thought and response.

    Expected format:
        Thought: ...
        Response: ...

    Args:
        response: Raw LLM response

    Returns:
        Tuple of (thought, tutor_response)
    """
    lines = response.strip().split('\n')

    thought = ""
    tutor_response = ""

    # Extract Thought
    for i, line in enumerate(lines):
        if line.strip().startswith("Thought:"):
            thought_lines = [line.replace("Thought:", "").strip()]
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("Response:"):
                    break
                thought_lines.append(lines[j].strip())
            thought = " ".join(thought_lines)
            break

    # Extract Response
    for i, line in enumerate(lines):
        if line.strip().startswith("Response:"):
            response_lines = [line.replace("Response:", "").strip()]
            for j in range(i + 1, len(lines)):
                response_lines.append(lines[j].strip())
            tutor_response = " ".join(response_lines)
            break

    return thought, tutor_response
