"""
Teaching quality metrics computation.

Metrics derived from tutoring literature and TeachLM-style work.
"""

import re
from typing import List, Dict, Union
from dataclasses import dataclass


@dataclass
class Turn:
    """Represents a single turn in the dialogue."""
    speaker: str  # "tutor" or "student"
    utterance: str
    turn_index: int
    action_type: str = None  # For tutor turns with pedagogical actions


def count_tokens(text: str) -> int:
    """
    Simple token counter (word-based approximation).
    For production, could use tiktoken or similar.
    """
    return len(text.split())


def is_question(text: str) -> bool:
    """Check if an utterance contains a question."""
    # Simple heuristic: contains '?' or starts with question words
    if '?' in text:
        return True
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'could you', 'do you', 'does']
    text_lower = text.lower().strip()
    return any(text_lower.startswith(qw) for qw in question_words)


def contains_confusion_indicator(text: str) -> bool:
    """Check if student utterance indicates confusion."""
    confusion_phrases = [
        "i don't understand",
        "i'm confused",
        "i'm not sure",
        "i don't know",
        "confused",
        "unclear",
        "not sure",
        "don't get it",
        "what do you mean",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in confusion_phrases)


def compute_metrics(dialogue_history: List[Turn]) -> Dict[str, Union[float, int, bool]]:
    """
    Compute teaching quality metrics from dialogue history.

    Args:
        dialogue_history: List of Turn objects representing the conversation

    Returns:
        Dictionary of metric name -> value
    """
    if not dialogue_history:
        return _get_empty_metrics()

    tutor_turns = [t for t in dialogue_history if t.speaker == "tutor"]
    student_turns = [t for t in dialogue_history if t.speaker == "student"]

    # Token counts
    tutor_tokens = sum(count_tokens(t.utterance) for t in tutor_turns)
    student_tokens = sum(count_tokens(t.utterance) for t in student_turns)
    total_tokens = tutor_tokens + student_tokens

    # Core metrics
    metrics = {
        # 1. Student talk ratio (target: 0.5-0.8)
        "student_talk_ratio": student_tokens / total_tokens if total_tokens > 0 else 0.0,

        # 2. Number of questions asked by tutor
        "num_questions_asked": sum(1 for t in tutor_turns if is_question(t.utterance)),

        # 3. Did tutor ask about background/prior knowledge?
        "asked_background": any(
            t.action_type == "Ask_Background" for t in tutor_turns if t.action_type
        ),

        # 4. Checks of understanding in last k turns (k=5)
        "checks_of_understanding_last_k_turns": _count_recent_checks(tutor_turns, k=5),

        # 5. Session progress
        "session_step": len(dialogue_history),

        # 6. Average tutor utterance length
        "avg_tutor_utterance_length": (
            tutor_tokens / len(tutor_turns) if tutor_turns else 0.0
        ),

        # 7. Student confusion indicators
        "student_confusion_indicators": sum(
            1 for t in student_turns if contains_confusion_indicator(t.utterance)
        ),

        # 8. Has tutor given a summary?
        "has_given_summary": any(
            t.action_type == "Summarize_And_Wrap_Up" for t in tutor_turns if t.action_type
        ),

        # 9. Number of practice problems assigned
        "num_practice_problems_assigned": sum(
            1 for t in tutor_turns
            if t.action_type == "Assign_Practice_Problem"
        ),

        # 10. Turn at which background was asked (if at all)
        "background_asked_at_turn": _find_background_turn(tutor_turns),
    }

    return metrics


def _count_recent_checks(tutor_turns: List[Turn], k: int = 5) -> int:
    """Count understanding checks in the last k tutor turns."""
    recent_turns = tutor_turns[-k:] if len(tutor_turns) >= k else tutor_turns
    return sum(
        1 for t in recent_turns
        if t.action_type == "Ask_Check_Understanding"
    )


def _find_background_turn(tutor_turns: List[Turn]) -> int:
    """Find the turn index where background was first asked. Returns -1 if never asked."""
    for t in tutor_turns:
        if t.action_type == "Ask_Background":
            return t.turn_index
    return -1


def _get_empty_metrics() -> Dict[str, Union[float, int, bool]]:
    """Return metrics for an empty dialogue."""
    return {
        "student_talk_ratio": 0.0,
        "num_questions_asked": 0,
        "asked_background": False,
        "checks_of_understanding_last_k_turns": 0,
        "session_step": 0,
        "avg_tutor_utterance_length": 0.0,
        "student_confusion_indicators": 0,
        "has_given_summary": False,
        "num_practice_problems_assigned": 0,
        "background_asked_at_turn": -1,
    }


def format_metrics_for_prompt(metrics: Dict[str, Union[float, int, bool]]) -> str:
    """
    Format metrics in a human-readable way for inclusion in prompts.

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Formatted string suitable for prompt inclusion
    """
    formatted = "Teaching Metrics:\n"
    formatted += f"- Student talk ratio: {metrics['student_talk_ratio']:.2f}\n"
    formatted += f"- Questions asked by tutor: {metrics['num_questions_asked']}\n"
    formatted += f"- Background probed: {'Yes' if metrics['asked_background'] else 'No'}\n"
    formatted += f"- Recent understanding checks: {metrics['checks_of_understanding_last_k_turns']}\n"
    formatted += f"- Current turn: {metrics['session_step']}\n"
    formatted += f"- Practice problems assigned: {metrics['num_practice_problems_assigned']}\n"

    return formatted


def evaluate_metrics_against_targets(
    metrics: Dict[str, Union[float, int, bool]],
    max_turns: int = 10
) -> Dict[str, float]:
    """
    Evaluate metrics against pedagogical best-practice targets.

    Args:
        metrics: Computed metrics
        max_turns: Maximum turns in episode (for normalization)

    Returns:
        Dictionary of metric_name -> score (0-1)
    """
    scores = {}

    # 1. Student talk ratio (target: 0.5-0.8)
    ratio = metrics["student_talk_ratio"]
    if 0.5 <= ratio <= 0.8:
        scores["student_talk_ratio"] = 1.0
    else:
        # Quadratic penalty for distance from range
        if ratio < 0.5:
            scores["student_talk_ratio"] = max(0, 1 - (0.5 - ratio) ** 2 * 4)
        else:
            scores["student_talk_ratio"] = max(0, 1 - (ratio - 0.8) ** 2 * 4)

    # 2. Questions asked (target: at least 5 per 10 turns)
    target_questions = (metrics["session_step"] / max_turns) * 5
    questions_ratio = metrics["num_questions_asked"] / max(target_questions, 1)
    scores["num_questions_asked"] = min(1.0, questions_ratio)

    # 3. Background asked early (target: by turn 3)
    if metrics["asked_background"]:
        bg_turn = metrics["background_asked_at_turn"]
        if bg_turn <= 3:
            scores["asked_background_early"] = 1.0
        else:
            # Linear penalty after turn 3
            scores["asked_background_early"] = max(0, 1 - (bg_turn - 3) * 0.1)
    else:
        scores["asked_background_early"] = 0.0

    # 4. Understanding checks (target: at least 3 per session)
    checks = metrics["checks_of_understanding_last_k_turns"]
    scores["checks_of_understanding"] = min(1.0, checks / 3.0)

    return scores
