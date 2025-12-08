"""
Base task interface for TeachTime learning domains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random


@dataclass
class QuizQuestion:
    """A single quiz question with answers and rubric."""
    question: str
    question_type: str  # "multiple_choice" or "short_answer"
    correct_answer: str
    options: Optional[List[str]] = None  # For multiple choice
    rubric: Optional[str] = None  # For grading short answer
    concept: str = ""  # Which concept this question tests

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "type": self.question_type,
            "correct_answer": self.correct_answer,
            "options": self.options,
            "rubric": self.rubric,
            "concept": self.concept,
        }


@dataclass
class TaskSpec:
    """Specification for a learning task."""
    task_id: str
    topic: str
    description: str
    learning_objectives: List[str]
    key_concepts: List[str]
    common_misconceptions: List[str]
    difficulty: str  # "easy", "medium", "hard"
    pre_quiz: List[QuizQuestion]
    post_quiz: List[QuizQuestion]
    hints: List[str] = None
    examples: List[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "description": self.description,
            "learning_objectives": self.learning_objectives,
            "key_concepts": self.key_concepts,
            "common_misconceptions": self.common_misconceptions,
            "difficulty": self.difficulty,
            "pre_quiz": [q.to_dict() for q in self.pre_quiz],
            "post_quiz": [q.to_dict() for q in self.post_quiz],
            "hints": self.hints or [],
            "examples": self.examples or [],
        }


class TaskDomain(ABC):
    """Abstract base class for learning domains."""

    def __init__(self, random_seed: Optional[int] = None):
        self.random = random.Random(random_seed)

    @abstractmethod
    def get_all_tasks(self) -> List[TaskSpec]:
        """Return all tasks in this domain."""
        pass

    def get_task(self, task_id: str) -> TaskSpec:
        """Get a specific task by ID."""
        tasks = {t.task_id: t for t in self.get_all_tasks()}
        if task_id not in tasks:
            raise ValueError(f"Unknown task: {task_id}")
        return tasks[task_id]

    def sample_task(self, difficulty: Optional[str] = None) -> TaskSpec:
        """Sample a random task, optionally filtered by difficulty."""
        tasks = self.get_all_tasks()
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if not tasks:
            raise ValueError(f"No tasks found with difficulty: {difficulty}")
        return self.random.choice(tasks)

    @abstractmethod
    def get_domain_name(self) -> str:
        """Return the name of this domain."""
        pass


def grade_multiple_choice(student_answer: str, correct_answer: str) -> float:
    """
    Grade a multiple choice question.

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # Normalize both answers
    student_norm = student_answer.strip().upper()
    correct_norm = correct_answer.strip().upper()
    return 1.0 if student_norm == correct_norm else 0.0


def grade_short_answer_simple(student_answer: str, correct_answer: str) -> float:
    """
    Simple rule-based grading for short answer questions.

    Checks if key terms from correct answer appear in student answer.
    Returns a score from 0.0 to 1.0.
    """
    if not student_answer or not correct_answer:
        return 0.0

    # Normalize
    student_lower = student_answer.lower().strip()
    correct_lower = correct_answer.lower().strip()

    # Exact match
    if student_lower == correct_lower:
        return 1.0

    # Check if all important words from correct answer appear in student answer
    # Simple heuristic: extract words longer than 3 chars
    correct_words = set(w for w in correct_lower.split() if len(w) > 3)
    student_words = set(student_lower.split())

    if not correct_words:
        return 0.5  # Can't determine, give partial credit

    overlap = correct_words & student_words
    score = len(overlap) / len(correct_words)

    return score


def compute_quiz_score(
    questions: List[QuizQuestion],
    student_answers: List[str]
) -> float:
    """
    Compute overall quiz score.

    Args:
        questions: List of quiz questions
        student_answers: List of student answers (parallel to questions)

    Returns:
        Score from 0.0 to 1.0
    """
    if len(questions) != len(student_answers):
        raise ValueError("Number of answers must match number of questions")

    if not questions:
        return 0.0

    total_score = 0.0
    for q, ans in zip(questions, student_answers):
        if q.question_type == "multiple_choice":
            total_score += grade_multiple_choice(ans, q.correct_answer)
        else:  # short_answer
            total_score += grade_short_answer_simple(ans, q.correct_answer)

    return total_score / len(questions)
