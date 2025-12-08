"""
Core data models for TutorBench evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class UseCase(str, Enum):
    """Three core tutoring use cases from TutorBench."""
    ADAPTIVE = "adaptive"  # Adaptive explanation generation
    ASSESSMENT = "assessment"  # Assessment and feedback
    ACTIVE_LEARNING = "active_learning"  # Active learning support (hints)


class EvaluationDimension(str, Enum):
    """Eight evaluation dimensions for rubric criteria."""
    INSTRUCTION_FOLLOWING = "instruction_following"
    TRUTHFULNESS = "truthfulness"
    CONCISENESS_RELEVANCE = "conciseness_relevance"
    STYLE_TONE = "style_tone"
    VISUAL_PERCEPTION = "visual_perception"  # Multimodal only
    VISUAL_REASONING = "visual_reasoning"  # Multimodal only
    STUDENT_LEVEL_CALIBRATION = "student_level_calibration"
    EMOTIONAL_COMPONENT = "emotional_component"


class TutoringSkill(str, Enum):
    """Eight tutoring skills assessed in rubrics."""
    ASKING_GUIDING_QUESTIONS = "asking_guiding_questions"
    IDENTIFYING_CORE_DIFFICULTY = "identifying_core_difficulty"
    IDENTIFYING_CORRECT_STEPS = "identifying_correct_steps"
    IDENTIFYING_INCORRECT_STEPS = "identifying_incorrect_steps"
    INCLUDING_EXAMPLES = "including_examples"
    PROVIDING_ALTERNATIVE_SOLUTIONS = "providing_alternative_solutions"
    STATING_KNOWLEDGE = "stating_knowledge"  # Definitions, theorems, etc.
    STEP_BY_STEP_HELP = "step_by_step_help"


@dataclass
class Rubric:
    """
    Sample-specific rubric criterion.

    Following TutorBench methodology:
    - Self-contained, mutually exclusive, verifiable
    - Weighted: -5 (critical negative), 1 (non-critical), 5 (critical positive)
    - Tagged with evaluation dimension and tutoring skill
    """
    criterion: str
    weight: int  # -5, 1, or 5
    evaluation_dimension: EvaluationDimension
    tutoring_skill: Optional[TutoringSkill] = None
    is_objective: bool = True
    is_explicit: bool = True

    def __post_init__(self):
        if self.weight not in {-5, 1, 5}:
            raise ValueError(f"Weight must be -5, 1, or 5, got {self.weight}")


@dataclass
class Sample:
    """
    TutorBench evaluation sample.

    Each sample includes:
    - System prompt (defines tutoring goal)
    - Question + student context (may include images)
    - Sample-specific rubrics for evaluation
    """
    sample_id: str
    use_case: UseCase
    subject: str  # Biology, Physics, Chemistry, Statistics, Calculus, CS
    system_prompt: str
    messages: List[Dict[str, Any]]  # Chat messages (user + assistant context)
    rubrics: List[Rubric]
    is_multimodal: bool = False
    images: Optional[List[str]] = None  # Paths or base64 encoded images

    @property
    def has_critical_rubrics(self) -> bool:
        """Check if sample has any critical rubrics (weight ±5)."""
        return any(abs(r.weight) == 5 for r in self.rubrics)


@dataclass
class RubricRating:
    """Individual rubric evaluation result."""
    rubric: Rubric
    passed: bool  # 0 or 1
    explanation: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a single sample.

    Includes:
    - Model response
    - Individual rubric ratings
    - Weighted score (ARR_w)
    """
    sample_id: str
    model_name: str
    model_response: str
    rubric_ratings: List[RubricRating]
    weighted_score: float  # ARR_w ∈ [0, 1]

    @property
    def pass_rate(self) -> float:
        """Simple pass rate (ignoring weights)."""
        if not self.rubric_ratings:
            return 0.0
        return sum(r.passed for r in self.rubric_ratings) / len(self.rubric_ratings)

    @property
    def critical_rubric_pass_rate(self) -> float:
        """Pass rate on critical rubrics only (|weight| = 5)."""
        critical = [r for r in self.rubric_ratings if abs(r.rubric.weight) == 5]
        if not critical:
            return 1.0
        return sum(r.passed for r in critical) / len(critical)
