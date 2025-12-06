"""
AppBench-style scoring configuration and computation.

Maps pedagogical specifications to structured scores.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class PedagogyScoreBreakdown:
    """Breakdown of pedagogy score components."""
    student_talk_ratio_score: float
    questions_asked_score: float
    background_early_score: float
    understanding_checks_score: float
    overall_score: float


@dataclass
class LearningScoreBreakdown:
    """Breakdown of learning score components."""
    learning_gain_score: float
    knowledge_state_score: float
    overall_score: float


@dataclass
class QualityScoreBreakdown:
    """Breakdown of quality score components."""
    pedagogical_usefulness: float
    student_felt_heard: float
    no_hallucinations: float
    overall_score: float


@dataclass
class EpisodeScores:
    """Complete scores for an episode."""
    pedagogy: PedagogyScoreBreakdown
    learning: LearningScoreBreakdown
    quality: QualityScoreBreakdown
    total_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pedagogy": {
                "student_talk_ratio_score": self.pedagogy.student_talk_ratio_score,
                "questions_asked_score": self.pedagogy.questions_asked_score,
                "background_early_score": self.pedagogy.background_early_score,
                "understanding_checks_score": self.pedagogy.understanding_checks_score,
                "overall_score": self.pedagogy.overall_score,
            },
            "learning": {
                "learning_gain_score": self.learning.learning_gain_score,
                "knowledge_state_score": self.learning.knowledge_state_score,
                "overall_score": self.learning.overall_score,
            },
            "quality": {
                "pedagogical_usefulness": self.quality.pedagogical_usefulness,
                "student_felt_heard": self.quality.student_felt_heard,
                "no_hallucinations": self.quality.no_hallucinations,
                "overall_score": self.quality.overall_score,
            },
            "total_score": self.total_score,
        }


class ScoringEngine:
    """
    AppBench-style scoring engine for tutoring episodes.

    Evaluates episodes across three dimensions:
    1. Pedagogy: adherence to teaching best practices
    2. Learning: student learning outcomes
    3. Quality: overall conversation quality
    """

    def __init__(
        self,
        pedagogy_weight: float = 0.4,
        learning_weight: float = 0.4,
        quality_weight: float = 0.2,
        # Pedagogy targets
        student_talk_ratio_range: Tuple[float, float] = (0.5, 0.8),
        target_questions: int = 5,
        background_by_turn: int = 3,
        target_checks: int = 3,
        # Learning thresholds
        min_learning_gain: float = 0.1,
        good_learning_gain: float = 0.3,
    ):
        """
        Args:
            pedagogy_weight: Weight for pedagogy score
            learning_weight: Weight for learning score
            quality_weight: Weight for quality score
            student_talk_ratio_range: Target range for student talk ratio
            target_questions: Target number of questions per session
            background_by_turn: Turn by which background should be asked
            target_checks: Target number of understanding checks
            min_learning_gain: Minimum acceptable learning gain
            good_learning_gain: Good learning gain threshold
        """
        # Validate weights
        total_weight = pedagogy_weight + learning_weight + quality_weight
        if not math.isclose(total_weight, 1.0, rel_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.pedagogy_weight = pedagogy_weight
        self.learning_weight = learning_weight
        self.quality_weight = quality_weight

        self.student_talk_ratio_range = student_talk_ratio_range
        self.target_questions = target_questions
        self.background_by_turn = background_by_turn
        self.target_checks = target_checks
        self.min_learning_gain = min_learning_gain
        self.good_learning_gain = good_learning_gain

    def score_episode(
        self,
        final_metrics: Dict,
        pre_quiz_score: float,
        post_quiz_score: float,
        learning_gain: float,
        final_knowledge_state: Dict[str, float],
        quality_scores: Dict[str, float] = None,
    ) -> EpisodeScores:
        """
        Compute comprehensive scores for an episode.

        Args:
            final_metrics: Final teaching metrics from episode
            pre_quiz_score: Pre-quiz score
            post_quiz_score: Post-quiz score
            learning_gain: Normalized learning gain
            final_knowledge_state: Final concept mastery levels
            quality_scores: Optional LLM-judge quality scores

        Returns:
            EpisodeScores object with detailed breakdown
        """
        # Compute pedagogy scores
        pedagogy = self._score_pedagogy(final_metrics)

        # Compute learning scores
        learning = self._score_learning(
            learning_gain=learning_gain,
            final_knowledge_state=final_knowledge_state,
        )

        # Compute quality scores
        if quality_scores is None:
            quality_scores = {
                "pedagogical_usefulness": 0.5,
                "student_felt_heard": 0.5,
                "no_hallucinations": 1.0,  # Assume no hallucinations if not judged
            }
        quality = self._score_quality(quality_scores)

        # Compute total weighted score
        total_score = (
            self.pedagogy_weight * pedagogy.overall_score +
            self.learning_weight * learning.overall_score +
            self.quality_weight * quality.overall_score
        )

        return EpisodeScores(
            pedagogy=pedagogy,
            learning=learning,
            quality=quality,
            total_score=total_score,
        )

    def _score_pedagogy(self, metrics: Dict) -> PedagogyScoreBreakdown:
        """Score pedagogical metrics."""
        # 1. Student talk ratio (target: 0.5-0.8)
        ratio = metrics.get("student_talk_ratio", 0.0)
        ratio_score = self._score_in_range(
            ratio,
            self.student_talk_ratio_range[0],
            self.student_talk_ratio_range[1]
        )

        # 2. Questions asked (target: 5+)
        num_questions = metrics.get("num_questions_asked", 0)
        questions_score = min(1.0, num_questions / self.target_questions)

        # 3. Background asked early (target: by turn 3)
        bg_turn = metrics.get("background_asked_at_turn", -1)
        if bg_turn == -1:
            # Never asked
            bg_score = 0.0
        elif bg_turn <= self.background_by_turn:
            # Asked on time
            bg_score = 1.0
        else:
            # Asked late - linear penalty
            bg_score = max(0.0, 1.0 - (bg_turn - self.background_by_turn) * 0.1)

        # 4. Understanding checks (target: 3+)
        checks = metrics.get("checks_of_understanding_last_k_turns", 0)
        checks_score = min(1.0, checks / self.target_checks)

        # Weighted average (equal weights for now)
        overall = (ratio_score + questions_score + bg_score + checks_score) / 4.0

        return PedagogyScoreBreakdown(
            student_talk_ratio_score=ratio_score,
            questions_asked_score=questions_score,
            background_early_score=bg_score,
            understanding_checks_score=checks_score,
            overall_score=overall,
        )

    def _score_learning(
        self,
        learning_gain: float,
        final_knowledge_state: Dict[str, float],
    ) -> LearningScoreBreakdown:
        """Score learning outcomes."""
        # 1. Learning gain (normalized by good_learning_gain threshold)
        if learning_gain >= self.good_learning_gain:
            gain_score = 1.0
        elif learning_gain >= self.min_learning_gain:
            # Linear interpolation between min and good
            gain_score = 0.5 + 0.5 * (learning_gain - self.min_learning_gain) / (
                self.good_learning_gain - self.min_learning_gain
            )
        else:
            # Below minimum
            gain_score = 0.5 * (learning_gain / self.min_learning_gain)

        # 2. Final knowledge state (average mastery)
        if final_knowledge_state:
            avg_mastery = sum(final_knowledge_state.values()) / len(final_knowledge_state)
            knowledge_score = avg_mastery
        else:
            knowledge_score = 0.0

        # Weighted average (60% gain, 40% final knowledge)
        overall = 0.6 * gain_score + 0.4 * knowledge_score

        return LearningScoreBreakdown(
            learning_gain_score=gain_score,
            knowledge_state_score=knowledge_score,
            overall_score=overall,
        )

    def _score_quality(self, quality_scores: Dict[str, float]) -> QualityScoreBreakdown:
        """Score conversation quality (from LLM judge or defaults)."""
        usefulness = quality_scores.get("pedagogical_usefulness", 0.5)
        felt_heard = quality_scores.get("student_felt_heard", 0.5)
        no_hallucinations = quality_scores.get("no_hallucinations", 1.0)

        # Weighted average (50% usefulness, 30% felt heard, 20% no hallucinations)
        overall = 0.5 * usefulness + 0.3 * felt_heard + 0.2 * no_hallucinations

        return QualityScoreBreakdown(
            pedagogical_usefulness=usefulness,
            student_felt_heard=felt_heard,
            no_hallucinations=no_hallucinations,
            overall_score=overall,
        )

    @staticmethod
    def _score_in_range(value: float, min_val: float, max_val: float) -> float:
        """
        Score a value based on whether it falls in target range.

        Returns 1.0 if in range, with quadratic penalty outside range.
        """
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            # Penalty for being below range
            distance = min_val - value
            return max(0.0, 1.0 - (distance ** 2) * 4)
        else:
            # Penalty for being above range
            distance = value - max_val
            return max(0.0, 1.0 - (distance ** 2) * 4)
