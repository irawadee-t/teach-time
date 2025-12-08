"""
Metrics calculation for Pedagogical Effectiveness Score (PES).

Implements weighted scoring:
- Layer 1 (8 dimensions): 80% weight
- Layer 2 (Question depth): 10% weight
- Layer 3 (ICAP engagement): 10% weight
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DimensionScore:
    """Score for a single pedagogical dimension."""
    dimension: str
    score: int  # 1-5
    justification: str
    evidence: List[str]

    def normalized_score(self) -> float:
        """Normalize 1-5 score to 0-1 range."""
        return (self.score - 1) / 4.0


@dataclass
class QuestionDepthScore:
    """Score for question depth analysis."""
    score: int  # 1-5
    question_count: Dict[str, int]
    question_examples: Dict[str, List[str]]
    justification: str

    def normalized_score(self) -> float:
        """Normalize 1-5 score to 0-1 range."""
        return (self.score - 1) / 4.0


@dataclass
class ICAPScore:
    """Score for ICAP engagement classification."""
    score: int  # 1-5
    engagement_distribution: Dict[str, float]
    turn_classifications: List[Dict]
    justification: str

    def normalized_score(self) -> float:
        """Normalize 1-5 score to 0-1 range."""
        return (self.score - 1) / 4.0


@dataclass
class PESComponents:
    """All components of the Pedagogical Effectiveness Score."""
    # Layer 1: 8 Dimensions (Maurya et al., 2025)
    comprehension_probing: DimensionScore
    background_knowledge: DimensionScore
    guidance_level: DimensionScore
    error_feedback: DimensionScore
    encouragement: DimensionScore
    coherence: DimensionScore
    relevance: DimensionScore
    student_talk_ratio: DimensionScore

    # Layer 2: Question Depth
    question_depth: QuestionDepthScore

    # Layer 3: ICAP Engagement
    icap_engagement: ICAPScore

    # Overall summary
    overall_quality: str
    strengths: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    summary: str

    def layer1_score(self) -> float:
        """Calculate average of 8 dimension scores (normalized to 0-1)."""
        dimensions = [
            self.comprehension_probing,
            self.background_knowledge,
            self.guidance_level,
            self.error_feedback,
            self.encouragement,
            self.coherence,
            self.relevance,
            self.student_talk_ratio,
        ]
        return sum(d.normalized_score() for d in dimensions) / len(dimensions)

    def layer2_score(self) -> float:
        """Question depth score (normalized to 0-1)."""
        return self.question_depth.normalized_score()

    def layer3_score(self) -> float:
        """ICAP engagement score (normalized to 0-1)."""
        return self.icap_engagement.normalized_score()


def calculate_pes(components: PESComponents) -> float:
    """
    Calculate composite Pedagogical Effectiveness Score (PES).

    Weighted formula:
    PES = (Layer1 * 0.8 + Layer2 * 0.1 + Layer3 * 0.1) * 100

    Args:
        components: All evaluation components

    Returns:
        PES score from 0-100
    """
    layer1 = components.layer1_score()  # 0-1
    layer2 = components.layer2_score()  # 0-1
    layer3 = components.layer3_score()  # 0-1

    # Weighted average
    weighted_score = (layer1 * 0.8) + (layer2 * 0.1) + (layer3 * 0.1)

    # Scale to 0-100
    pes = weighted_score * 100

    return round(pes, 2)


def calculate_damr(
    dimension_scores: Dict[str, int],
    desired_scores: Dict[str, int]
) -> float:
    """
    Calculate Desired Annotation Match Rate (DAMR).

    DAMR = (# of dimensions matching desired score) / (total dimensions) * 100

    Args:
        dimension_scores: Actual scores for each dimension (1-5)
        desired_scores: Target scores for each dimension (1-5)

    Returns:
        DAMR percentage (0-100)
    """
    if not dimension_scores or not desired_scores:
        return 0.0

    matches = sum(
        1 for dim, score in dimension_scores.items()
        if dim in desired_scores and score == desired_scores[dim]
    )

    total = len(desired_scores)
    damr = (matches / total) * 100

    return round(damr, 2)


def get_pes_category(pes: float) -> str:
    """
    Categorize PES score into quality levels.

    Args:
        pes: PES score (0-100)

    Returns:
        Quality category string
    """
    if pes >= 85:
        return "Excellent"
    elif pes >= 70:
        return "Good"
    elif pes >= 55:
        return "Adequate"
    elif pes >= 40:
        return "Poor"
    else:
        return "Very Poor"


def dimension_breakdown(components: PESComponents) -> Dict[str, Dict]:
    """
    Get detailed breakdown of all dimension scores.

    Args:
        components: All evaluation components

    Returns:
        Dictionary with dimension names, scores, and justifications
    """
    dimensions = {
        "comprehension_probing": components.comprehension_probing,
        "background_knowledge": components.background_knowledge,
        "guidance_level": components.guidance_level,
        "error_feedback": components.error_feedback,
        "encouragement": components.encouragement,
        "coherence": components.coherence,
        "relevance": components.relevance,
        "student_talk_ratio": components.student_talk_ratio,
    }

    breakdown = {}
    for name, dim_score in dimensions.items():
        breakdown[name] = {
            "score": dim_score.score,
            "normalized": dim_score.normalized_score(),
            "justification": dim_score.justification,
            "evidence": dim_score.evidence,
        }

    return breakdown


def layer_breakdown(components: PESComponents) -> Dict[str, Dict]:
    """
    Get breakdown by evaluation layer.

    Args:
        components: All evaluation components

    Returns:
        Dictionary with layer scores and weights
    """
    return {
        "layer1_dimensions": {
            "score": components.layer1_score(),
            "weight": 0.8,
            "weighted_contribution": components.layer1_score() * 0.8 * 100,
            "dimensions": dimension_breakdown(components),
        },
        "layer2_question_depth": {
            "score": components.layer2_score(),
            "weight": 0.1,
            "weighted_contribution": components.layer2_score() * 0.1 * 100,
            "details": {
                "raw_score": components.question_depth.score,
                "question_count": components.question_depth.question_count,
                "justification": components.question_depth.justification,
            },
        },
        "layer3_icap": {
            "score": components.layer3_score(),
            "weight": 0.1,
            "weighted_contribution": components.layer3_score() * 0.1 * 100,
            "details": {
                "raw_score": components.icap_engagement.score,
                "distribution": components.icap_engagement.engagement_distribution,
                "justification": components.icap_engagement.justification,
            },
        },
    }
