"""
LLM-as-a-Judge Pedagogical Evaluation System

A comprehensive evaluation framework for assessing tutoring conversations using:
- Layer 1: 8-dimension tutor response quality (Maurya et al., 2025)
- Layer 2: Question depth analysis (recall/procedural/conceptual/metacognitive)
- Layer 3: ICAP student engagement classification (Chi & Wylie, 2014)

Produces a composite Pedagogical Effectiveness Score (PES) from 0-100.
"""

from .evaluator import PedagogicalEvaluator
from .metrics import calculate_pes, PESComponents
from .report import generate_report, generate_summary_report

__version__ = "1.0.0"

__all__ = [
    "PedagogicalEvaluator",
    "calculate_pes",
    "PESComponents",
    "generate_report",
    "generate_summary_report",
]
