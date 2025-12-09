"""
LLM-as-a-Judge Pedagogical Evaluation System

A comprehensive evaluation framework for assessing tutoring conversations using:
- Layer 1: 8-dimension tutor response quality (Maurya et al., 2025)
- Layer 2: Question depth analysis (recall/procedural/conceptual/metacognitive)
- Layer 3: ICAP student engagement classification (Chi & Wylie, 2014)

Produces a composite Pedagogical Effectiveness Score (PES) from 0-100.

Adapted for new-teach-time architecture.
"""

from .evaluator import PedagogicalEvaluator, JUDGE_MODEL_ID
from .metrics import calculate_pes, get_pes_category, PESComponents
from .report import generate_report, generate_summary_report, save_evaluation_results

__version__ = "2.0.0"

__all__ = [
    "PedagogicalEvaluator",
    "JUDGE_MODEL_ID",
    "calculate_pes",
    "get_pes_category",
    "PESComponents",
    "generate_report",
    "generate_summary_report",
    "save_evaluation_results",
]
