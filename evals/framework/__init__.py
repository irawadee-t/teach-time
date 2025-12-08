"""
Framework for running statistically rigorous TutorBench evaluations.

This module provides enhanced evaluation capabilities on top of the core
TutorBench implementation, including:
- Experiment configuration and tracking
- Multi-run evaluations with confidence intervals
- Comprehensive reporting with all breakdowns
- Leaderboard management
"""

from .experiment import ExperimentConfig
from .statistics import compute_confidence_interval, aggregate_multiple_runs, compute_statistical_significance
from .reporting import generate_full_report, print_summary, save_leaderboard_entry
from .run_evaluation import run_evaluation
from .comparison import (
    compute_comparison_stats,
    print_comparison_table,
    save_comparison_report,
)

__all__ = [
    'ExperimentConfig',
    'compute_confidence_interval',
    'aggregate_multiple_runs',
    'compute_statistical_significance',
    'generate_full_report',
    'print_summary',
    'save_leaderboard_entry',
    'run_evaluation',
    'compute_comparison_stats',
    'print_comparison_table',
    'save_comparison_report',
]
