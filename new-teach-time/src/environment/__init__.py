"""
Teaching environment for Socratic tutoring experiments.

Clean architecture:
- StudentAgent / TeacherAgent: LLM-based agents (no overrides)
- ExternalJudge: DeepSeek v3 for answer evaluation (single source of truth)
- InteractionLoop: Multi-turn dialogue runner (pure logging)
- ExperimentRunner: Batch experiment execution
"""

# Config
from .config import (
    TeacherConfig,
    StudentConfig,
    EnvironmentConfig,
    ExperimentConfig,
    SamplingConfig,
    TEACHER_CONFIGS,
    DEFAULT_STUDENT_CONFIG,
    SAMPLING_PILOT,
    SAMPLING_SMALL,
    SAMPLING_MEDIUM,
    STEM_CATEGORIES,
)

# Data
from .datasets import Question, load_mmlu_pro_stratified, load_mmlu_pro_from_huggingface

# Agents
from .agents import StudentAgent, TeacherAgent, StudentTurn, TeacherTurn

# Judge (external evaluation)
from .judge import ExternalJudge, JudgeVerdict, extract_final_answer

# Runner
from .runner import (
    InteractionLoop,
    InteractionSession,
    ExperimentRunner,
    ExperimentOutput,
    sample_questions,
    create_interaction_loop,
)

__all__ = [
    # Config
    "TeacherConfig",
    "StudentConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "SamplingConfig",
    "TEACHER_CONFIGS",
    "DEFAULT_STUDENT_CONFIG",
    "SAMPLING_PILOT",
    "SAMPLING_SMALL",
    "SAMPLING_MEDIUM",
    "STEM_CATEGORIES",
    # Data
    "Question",
    "load_mmlu_pro_stratified",
    "load_mmlu_pro_from_huggingface",
    # Agents
    "StudentAgent",
    "TeacherAgent",
    "StudentTurn",
    "TeacherTurn",
    # Judge
    "ExternalJudge",
    "JudgeVerdict",
    "extract_final_answer",
    # Runner
    "InteractionLoop",
    "InteractionSession",
    "ExperimentRunner",
    "ExperimentOutput",
    "sample_questions",
    "create_interaction_loop",
]
