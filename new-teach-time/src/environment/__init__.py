"""
Teaching environment for Socratic tutoring experiments.

Clean architecture:
- StudentAgent / TeacherAgent: LLM-based agents (no overrides)
- PlainTeacherAgent: Teacher without ReAct framework (for ablation)
- ExternalJudge: DeepSeek v3 for answer evaluation (single source of truth)
- InteractionLoop: Multi-turn dialogue runner (pure logging)
- ExperimentRunner: Batch experiment execution
- AblationRunner: Run multiple teacher variants for comparison
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

# Ablation Configs
from .ablation_configs import (
    AblationTeacherConfig,
    AblationExperimentConfig,
    # Strong baseline
    DEEPSEEK_V3_PLAIN,
    DEEPSEEK_V3_REACT,
    # 3B models
    QWEN_3B_PLAIN,
    QWEN_3B_REACT,
    LLAMA_3B_PLAIN,
    LLAMA_3B_REACT,
    # 7-8B models
    QWEN_7B_PLAIN,
    QWEN_7B_REACT,
    LLAMA_8B_PLAIN,
    LLAMA_8B_REACT,
    # Large models
    LLAMA_70B_PLAIN,
    LLAMA_70B_REACT,
    # Pre-defined ablations
    ABLATION_REACT_3B,
    ABLATION_REACT_7B,
    ABLATION_REACT_EFFECT,
    ABLATION_FULL_BASE,
    # Helpers
    create_react_ablation,
    create_finetuning_ablation,
    create_custom_ablation,
    get_available_configs,
)

# Data
from .datasets import Question, load_mmlu_pro_stratified, load_mmlu_pro_from_huggingface

# Agents
from .agents import StudentAgent, TeacherAgent, StudentTurn, TeacherTurn
from .agents import PlainTeacherAgent, PlainTeacherTurn, create_teacher

# Judge (external evaluation)
from .judge import ExternalJudge, JudgeVerdict, extract_final_answer

# Runner
from .runner import (
    InteractionLoop,
    InteractionSession,
    ExperimentRunner,
    ExperimentOutput,
    AblationRunner,
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
    # Ablation
    "AblationTeacherConfig",
    "AblationExperimentConfig",
    "DEEPSEEK_V3_PLAIN",
    "DEEPSEEK_V3_REACT",
    "QWEN_3B_PLAIN",
    "QWEN_3B_REACT",
    "LLAMA_3B_PLAIN",
    "LLAMA_3B_REACT",
    "QWEN_7B_PLAIN",
    "QWEN_7B_REACT",
    "LLAMA_8B_PLAIN",
    "LLAMA_8B_REACT",
    "LLAMA_70B_PLAIN",
    "LLAMA_70B_REACT",
    "ABLATION_REACT_3B",
    "ABLATION_REACT_7B",
    "ABLATION_REACT_EFFECT",
    "ABLATION_FULL_BASE",
    "create_react_ablation",
    "create_finetuning_ablation",
    "create_custom_ablation",
    "get_available_configs",
    # Data
    "Question",
    "load_mmlu_pro_stratified",
    "load_mmlu_pro_from_huggingface",
    # Agents
    "StudentAgent",
    "TeacherAgent",
    "StudentTurn",
    "TeacherTurn",
    "PlainTeacherAgent",
    "PlainTeacherTurn",
    "create_teacher",
    # Judge
    "ExternalJudge",
    "JudgeVerdict",
    "extract_final_answer",
    # Runner
    "InteractionLoop",
    "InteractionSession",
    "ExperimentRunner",
    "ExperimentOutput",
    "AblationRunner",
    "sample_questions",
    "create_interaction_loop",
]
