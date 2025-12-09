"""
Ablation study configurations for comparing different teaching approaches.

Experiments test:
1. Model capability (DeepSeek V3 vs Qwen)
2. Fine-tuning effect (base vs fine-tuned)
3. ReAct framework effect (with vs without)

All experiments use math-only MMLU-Pro questions.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .config import (
    TeacherConfig,
    StudentConfig,
    EnvironmentConfig,
    ExperimentConfig,
    SamplingConfig,
    DEFAULT_STUDENT_CONFIG,
)


# =============================================================================
# Extended Teacher Config with ReAct Toggle
# =============================================================================

@dataclass(frozen=True)
class AblationTeacherConfig:
    """
    Extended teacher config for ablation studies.
    
    Adds:
    - use_react: Whether to use ReAct framework
    - is_finetuned: Metadata flag for tracking
    - base_model: If fine-tuned, what was the base model
    """
    name: str
    model_id: str
    description: str
    use_react: bool = True
    is_finetuned: bool = False
    base_model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    
    def to_teacher_config(self) -> TeacherConfig:
        """Convert to standard TeacherConfig for compatibility."""
        return TeacherConfig(
            name=self.name,
            model_id=self.model_id,
            description=self.description,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "description": self.description,
            "use_react": self.use_react,
            "is_finetuned": self.is_finetuned,
            "base_model": self.base_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# =============================================================================
# Pre-defined Ablation Variants
# =============================================================================

# =============================================================================
# Strong Baseline (for reference)
# =============================================================================

DEEPSEEK_V3_PLAIN = AblationTeacherConfig(
    name="deepseek_v3_plain",
    model_id="deepseek-ai/DeepSeek-V3",
    description="DeepSeek V3 - strong baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

DEEPSEEK_V3_REACT = AblationTeacherConfig(
    name="deepseek_v3_react",
    model_id="deepseek-ai/DeepSeek-V3",
    description="DeepSeek V3 - strong baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)

# =============================================================================
# Small Models (3B) - Good for fine-tuning experiments
# =============================================================================

QWEN_3B_PLAIN = AblationTeacherConfig(
    name="qwen_3b_plain",
    model_id="Qwen/Qwen2.5-3B-Instruct-Turbo",
    description="Qwen 2.5 3B - small baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

QWEN_3B_REACT = AblationTeacherConfig(
    name="qwen_3b_react",
    model_id="Qwen/Qwen2.5-3B-Instruct-Turbo",
    description="Qwen 2.5 3B - small baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)

LLAMA_3B_PLAIN = AblationTeacherConfig(
    name="llama_3b_plain",
    model_id="meta-llama/Llama-3.2-3B-Instruct-Turbo",
    description="Llama 3.2 3B - small baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

LLAMA_3B_REACT = AblationTeacherConfig(
    name="llama_3b_react",
    model_id="meta-llama/Llama-3.2-3B-Instruct-Turbo",
    description="Llama 3.2 3B - small baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)

# =============================================================================
# Medium Models (7-8B) - Balance of capability and fine-tuning cost
# =============================================================================

QWEN_7B_PLAIN = AblationTeacherConfig(
    name="qwen_7b_plain",
    model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
    description="Qwen 2.5 7B - medium baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

QWEN_7B_REACT = AblationTeacherConfig(
    name="qwen_7b_react",
    model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
    description="Qwen 2.5 7B - medium baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)

LLAMA_8B_PLAIN = AblationTeacherConfig(
    name="llama_8b_plain",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    description="Llama 3.1 8B - medium baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

LLAMA_8B_REACT = AblationTeacherConfig(
    name="llama_8b_react",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    description="Llama 3.1 8B - medium baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)

# =============================================================================
# Fine-tuned Models (Placeholders - replace with your endpoints)
# =============================================================================

# 3B Fine-tuned
QWEN_3B_FINETUNED_PLAIN = AblationTeacherConfig(
    name="qwen_3b_finetuned_plain",
    model_id="YOUR_3B_FINETUNED_MODEL_ID",  # TODO: Replace with actual endpoint
    description="Qwen 2.5 3B fine-tuned for teaching, no ReAct",
    use_react=False,
    is_finetuned=True,
    base_model="Qwen/Qwen2.5-3B-Instruct-Turbo",
)

QWEN_3B_FINETUNED_REACT = AblationTeacherConfig(
    name="qwen_3b_finetuned_react",
    model_id="YOUR_3B_FINETUNED_MODEL_ID",  # TODO: Replace with actual endpoint
    description="Qwen 2.5 3B fine-tuned for teaching, with ReAct",
    use_react=True,
    is_finetuned=True,
    base_model="Qwen/Qwen2.5-3B-Instruct-Turbo",
)

# 7B Fine-tuned
QWEN_7B_FINETUNED_PLAIN = AblationTeacherConfig(
    name="qwen_7b_finetuned_plain",
    model_id="YOUR_7B_FINETUNED_MODEL_ID",  # TODO: Replace with actual endpoint
    description="Qwen 2.5 7B fine-tuned for teaching, no ReAct",
    use_react=False,
    is_finetuned=True,
    base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
)

QWEN_7B_FINETUNED_REACT = AblationTeacherConfig(
    name="qwen_7b_finetuned_react",
    model_id="YOUR_7B_FINETUNED_MODEL_ID",  # TODO: Replace with actual endpoint
    description="Qwen 2.5 7B fine-tuned for teaching, with ReAct",
    use_react=True,
    is_finetuned=True,
    base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
)

# =============================================================================
# Large Models (70B) - For comparison only
# =============================================================================

LLAMA_70B_PLAIN = AblationTeacherConfig(
    name="llama_70b_plain",
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    description="Llama 3.1 70B - large baseline, no ReAct",
    use_react=False,
    is_finetuned=False,
)

LLAMA_70B_REACT = AblationTeacherConfig(
    name="llama_70b_react",
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    description="Llama 3.1 70B - large baseline, with ReAct",
    use_react=True,
    is_finetuned=False,
)


# =============================================================================
# Ablation Study Definitions
# =============================================================================

# ReAct effect on 3B models (cheap, fast)
ABLATION_REACT_3B = [
    QWEN_3B_PLAIN,
    QWEN_3B_REACT,
    LLAMA_3B_PLAIN,
    LLAMA_3B_REACT,
]

# ReAct effect on 7-8B models
ABLATION_REACT_7B = [
    QWEN_7B_PLAIN,
    QWEN_7B_REACT,
    LLAMA_8B_PLAIN,
    LLAMA_8B_REACT,
]

# Model size comparison (plain mode)
ABLATION_SIZE_PLAIN = [
    QWEN_3B_PLAIN,
    QWEN_7B_PLAIN,
    LLAMA_3B_PLAIN,
    LLAMA_8B_PLAIN,
]

# Model size comparison (ReAct mode)
ABLATION_SIZE_REACT = [
    QWEN_3B_REACT,
    QWEN_7B_REACT,
    LLAMA_3B_REACT,
    LLAMA_8B_REACT,
]

# Primary ablation: Test ReAct effect across model sizes
ABLATION_REACT_EFFECT = [
    QWEN_3B_PLAIN,
    QWEN_3B_REACT,
    QWEN_7B_PLAIN,
    QWEN_7B_REACT,
]

# Fine-tuning ablation (7B) - requires fine-tuned model endpoints
ABLATION_FINETUNING_7B = [
    QWEN_7B_PLAIN,
    QWEN_7B_FINETUNED_PLAIN,
    QWEN_7B_REACT,
    QWEN_7B_FINETUNED_REACT,
]

# Fine-tuning ablation (3B) - requires fine-tuned model endpoints
ABLATION_FINETUNING_3B = [
    QWEN_3B_PLAIN,
    QWEN_3B_FINETUNED_PLAIN,
    QWEN_3B_REACT,
    QWEN_3B_FINETUNED_REACT,
]

# Full ablation: All base models with ReAct toggle
ABLATION_FULL_BASE = [
    QWEN_3B_PLAIN,
    QWEN_3B_REACT,
    QWEN_7B_PLAIN,
    QWEN_7B_REACT,
    LLAMA_3B_PLAIN,
    LLAMA_3B_REACT,
    LLAMA_8B_PLAIN,
    LLAMA_8B_REACT,
]


# =============================================================================
# Ablation Experiment Config
# =============================================================================

@dataclass
class AblationExperimentConfig:
    """
    Configuration for an ablation experiment run.
    """
    name: str
    description: str
    teacher_configs: list[AblationTeacherConfig]
    
    # Student config (same for all - controlled variable)
    student_config: StudentConfig = field(default_factory=lambda: DEFAULT_STUDENT_CONFIG)
    
    # Environment settings
    max_teacher_turns: int = 10
    questions_per_category: int = 5
    categories: tuple[str, ...] = ("math",)
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("experiments/ablation"))
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "teacher_variants": [t.to_dict() for t in self.teacher_configs],
            "student": {
                "name": self.student_config.name,
                "model_id": self.student_config.model_id,
            },
            "max_teacher_turns": self.max_teacher_turns,
            "questions_per_category": self.questions_per_category,
            "categories": list(self.categories),
        }


# =============================================================================
# Pre-defined Ablation Experiments
# =============================================================================

def create_react_ablation(
    questions: int = 5,
    max_turns: int = 10,
) -> AblationExperimentConfig:
    """
    Create ablation experiment testing ReAct framework effect.
    
    Compares:
    - DeepSeek V3 (plain vs ReAct)
    - Qwen 7B (plain vs ReAct)
    """
    return AblationExperimentConfig(
        name="react_ablation",
        description="Testing effect of ReAct framework on different model sizes",
        teacher_configs=ABLATION_REACT_EFFECT,
        max_teacher_turns=max_turns,
        questions_per_category=questions,
        categories=("math",),
    )


def create_finetuning_ablation(
    finetuned_model_id: str,
    questions: int = 5,
    max_turns: int = 10,
) -> AblationExperimentConfig:
    """
    Create ablation experiment testing fine-tuning effect.
    
    Args:
        finetuned_model_id: Together AI endpoint for your fine-tuned model
        
    Compares:
    - Qwen 7B base (plain vs ReAct)
    - Qwen 7B fine-tuned (plain vs ReAct)
    """
    # Create configs with actual fine-tuned model ID
    finetuned_plain = AblationTeacherConfig(
        name="qwen_7b_finetuned_plain",
        model_id=finetuned_model_id,
        description="Qwen 2.5 7B fine-tuned for teaching, no ReAct",
        use_react=False,
        is_finetuned=True,
        base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    )
    
    finetuned_react = AblationTeacherConfig(
        name="qwen_7b_finetuned_react",
        model_id=finetuned_model_id,
        description="Qwen 2.5 7B fine-tuned for teaching, with ReAct",
        use_react=True,
        is_finetuned=True,
        base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    )
    
    return AblationExperimentConfig(
        name="finetuning_ablation",
        description="Testing effect of fine-tuning on Qwen 7B",
        teacher_configs=[
            QWEN_7B_PLAIN,
            finetuned_plain,
            QWEN_7B_REACT,
            finetuned_react,
        ],
        max_teacher_turns=max_turns,
        questions_per_category=questions,
        categories=("math",),
    )


def create_custom_ablation(
    name: str,
    teacher_configs: list[AblationTeacherConfig],
    questions: int = 5,
    max_turns: int = 10,
    categories: tuple[str, ...] = ("math",),
) -> AblationExperimentConfig:
    """
    Create custom ablation experiment with specified teacher variants.
    """
    return AblationExperimentConfig(
        name=name,
        description=f"Custom ablation with {len(teacher_configs)} variants",
        teacher_configs=teacher_configs,
        max_teacher_turns=max_turns,
        questions_per_category=questions,
        categories=categories,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def get_available_configs() -> dict[str, AblationTeacherConfig]:
    """Get all pre-defined teacher configs by name."""
    return {
        # Strong baseline
        "deepseek_v3_plain": DEEPSEEK_V3_PLAIN,
        "deepseek_v3_react": DEEPSEEK_V3_REACT,
        # 3B models
        "qwen_3b_plain": QWEN_3B_PLAIN,
        "qwen_3b_react": QWEN_3B_REACT,
        "llama_3b_plain": LLAMA_3B_PLAIN,
        "llama_3b_react": LLAMA_3B_REACT,
        # 7-8B models
        "qwen_7b_plain": QWEN_7B_PLAIN,
        "qwen_7b_react": QWEN_7B_REACT,
        "llama_8b_plain": LLAMA_8B_PLAIN,
        "llama_8b_react": LLAMA_8B_REACT,
        # Large models
        "llama_70b_plain": LLAMA_70B_PLAIN,
        "llama_70b_react": LLAMA_70B_REACT,
    }


def list_available_configs():
    """Print all available pre-defined configs."""
    configs = get_available_configs()
    print("Available Teacher Configurations:")
    print("-" * 60)
    for name, config in configs.items():
        react_str = "ReAct" if config.use_react else "Plain"
        print(f"  {name:25} | {react_str:6} | {config.model_id}")

