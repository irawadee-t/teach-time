"""
Configuration for the EducationQ-based teaching environment.

Contains model configs, API placeholders, and hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# API Configuration
# =============================================================================

TOGETHER_API_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"

# Model IDs for Together AI
DEFAULT_STUDENT_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
DEFAULT_TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-V3"


# =============================================================================
# Dataset Paths
# =============================================================================

# Relative to project root
EDUCATIONQ_DATA_ROOT = Path("EducationQ-main/src/data")
MMLU_PRO_DATASET_NAME = "TIGER-Lab/MMLU-Pro"

# For pre-downloaded MMLU-Pro stratified subset
MMLU_PRO_STRATIFIED_PATH = (
    EDUCATIONQ_DATA_ROOT / "output" / "EduQ-Bench_Student-llama31-70b-instruct"
    / "MMLU-Pro-stratified"
)


# =============================================================================
# Dataset Category Definitions (MMLU-Pro)
# =============================================================================

# MMLU-Pro: 14 categories, ~12,032 total questions
MMLU_PRO_CATEGORIES = (
    "math", "physics", "chemistry", "law", "engineering", "other",
    "economics", "health", "psychology", "business", "biology",
    "philosophy", "computer science", "history",
)
MMLU_PRO_CATEGORY_SIZES = {
    "math": 1351, "physics": 1299, "chemistry": 1132, "law": 1101,
    "engineering": 969, "other": 924, "economics": 844, "health": 818,
    "psychology": 798, "business": 789, "biology": 717, "philosophy": 499,
    "computer science": 410, "history": 381,
}
MMLU_PRO_TOTAL = 12032


# =============================================================================
# Sampling Configuration (Experimentally Sound Design)
# =============================================================================

@dataclass
class SamplingConfig:
    """
    Configuration for balanced question sampling across categories.
    
    Design principles:
    1. **Balance**: Equal questions per category prevents subject bias
    2. **Minimum n**: Need ≥3 per category for meaningful per-category stats
    3. **Proportional scaling**: Percentage mode scales all categories equally
    4. **Reproducibility**: Fixed seed ensures identical samples
    
    Cost model (Together AI, ~15 turns avg):
    - ~30 API calls per question
    - ~2000 tokens per call
    - ~$0.01 per question (rough estimate)
    """
    
    # === Primary sampling mode (choose ONE) ===
    
    # Mode 1: Fixed per-category (recommended for balanced experiments)
    questions_per_category: Optional[int] = None
    
    # Mode 2: Percentage of dataset (proportional across categories)
    # E.g., 0.10 = 10% of each category
    percentage: Optional[float] = None
    
    # Mode 3: Total cap with balanced distribution (legacy)
    max_total_questions: Optional[int] = None
    
    # === Category selection ===
    
    # Which categories to include (None = all)
    categories: Optional[tuple[str, ...]] = None
    
    # Exclude specific categories (applied after include filter)
    exclude_categories: Optional[tuple[str, ...]] = None
    
    # === Reproducibility ===
    seed: int = 42
    
    def __post_init__(self):
        # Validate: exactly one mode must be specified
        modes = [
            self.questions_per_category is not None,
            self.percentage is not None,
            self.max_total_questions is not None,
        ]
        if sum(modes) == 0:
            # Default to questions_per_category
            object.__setattr__(self, 'questions_per_category', 3)
        elif sum(modes) > 1:
            raise ValueError(
                "Specify exactly ONE of: questions_per_category, percentage, max_total_questions"
            )
        
        if self.percentage is not None and not (0 < self.percentage <= 1.0):
            raise ValueError("percentage must be between 0 and 1")
    
    def compute_sample_sizes(
        self,
        available_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, int]:
        """
        Compute how many questions to sample from each category.
        
        Args:
            available_counts: Actual counts per category (overrides defaults)
            
        Returns:
            Dict mapping category -> sample count
        """
        # Get category sizes (default to MMLU-Pro)
        if available_counts:
            sizes = available_counts
        else:
            sizes = MMLU_PRO_CATEGORY_SIZES.copy()
        
        # Filter categories
        if self.categories:
            sizes = {k: v for k, v in sizes.items() if k in self.categories}
        if self.exclude_categories:
            sizes = {k: v for k, v in sizes.items() if k not in self.exclude_categories}
        
        if not sizes:
            raise ValueError("No categories remain after filtering")
        
        # Compute per-category sample sizes
        sample_sizes = {}
        
        if self.questions_per_category is not None:
            # Mode 1: Fixed per category
            for cat, avail in sizes.items():
                sample_sizes[cat] = min(self.questions_per_category, avail)
        
        elif self.percentage is not None:
            # Mode 2: Percentage of each category (proportional)
            for cat, avail in sizes.items():
                sample_sizes[cat] = max(1, int(avail * self.percentage))
        
        else:
            # Mode 3: Total cap, distributed equally
            n_cats = len(sizes)
            per_cat = self.max_total_questions // n_cats
            for cat, avail in sizes.items():
                sample_sizes[cat] = min(per_cat, avail)
        
        return sample_sizes
    
    def estimate_total_questions(self) -> int:
        """Estimate total questions that will be sampled."""
        sizes = self.compute_sample_sizes()
        return sum(sizes.values())
    
    def estimate_api_calls(self, avg_turns: float = 8.0) -> int:
        """Estimate total API calls."""
        total_q = self.estimate_total_questions()
        calls_per_question = 1 + (2 * avg_turns)  # 1 initial + 2*turns
        return int(total_q * calls_per_question)
    
    def estimate_cost_usd(self, avg_turns: float = 8.0) -> float:
        """Rough cost estimate (Together AI pricing)."""
        calls = self.estimate_api_calls(avg_turns)
        # ~2000 tokens/call, ~$0.40/1M tokens average
        return calls * 2000 / 1_000_000 * 0.40
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "questions_per_category": self.questions_per_category,
            "percentage": self.percentage,
            "max_total_questions": self.max_total_questions,
            "categories": list(self.categories) if self.categories else None,
            "exclude_categories": list(self.exclude_categories) if self.exclude_categories else None,
            "seed": self.seed,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        sizes = self.compute_sample_sizes()
        total = sum(sizes.values())
        api_calls = self.estimate_api_calls()
        cost = self.estimate_cost_usd()
        
        lines = [
            f"Sampling Config (MMLU-Pro):",
            f"  Total questions: {total}",
            f"  Categories: {len(sizes)}",
            f"  Per category: {dict(sizes)}",
            f"  Estimated API calls: ~{api_calls}",
            f"  Estimated cost: ~${cost:.2f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Pre-defined Sampling Tiers (Experimentally Sound)
# =============================================================================

# Tier 1: PILOT - Quick sanity check (~10 min, ~$0.60)
# - 3 questions per category × 14 categories = 42 questions
SAMPLING_PILOT = SamplingConfig(questions_per_category=3)

# Tier 2: SMALL - Development iterations (~30 min, ~$1.00)
# - 5 questions per category × 14 categories = 70 questions
SAMPLING_SMALL = SamplingConfig(questions_per_category=5)

# Tier 3: MEDIUM - Preliminary results (~2 hrs, ~$3.00)
# - 15 questions per category × 14 categories = 210 questions
SAMPLING_MEDIUM = SamplingConfig(questions_per_category=15)

# Tier 4: FULL - Publication-ready (~8 hrs, ~$6.00)
# - 30 questions per category × 14 categories = 420 questions
SAMPLING_FULL = SamplingConfig(questions_per_category=30)


# =============================================================================
# Percentage-based Configs (for proportional scaling)
# =============================================================================

def create_percentage_sampling(
    percentage: float,
    exclude_other: bool = True,
) -> SamplingConfig:
    """
    Create sampling config that takes X% of each category.
    
    Args:
        percentage: 0.0-1.0, fraction of each category to sample
        exclude_other: Whether to exclude MMLU-Pro "other" category
        
    Returns:
        SamplingConfig with percentage mode
        
    Example:
        >>> config = create_percentage_sampling(0.05)  # 5% of each category
        >>> # ~600 questions total
    """
    exclude = ("other",) if exclude_other else None
    return SamplingConfig(percentage=percentage, exclude_categories=exclude)


# Common percentage configs
SAMPLING_5_PERCENT = create_percentage_sampling(0.05)   # ~550 questions
SAMPLING_10_PERCENT = create_percentage_sampling(0.10)  # ~1100 questions
SAMPLING_25_PERCENT = create_percentage_sampling(0.25)  # ~2750 questions


# =============================================================================
# STEM-focused Configs (for science/math evaluation)
# =============================================================================

STEM_CATEGORIES = ("math", "physics", "chemistry", "biology", "computer science", "engineering")

SAMPLING_STEM_SMALL = SamplingConfig(
    questions_per_category=5,
    categories=STEM_CATEGORIES,
)

SAMPLING_STEM_MEDIUM = SamplingConfig(
    questions_per_category=15,
    categories=STEM_CATEGORIES,
)


# =============================================================================
# Teacher Configuration
# =============================================================================

@dataclass(frozen=True)
class TeacherConfig:
    """Configuration for a teacher model variant."""
    
    name: str
    model_id: str
    description: str
    temperature: float = 0.0
    max_tokens: int = 512
    
    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")


# Pre-defined teacher configurations for experiments
TEACHER_CONFIGS: list[TeacherConfig] = [
    TeacherConfig(
        name="deepseek_v3",
        model_id="deepseek-ai/DeepSeek-V3",
        description="DeepSeek V3 - state-of-the-art open model.",
    ),
    TeacherConfig(
        name="qwen_baseline",
        model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
        description="Baseline Qwen2.5-7B-Instruct-Turbo (serverless).",
    ),
    TeacherConfig(
        name="llama_70b",
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        description="Llama 3.1 70B Instruct Turbo.",
    ),
]


def get_teacher_config(name: str) -> TeacherConfig:
    """Retrieve a teacher config by name."""
    for config in TEACHER_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown teacher config: {name}. Available: {[c.name for c in TEACHER_CONFIGS]}")


# =============================================================================
# Student Configuration
# =============================================================================

@dataclass(frozen=True)
class StudentConfig:
    """Configuration for the student agent."""
    
    name: str
    model_id: str
    description: str
    temperature: float = 0.0
    max_tokens: int = 512
    recommended_response_tokens: int = 150


DEFAULT_STUDENT_CONFIG = StudentConfig(
    name="llama_70b_student",
    model_id=DEFAULT_STUDENT_MODEL_ID,
    description="Llama 3.1 70B Instruct as student. Simulates struggling student via prompting.",
    temperature=0.0,  # 0.0 for reproducibility
    max_tokens=512,   # Allow longer reasoning
    recommended_response_tokens=150,
)


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Global configuration for the teaching environment."""
    
    # Turn limits
    max_teacher_turns: int = 15
    
    # Dataset
    mmlu_pro_stratified_path: Optional[Path] = None
    
    # Sampling (use pre-defined configs or custom)
    sampling: SamplingConfig = field(default_factory=lambda: SAMPLING_PILOT)
    
    # Caching
    enable_cache: bool = True
    cache_dir: str = ".cache/environment"
    
    # Logging
    verbose: bool = False
    
    def __post_init__(self):
        if self.mmlu_pro_stratified_path is None:
            self.mmlu_pro_stratified_path = MMLU_PRO_STRATIFIED_PATH
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_teacher_turns": self.max_teacher_turns,
            "sampling": self.sampling.to_dict(),
            "enable_cache": self.enable_cache,
            "verbose": self.verbose,
        }


# =============================================================================
# Experiment Configuration (for tracking what was run)
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration for reproducibility and tracking.
    
    Use this to define a full experiment, then run it with the ExperimentRunner.
    """
    
    # Experiment metadata
    name: str
    description: str = ""
    
    # Models
    teacher_config: TeacherConfig = field(default_factory=lambda: TEACHER_CONFIGS[0])
    student_config: StudentConfig = field(default_factory=lambda: DEFAULT_STUDENT_CONFIG)
    
    # Environment
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("experiments/outputs"))
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
    
    @property
    def experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}"
    
    def to_dict(self) -> dict:
        """Convert full experiment config to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "teacher": {
                "name": self.teacher_config.name,
                "model_id": self.teacher_config.model_id,
                "description": self.teacher_config.description,
                "temperature": self.teacher_config.temperature,
                "max_tokens": self.teacher_config.max_tokens,
            },
            "student": {
                "name": self.student_config.name,
                "model_id": self.student_config.model_id,
                "description": self.student_config.description,
                "temperature": self.student_config.temperature,
                "max_tokens": self.student_config.max_tokens,
            },
            "environment": self.env_config.to_dict(),
            "output_dir": str(self.output_dir),
        }


# =============================================================================
# LLM Client Factory
# =============================================================================

@dataclass
class LLMClientConfig:
    """Configuration for the LLM client."""
    
    api_base_url: str = TOGETHER_API_BASE_URL
    api_key: Optional[str] = None  # Read from env if None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


# =============================================================================
# Quick Config Builders
# =============================================================================

def create_pilot_experiment(
    name: str,
    teacher_name: str = "qwen_baseline",
    description: str = "Pilot experiment with minimal questions",
) -> ExperimentConfig:
    """Create a minimal pilot experiment config."""
    return ExperimentConfig(
        name=name,
        description=description,
        teacher_config=get_teacher_config(teacher_name),
        env_config=EnvironmentConfig(
            max_teacher_turns=10,  # Reduced for pilot
            sampling=SAMPLING_PILOT,
        ),
    )


def create_small_experiment(
    name: str,
    teacher_name: str = "qwen_baseline",
    description: str = "Small-scale experiment",
) -> ExperimentConfig:
    """Create a small experiment config."""
    return ExperimentConfig(
        name=name,
        description=description,
        teacher_config=get_teacher_config(teacher_name),
        env_config=EnvironmentConfig(
            max_teacher_turns=15,
            sampling=SAMPLING_SMALL,
        ),
    )

