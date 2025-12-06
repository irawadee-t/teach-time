"""
Configuration loading and validation utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    default_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.7
    max_tokens: int = 300
    enable_cache: bool = True
    cache_dir: str = ".cache"


class EnvConfig(BaseModel):
    """Configuration for teaching environment."""
    max_turns: int = 10
    enable_quizzes: bool = True
    summary_window: int = 5
    default_domain: str = "mathematics"


class ScoringWeights(BaseModel):
    """Weights for scoring components."""
    pedagogy: float = 0.4
    learning: float = 0.4
    quality: float = 0.2

    @validator('*')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError("Weights must be non-negative")
        return v


class ScoringConfig(BaseModel):
    """Configuration for AppBench-style scoring."""
    weights: ScoringWeights = ScoringWeights()

    # Pedagogy targets
    student_talk_ratio_range: tuple = (0.5, 0.8)
    target_questions_per_session: int = 5
    background_asked_by_turn: int = 3
    target_understanding_checks: int = 3

    # Learning thresholds
    min_learning_gain: float = 0.1
    good_learning_gain: float = 0.3

    # Quality assessment
    use_llm_judge: bool = True
    judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment."""
    name: str
    domain: str
    students: list[str] = ["struggling", "confident_mistaken"]
    agents: list[str] = ["baseline_cot", "metric_cot", "react_teacher"]
    num_episodes_per_condition: int = 100
    max_turns: int = 10
    enable_quizzes: bool = True
    scoring_focus: str = "full"  # "pedagogy_only", "learning_only", or "full"
    random_seed: int = 42


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Dictionary of configuration values
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Load model configuration."""
    if config_path is None:
        config_path = CONFIGS_DIR / "model.yaml"

    if config_path.exists():
        config_dict = load_yaml_config(config_path)
        return ModelConfig(**config_dict)
    else:
        # Return defaults
        return ModelConfig()


def load_env_config(config_path: Optional[Path] = None) -> EnvConfig:
    """Load environment configuration."""
    if config_path is None:
        config_path = CONFIGS_DIR / "env.yaml"

    if config_path.exists():
        config_dict = load_yaml_config(config_path)
        return EnvConfig(**config_dict)
    else:
        return EnvConfig()


def load_scoring_config(config_path: Optional[Path] = None) -> ScoringConfig:
    """Load scoring configuration."""
    if config_path is None:
        config_path = CONFIGS_DIR / "scoring.yaml"

    if config_path.exists():
        config_dict = load_yaml_config(config_path)
        return ScoringConfig(**config_dict)
    else:
        return ScoringConfig()


def load_experiment_config(experiment_name: str) -> ExperimentConfig:
    """
    Load experiment configuration by name.

    Args:
        experiment_name: Name of experiment (e.g., "exp1_metrics_match")

    Returns:
        ExperimentConfig object
    """
    config_path = CONFIGS_DIR / "experiments" / f"{experiment_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    config_dict = load_yaml_config(config_path)
    return ExperimentConfig(**config_dict)


def save_config_snapshot(config: BaseModel, output_path: Path):
    """
    Save configuration snapshot for reproducibility.

    Args:
        config: Pydantic config model
        output_path: Where to save the snapshot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
