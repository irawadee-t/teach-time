"""
Experiment configuration and metadata tracking for reproducible evaluations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class ExperimentConfig:
    """
    Configuration for reproducible TutorBench experiments.

    This tracks all parameters needed to reproduce an evaluation run,
    following scientific best practices from the TutorBench paper.
    """
    model_name: str
    model_version: str  # e.g., "gpt-4-2024-01-15", "claude-opus-4-20250805"

    # Model parameters
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None

    # Evaluation parameters
    use_async: bool = True
    max_samples: Optional[int] = None  # None = use all samples

    # Metadata
    run_id: str = field(default=None)
    timestamp: str = field(default=None)
    notes: str = ""

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "use_async": self.use_async,
            "max_samples": self.max_samples,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Load from dictionary"""
        return cls(**data)

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
