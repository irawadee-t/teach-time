"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np
from typing import Optional


def set_global_seed(seed: int):
    """
    Set global random seed for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Global random seed set to: {seed}")


def get_episode_seed(experiment_seed: int, episode_idx: int) -> int:
    """
    Generate deterministic seed for an episode.

    Args:
        experiment_seed: Base seed for the experiment
        episode_idx: Episode index

    Returns:
        Deterministic seed for this episode
    """
    # Use a simple but deterministic combination
    return experiment_seed + episode_idx * 1000


class SeededRandom:
    """
    Context manager for temporarily setting a random seed.

    Example:
        with SeededRandom(42):
            # Code here uses seed 42
            value = random.random()
        # Original random state restored
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.random_state = None
        self.np_state = None

    def __enter__(self):
        # Save current states
        self.random_state = random.getstate()
        self.np_state = np.random.get_state()

        # Set new seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        random.setstate(self.random_state)
        np.random.set_state(self.np_state)
