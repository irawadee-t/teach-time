"""
Structured logging utilities for TeachTime experiments.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class TrajectoryLogger:
    """
    Logger for tutoring episode trajectories.

    Saves trajectories in JSONL format (one JSON object per line).
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save trajectory files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_episode(
        self,
        episode_data: Dict[str, Any],
        experiment_id: str,
        episode_id: str,
    ):
        """
        Log a complete episode to JSONL file.

        Args:
            episode_data: Dictionary containing episode information
            experiment_id: Experiment identifier
            episode_id: Episode identifier
        """
        # Ensure episode_data has required metadata
        episode_data["experiment_id"] = experiment_id
        episode_data["episode_id"] = episode_id
        episode_data["logged_at"] = datetime.now().isoformat()

        # Write to JSONL file
        output_file = self.output_dir / f"{experiment_id}_trajectories.jsonl"

        with open(output_file, 'a') as f:
            f.write(json.dumps(episode_data) + '\n')

    def log_batch_summary(
        self,
        summary_data: Dict[str, Any],
        experiment_id: str,
    ):
        """
        Log summary statistics for a batch of episodes.

        Args:
            summary_data: Dictionary containing summary statistics
            experiment_id: Experiment identifier
        """
        summary_data["experiment_id"] = experiment_id
        summary_data["logged_at"] = datetime.now().isoformat()

        output_file = self.output_dir / f"{experiment_id}_summary.json"

        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2)


class MetricsAccumulator:
    """
    Accumulates metrics across multiple episodes for analysis.
    """

    def __init__(self):
        self.episodes = []

    def add_episode(
        self,
        episode_id: str,
        agent_type: str,
        student_persona: str,
        task_id: str,
        metrics: Dict[str, Any],
        scores: Dict[str, float],
        pre_quiz: float,
        post_quiz: float,
        learning_gain: float,
    ):
        """Add episode results."""
        self.episodes.append({
            "episode_id": episode_id,
            "agent_type": agent_type,
            "student_persona": student_persona,
            "task_id": task_id,
            "metrics": metrics,
            "scores": scores,
            "pre_quiz_score": pre_quiz,
            "post_quiz_score": post_quiz,
            "learning_gain": learning_gain,
        })

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across all episodes."""
        import numpy as np

        if not self.episodes:
            return {}

        # Group by agent type
        agent_stats = {}
        for agent_type in set(ep["agent_type"] for ep in self.episodes):
            agent_eps = [ep for ep in self.episodes if ep["agent_type"] == agent_type]

            learning_gains = [ep["learning_gain"] for ep in agent_eps]
            pedagogy_scores = [ep["scores"].get("pedagogy_score", 0) for ep in agent_eps]

            agent_stats[agent_type] = {
                "num_episodes": len(agent_eps),
                "learning_gain": {
                    "mean": np.mean(learning_gains),
                    "std": np.std(learning_gains),
                    "median": np.median(learning_gains),
                },
                "pedagogy_score": {
                    "mean": np.mean(pedagogy_scores),
                    "std": np.std(pedagogy_scores),
                },
            }

        return agent_stats

    def save_to_csv(self, output_path: Path):
        """Save accumulated episodes to CSV for analysis."""
        import pandas as pd

        if not self.episodes:
            return

        # Flatten nested dictionaries for CSV
        rows = []
        for ep in self.episodes:
            row = {
                "episode_id": ep["episode_id"],
                "agent_type": ep["agent_type"],
                "student_persona": ep["student_persona"],
                "task_id": ep["task_id"],
                "pre_quiz_score": ep["pre_quiz_score"],
                "post_quiz_score": ep["post_quiz_score"],
                "learning_gain": ep["learning_gain"],
            }

            # Add metrics
            for metric_name, metric_value in ep["metrics"].items():
                row[f"metric_{metric_name}"] = metric_value

            # Add scores
            for score_name, score_value in ep["scores"].items():
                row[f"score_{score_name}"] = score_value

            rows.append(row)

        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


def format_episode_for_logging(
    episode_id: str,
    agent_info: Dict,
    task_info: Dict,
    student_info: Dict,
    turns: List[Dict],
    final_metrics: Dict,
    scores: Dict,
    pre_quiz_score: float,
    post_quiz_score: float,
    learning_gain: float,
) -> Dict[str, Any]:
    """
    Format episode data for structured logging.

    Args:
        episode_id: Episode identifier
        agent_info: Agent metadata
        task_info: Task metadata
        student_info: Student metadata
        turns: List of dialogue turns
        final_metrics: Final teaching metrics
        scores: Computed scores (pedagogy, learning, quality)
        pre_quiz_score: Pre-test score
        post_quiz_score: Post-test score
        learning_gain: Normalized learning gain

    Returns:
        Formatted dictionary ready for JSON serialization
    """
    return {
        "episode_id": episode_id,
        "timestamp": datetime.now().isoformat(),
        "agent": agent_info,
        "task": task_info,
        "student": student_info,
        "turns": turns,
        "final_metrics": final_metrics,
        "scores": scores,
        "quiz_results": {
            "pre_quiz_score": pre_quiz_score,
            "post_quiz_score": post_quiz_score,
            "learning_gain": learning_gain,
        },
    }


def create_experiment_directory(experiment_id: str, base_dir: Path = None) -> Path:
    """
    Create directory structure for an experiment.

    Args:
        experiment_id: Experiment identifier
        base_dir: Base directory (defaults to results/raw/)

    Returns:
        Path to experiment directory
    """
    if base_dir is None:
        from .config import PROJECT_ROOT
        base_dir = PROJECT_ROOT / "results" / "raw"

    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir
