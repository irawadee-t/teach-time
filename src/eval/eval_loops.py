"""
Evaluation loops for running tutoring episodes and computing scores.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from ..env.teaching_env import TeachingEnv, Observation, EpisodeInfo
from ..env.tasks.base import TaskSpec
from ..agents.base_agent import BaseTutorAgent
from .scoring_config import ScoringEngine, EpisodeScores
from ..utils.logging_utils import format_episode_for_logging, TrajectoryLogger, MetricsAccumulator
from ..utils.random_utils import get_episode_seed


def run_single_episode(
    env: TeachingEnv,
    agent: BaseTutorAgent,
    task: TaskSpec,
    student_persona: str,
    episode_seed: int,
    verbose: bool = False,
) -> Tuple[EpisodeInfo, Dict]:
    """
    Run a single tutoring episode.

    Args:
        env: TeachingEnv instance
        agent: Tutor agent
        task: Task specification
        student_persona: Student persona type
        episode_seed: Random seed for this episode
        verbose: Whether to print progress

    Returns:
        Tuple of (EpisodeInfo, episode_metadata)
    """
    # Reset agent and environment
    agent.reset(task)
    obs = env.reset(task=task, student_persona=student_persona)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode: {task.task_id} | Student: {student_persona}")
        print(f"{'='*60}")

    # Track dialogue turns
    turns = []
    done = False
    episode_info = None

    while not done:
        # Agent chooses action
        action = agent.act(obs)

        if verbose:
            print(f"\n[Turn {obs.step_index}]")
            print(f"Tutor ({action.action_type.value}): {action.content[:100]}...")

        # Record turn
        turns.append({
            "turn_index": obs.step_index,
            "speaker": "tutor",
            "action_type": action.action_type.value,
            "utterance": action.content,
            "thought": action.metadata.get("thought", "") if action.metadata else "",
            "metrics": obs.metrics.copy(),
        })

        # Step environment
        obs, reward, done, info = env.step(action)

        if not done:
            if verbose:
                print(f"Student: {obs.student_utterance[:100]}...")

            # Record student turn
            turns.append({
                "turn_index": obs.step_index,
                "speaker": "student",
                "action_type": None,
                "utterance": obs.student_utterance,
                "thought": None,
                "metrics": obs.metrics.copy(),
            })

            # Update agent dialogue history
            agent.record_turn("student", obs.student_utterance)

    # Episode completed
    episode_info = info

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode completed!")
        print(f"Pre-quiz: {episode_info.pre_quiz_score:.2f}")
        print(f"Post-quiz: {episode_info.post_quiz_score:.2f}")
        print(f"Learning gain: {episode_info.learning_gain:.2f}")
        print(f"{'='*60}\n")

    # Compile episode metadata
    episode_metadata = {
        "task": {
            "task_id": task.task_id,
            "topic": task.topic,
            "difficulty": task.difficulty,
        },
        "student": {
            "persona": student_persona,
            "final_knowledge": episode_info.final_knowledge_state,
        },
        "agent": agent.get_agent_info(),
        "turns": turns,
        "seed": episode_seed,
    }

    return episode_info, episode_metadata


def score_episode(
    episode_info: EpisodeInfo,
    episode_metadata: Dict,
    scoring_engine: ScoringEngine,
) -> EpisodeScores:
    """
    Score a completed episode.

    Args:
        episode_info: Episode information from environment
        episode_metadata: Metadata from run_single_episode
        scoring_engine: Scoring engine instance

    Returns:
        EpisodeScores object
    """
    # Extract final metrics from last turn
    final_metrics = episode_metadata["turns"][-1]["metrics"]

    # Score the episode
    scores = scoring_engine.score_episode(
        final_metrics=final_metrics,
        pre_quiz_score=episode_info.pre_quiz_score,
        post_quiz_score=episode_info.post_quiz_score,
        learning_gain=episode_info.learning_gain,
        final_knowledge_state=episode_info.final_knowledge_state,
        quality_scores=None,  # TODO: Add LLM-as-judge if needed
    )

    return scores


def run_experiment_batch(
    env: TeachingEnv,
    agent: BaseTutorAgent,
    tasks: List[TaskSpec],
    student_personas: List[str],
    num_episodes_per_condition: int,
    experiment_id: str,
    base_seed: int = 42,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> MetricsAccumulator:
    """
    Run a batch of episodes for an experiment.

    Args:
        env: TeachingEnv instance
        agent: Tutor agent
        tasks: List of tasks to sample from
        student_personas: List of student personas to use
        num_episodes_per_condition: Number of episodes per task-persona pair
        experiment_id: Experiment identifier
        base_seed: Base random seed
        output_dir: Directory for logging (defaults to results/raw/)
        verbose: Whether to print progress

    Returns:
        MetricsAccumulator with aggregated results
    """
    # Setup logging
    if output_dir is None:
        from ..utils.logging_utils import create_experiment_directory
        output_dir = create_experiment_directory(experiment_id)

    logger = TrajectoryLogger(output_dir)
    accumulator = MetricsAccumulator()
    scoring_engine = ScoringEngine()

    # Generate all conditions
    conditions = []
    for task in tasks:
        for persona in student_personas:
            for rep in range(num_episodes_per_condition):
                conditions.append((task, persona, rep))

    # Run episodes with progress bar
    print(f"\nRunning experiment: {experiment_id}")
    print(f"Agent: {agent.agent_id}")
    print(f"Total episodes: {len(conditions)}\n")

    for idx, (task, persona, rep) in enumerate(tqdm(conditions, desc="Episodes")):
        episode_id = f"{experiment_id}_{agent.agent_id}_{task.task_id}_{persona}_{rep:03d}"
        episode_seed = get_episode_seed(base_seed, idx)

        # Run episode
        episode_info, episode_metadata = run_single_episode(
            env=env,
            agent=agent,
            task=task,
            student_persona=persona,
            episode_seed=episode_seed,
            verbose=verbose,
        )

        # Score episode
        scores = score_episode(episode_info, episode_metadata, scoring_engine)

        # Log episode
        episode_data = format_episode_for_logging(
            episode_id=episode_id,
            agent_info=episode_metadata["agent"],
            task_info=episode_metadata["task"],
            student_info=episode_metadata["student"],
            turns=episode_metadata["turns"],
            final_metrics=episode_metadata["turns"][-1]["metrics"],
            scores=scores.to_dict(),
            pre_quiz_score=episode_info.pre_quiz_score,
            post_quiz_score=episode_info.post_quiz_score,
            learning_gain=episode_info.learning_gain,
        )
        logger.log_episode(episode_data, experiment_id, episode_id)

        # Accumulate metrics
        accumulator.add_episode(
            episode_id=episode_id,
            agent_type=agent.agent_id,
            student_persona=persona,
            task_id=task.task_id,
            metrics=episode_metadata["turns"][-1]["metrics"],
            scores=scores.to_dict(),
            pre_quiz=episode_info.pre_quiz_score,
            post_quiz=episode_info.post_quiz_score,
            learning_gain=episode_info.learning_gain,
        )

    # Save accumulated metrics to CSV
    csv_path = output_dir.parent.parent / "processed" / f"{experiment_id}_{agent.agent_id}_summary.csv"
    accumulator.save_to_csv(csv_path)

    print(f"\n✓ Completed {len(conditions)} episodes")
    print(f"✓ Trajectories saved to: {output_dir}")
    print(f"✓ Summary CSV saved to: {csv_path}\n")

    return accumulator


def compare_agents(
    env: TeachingEnv,
    agents: List[BaseTutorAgent],
    tasks: List[TaskSpec],
    student_personas: List[str],
    num_episodes_per_condition: int,
    experiment_id: str,
    base_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, MetricsAccumulator]:
    """
    Compare multiple agents on the same tasks and students.

    Args:
        env: TeachingEnv instance
        agents: List of tutor agents to compare
        tasks: List of tasks
        student_personas: List of student personas
        num_episodes_per_condition: Episodes per condition
        experiment_id: Experiment identifier
        base_seed: Base random seed
        output_dir: Output directory

    Returns:
        Dictionary mapping agent_id -> MetricsAccumulator
    """
    results = {}

    for agent in agents:
        print(f"\n{'='*70}")
        print(f"Running agent: {agent.agent_id}")
        print(f"{'='*70}\n")

        accumulator = run_experiment_batch(
            env=env,
            agent=agent,
            tasks=tasks,
            student_personas=student_personas,
            num_episodes_per_condition=num_episodes_per_condition,
            experiment_id=experiment_id,
            base_seed=base_seed,
            output_dir=output_dir,
            verbose=False,
        )

        results[agent.agent_id] = accumulator

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY: {experiment_id}")
    print(f"{'='*70}\n")

    for agent_id, accumulator in results.items():
        stats = accumulator.get_summary_stats()
        agent_stats = stats.get(agent_id, {})

        print(f"{agent_id}:")
        print(f"  Learning Gain: {agent_stats.get('learning_gain', {}).get('mean', 0):.3f} "
              f"± {agent_stats.get('learning_gain', {}).get('std', 0):.3f}")
        print(f"  Pedagogy Score: {agent_stats.get('pedagogy_score', {}).get('mean', 0):.3f} "
              f"± {agent_stats.get('pedagogy_score', {}).get('std', 0):.3f}")
        print()

    return results
