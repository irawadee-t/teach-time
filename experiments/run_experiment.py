#!/usr/bin/env python3
"""
Run a single TeachTime experiment from configuration.

Usage:
    python experiments/run_experiment.py --config exp1_metrics_match
    python experiments/run_experiment.py --config exp2_learning_gains --verbose
"""

import argparse
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_experiment_config, load_model_config, load_env_config
from src.utils.random_utils import set_global_seed
from src.llm.llm_client import create_llm_client
from src.env.teaching_env import TeachingEnv
from src.env.tasks.math_tasks import MathDomain
from src.agents.cot_tutor import CoTTutorAgent
from src.agents.metric_cot_tutor import MetricCoTTutorAgent
from src.agents.react_teacher import ReActTeacherAgent, ReActTeacherNoMetrics
from src.eval.eval_loops import compare_agents


def create_agent(agent_type: str, llm_client):
    """Factory function to create agent by type string."""
    agent_map = {
        "baseline_cot": lambda: CoTTutorAgent(
            agent_id="baseline_cot",
            llm_client=llm_client,
        ),
        "metric_cot": lambda: MetricCoTTutorAgent(
            agent_id="metric_cot",
            llm_client=llm_client,
        ),
        "react_teacher": lambda: ReActTeacherAgent(
            agent_id="react_teacher",
            llm_client=llm_client,
        ),
        "react_teacher_no_metrics": lambda: ReActTeacherNoMetrics(
            agent_id="react_teacher_no_metrics",
            llm_client=llm_client,
        ),
    }

    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent_map[agent_type]()


def main():
    parser = argparse.ArgumentParser(description="Run a TeachTime experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment config name (e.g., 'exp1_metrics_match')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override default model"
    )

    args = parser.parse_args()

    # Load configurations
    print(f"\n{'='*70}")
    print(f"Loading experiment configuration: {args.config}")
    print(f"{'='*70}\n")

    exp_config = load_experiment_config(args.config)
    model_config = load_model_config()
    env_config = load_env_config()

    # Override model if specified
    if args.model:
        model_config.default_model = args.model

    # Set global random seed
    set_global_seed(exp_config.random_seed)

    # Create LLM client
    print(f"Initializing LLM client (model: {model_config.default_model})...")
    llm_client = create_llm_client(
        model=model_config.default_model,
        enable_cache=model_config.enable_cache,
        verbose=args.verbose,
    )

    # Create environment
    print(f"Creating teaching environment...")
    env = TeachingEnv(
        llm_client=llm_client,
        max_turns=exp_config.max_turns,
        enable_quizzes=exp_config.enable_quizzes,
        summary_window=env_config.summary_window,
        random_seed=exp_config.random_seed,
    )

    # Load tasks
    print(f"Loading tasks from domain: {exp_config.domain}...")
    domain = MathDomain(random_seed=exp_config.random_seed)
    tasks = [domain.get_task(task_id) for task_id in exp_config.tasks]

    print(f"Loaded {len(tasks)} tasks: {[t.task_id for t in tasks]}")

    # Create agents
    print(f"\nCreating agents: {exp_config.agents}")
    agents = [create_agent(agent_type, llm_client) for agent_type in exp_config.agents]

    # Run experiment
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {exp_config.name}")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  - Agents: {len(agents)}")
    print(f"  - Tasks: {len(tasks)}")
    print(f"  - Student personas: {len(exp_config.students)}")
    print(f"  - Episodes per condition: {exp_config.num_episodes_per_condition}")
    print(f"  - Total episodes: {len(agents) * len(tasks) * len(exp_config.students) * exp_config.num_episodes_per_condition}")
    print()

    results = compare_agents(
        env=env,
        agents=agents,
        tasks=tasks,
        student_personas=exp_config.students,
        num_episodes_per_condition=exp_config.num_episodes_per_condition,
        experiment_id=exp_config.name,
        base_seed=exp_config.random_seed,
    )

    # Print final statistics
    print(f"\n{'='*70}")
    print(f"Experiment completed: {exp_config.name}")
    print(f"{'='*70}\n")

    print("LLM Usage Statistics:")
    usage = llm_client.get_usage_stats()
    print(f"  - Total API calls: {usage['total_api_calls']}")
    print(f"  - Cached hits: {usage['total_cached_hits']}")
    print(f"  - Total tokens: {usage['total_tokens']}")
    print(f"  - Cache size: {usage['cache_size']} entries")
    print()

    print("✓ Results saved to results/raw/ and results/processed/")
    print("✓ Run analysis notebooks to generate plots\n")


if __name__ == "__main__":
    main()
