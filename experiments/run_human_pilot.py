#!/usr/bin/env python3
"""
Human pilot study CLI interface.

Allows human users to interact with tutor agents for qualitative evaluation.

Usage:
    python experiments/run_human_pilot.py
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import random

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_model_config
from src.llm.llm_client import create_llm_client
from src.env.tasks.math_tasks import MathDomain
from src.agents.cot_tutor import CoTTutorAgent
from src.agents.metric_cot_tutor import MetricCoTTutorAgent
from src.agents.react_teacher import ReActTeacherAgent
from src.env.actions import PedagogicalAction, ActionType


class HumanPilotSession:
    """Manages a human pilot study session."""

    def __init__(self, agent, task, llm_client):
        self.agent = agent
        self.task = task
        self.llm_client = llm_client
        self.dialogue_history = []
        self.turn_count = 0

    def start_session(self):
        """Start the tutoring session."""
        print(f"\n{'='*70}")
        print(f"Welcome to TeachTime Tutoring Session")
        print(f"{'='*70}\n")
        print(f"Topic: {self.task.topic}")
        print(f"Description: {self.task.description}\n")
        print("Instructions:")
        print("  - Respond naturally to the tutor's prompts")
        print("  - Type '/end' at any time to finish the session")
        print("  - Type '/help' for more commands\n")
        print(f"{'='*70}\n")

        # Reset agent
        self.agent.reset(self.task)

        # Initial tutor greeting
        initial_message = f"Hi! I'm here to help you learn about {self.task.topic}. Let's get started!"

        print(f"Tutor: {initial_message}\n")

        self.dialogue_history.append(("tutor", initial_message))

    def run_turn(self, user_input: str) -> bool:
        """
        Process one turn of dialogue.

        Returns:
            True to continue, False to end session
        """
        # Check for commands
        if user_input.startswith('/'):
            return self.handle_command(user_input)

        # Record student turn
        self.dialogue_history.append(("student", user_input))
        self.turn_count += 1

        # Create observation-like structure for agent
        from src.env.metrics import compute_metrics, Turn

        # Convert dialogue history to Turn objects
        turns = []
        for i, (speaker, utterance) in enumerate(self.dialogue_history):
            action_type = None
            if speaker == "tutor" and i < len(self.dialogue_history) - 1:
                # Infer action type for metrics (simplified)
                if '?' in utterance:
                    action_type = "Ask_Open_Question"
                else:
                    action_type = "Give_Step_By_Step_Explanation"

            turns.append(Turn(
                speaker=speaker,
                utterance=utterance,
                turn_index=i,
                action_type=action_type,
            ))

        # Compute metrics
        metrics = compute_metrics(turns)

        # Create pseudo-observation
        from src.env.teaching_env import Observation

        obs = Observation(
            student_utterance=user_input,
            dialogue_summary=self._build_summary(),
            metrics=metrics,
            step_index=self.turn_count,
            task_id=self.task.task_id,
            student_id="human",
            done=False,
        )

        # Get agent response
        action = self.agent.act(obs)

        # Display tutor response
        print(f"\nTutor: {action.content}\n")

        # Record tutor turn
        self.dialogue_history.append(("tutor", action.content))
        self.agent.record_turn("student", user_input)

        return True  # Continue session

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Returns:
            True to continue, False to end session
        """
        if command == "/end":
            return False
        elif command == "/help":
            print("\nAvailable commands:")
            print("  /end   - End the session")
            print("  /help  - Show this help message")
            print()
            return True
        else:
            print(f"\nUnknown command: {command}")
            print("Type '/help' for available commands.\n")
            return True

    def _build_summary(self) -> str:
        """Build dialogue summary for agent context."""
        recent = self.dialogue_history[-5:]
        summary = "Recent conversation:\n"
        for speaker, utterance in recent:
            summary += f"{speaker.capitalize()}: {utterance}\n"
        return summary

    def end_session(self):
        """End the session and collect survey responses."""
        print(f"\n{'='*70}")
        print("Session completed! Thank you for participating.")
        print(f"{'='*70}\n")

        # Post-session survey
        print("Please answer a few quick questions (1-5 scale, 5 = strongly agree):\n")

        questions = [
            "I felt listened to",
            "The tutor asked good questions",
            "I understood the topic better after this session",
            "The conversation felt natural",
        ]

        responses = {}
        for q in questions:
            while True:
                try:
                    response = input(f"{q} (1-5): ")
                    score = int(response)
                    if 1 <= score <= 5:
                        responses[q] = score
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

        # Optional open-ended feedback
        print("\nOptional: Any additional comments?")
        feedback = input("> ")

        return responses, feedback

    def save_session(self, session_id: str, agent_type: str, responses: dict, feedback: str):
        """Save session data for analysis."""
        output_dir = PROJECT_ROOT / "results" / "raw" / "exp7_human"
        output_dir.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "task": {
                "task_id": self.task.task_id,
                "topic": self.task.topic,
                "difficulty": self.task.difficulty,
            },
            "dialogue": [
                {"speaker": speaker, "utterance": utterance}
                for speaker, utterance in self.dialogue_history
            ],
            "survey_responses": responses,
            "feedback": feedback,
            "num_turns": self.turn_count,
        }

        output_file = output_dir / f"{session_id}.json"
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"\n✓ Session saved to: {output_file}\n")


def select_task() -> str:
    """Interactive task selection."""
    print("Available topics:")
    print("  1. Linear Equations")
    print("  2. Quadratic Equations")
    print("  3. Function Composition")
    print("  4. Fraction Operations")
    print()

    task_map = {
        "1": "linear_equations",
        "2": "quadratic_equations",
        "3": "function_composition",
        "4": "fraction_operations",
    }

    while True:
        choice = input("Choose a topic (1-4): ")
        if choice in task_map:
            return task_map[choice]
        print("Invalid choice. Please enter 1-4.")


def main():
    parser = argparse.ArgumentParser(description="Human pilot study interface")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["baseline_cot", "metric_cot", "react_teacher", "random"],
        default="random",
        help="Agent to use (default: random assignment)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task ID (default: interactive selection)"
    )

    args = parser.parse_args()

    # Load config
    model_config = load_model_config()

    # Create LLM client
    llm_client = create_llm_client(
        model=model_config.default_model,
        enable_cache=model_config.enable_cache,
        verbose=False,
    )

    # Get participant ID
    print("\nTeachTime Human Pilot Study")
    print("="*70)
    participant_id = input("\nEnter your name or ID (anonymous): ")

    # Select task
    if args.task:
        task_id = args.task
    else:
        task_id = select_task()

    domain = MathDomain()
    task = domain.get_task(task_id)

    # Select/assign agent (blind to participant)
    if args.agent == "random":
        agent_type = random.choice(["baseline_cot", "react_teacher"])
    else:
        agent_type = args.agent

    # Create agent (participant doesn't know which)
    if agent_type == "baseline_cot":
        agent = CoTTutorAgent(llm_client=llm_client)
    elif agent_type == "metric_cot":
        agent = MetricCoTTutorAgent(llm_client=llm_client)
    else:
        agent = ReActTeacherAgent(llm_client=llm_client)

    # Create session
    session_id = f"human_{participant_id}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = HumanPilotSession(agent, task, llm_client)

    # Run session
    session.start_session()

    continue_session = True
    while continue_session:
        try:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            continue_session = session.run_turn(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession interrupted.")
            break

    # End session and collect survey
    responses, feedback = session.end_session()

    # Save session data
    session.save_session(session_id, agent_type, responses, feedback)

    print("Thank you for participating!\n")


if __name__ == "__main__":
    main()
