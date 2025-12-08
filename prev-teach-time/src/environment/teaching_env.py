"""
TeachTime teaching environment.

A gym-like environment for tutoring episodes with observable teaching metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import random

from .actions import PedagogicalAction, ActionType
from .metrics import Turn, compute_metrics, format_metrics_for_prompt
from .student_models import LLMStudent, create_student
from .tasks.base import TaskSpec, compute_quiz_score


@dataclass
class Observation:
    """
    What the agent observes at each step.

    This is the core state representation that agents use to make decisions.
    """
    student_utterance: str
    dialogue_summary: str  # Rolling summary of last N turns
    metrics: Dict[str, Union[float, int, bool]]
    step_index: int
    task_id: str
    student_id: str
    done: bool = False

    def to_dict(self) -> dict:
        return {
            "student_utterance": self.student_utterance,
            "dialogue_summary": self.dialogue_summary,
            "metrics": self.metrics,
            "step_index": self.step_index,
            "task_id": self.task_id,
            "student_id": self.student_id,
            "done": self.done,
        }


@dataclass
class EpisodeInfo:
    """Additional information returned with terminal observations."""
    pre_quiz_score: float
    post_quiz_score: float
    learning_gain: float
    final_knowledge_state: Dict[str, float]
    dialogue_history: List[Tuple[str, str]]
    success: bool = False


class TeachingEnv:
    """
    Gym-like environment for tutoring episodes.

    Each episode:
    1. Administers pre-quiz (optional)
    2. Runs tutoring session for max_turns
    3. Administers post-quiz (optional)
    4. Computes learning gain and metrics
    """

    def __init__(
        self,
        llm_client,
        max_turns: int = 10,
        enable_quizzes: bool = True,
        summary_window: int = 5,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            llm_client: LLM client for student simulation and quiz grading
            max_turns: Maximum number of dialogue turns per episode
            enable_quizzes: Whether to administer pre/post quizzes
            summary_window: Number of recent turns to include in dialogue summary
            random_seed: Random seed for reproducibility
        """
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.enable_quizzes = enable_quizzes
        self.summary_window = summary_window
        self.random = random.Random(random_seed)

        # Episode state
        self.current_task: Optional[TaskSpec] = None
        self.current_student: Optional[LLMStudent] = None
        self.dialogue_history: List[Turn] = []
        self.step_count: int = 0
        self.pre_quiz_score: float = 0.0
        self.post_quiz_score: float = 0.0
        self.episode_done: bool = False

    def reset(
        self,
        task: TaskSpec,
        student_persona: str = "struggling",
        student_id: Optional[str] = None,
    ) -> Observation:
        """
        Start a new episode.

        Args:
            task: TaskSpec defining the learning task
            student_persona: Student persona type ("struggling", "confident_mistaken", "minimal_talker")
            student_id: Optional specific student ID

        Returns:
            Initial observation
        """
        self.current_task = task
        self.dialogue_history = []
        self.step_count = 0
        self.episode_done = False

        # Create student
        self.current_student = create_student(
            persona_type=student_persona,
            task_concepts=task.key_concepts,
            llm_client=self.llm_client,
            student_id=student_id,
            random_seed=self.random.randint(0, 1000000),
        )

        # Administer pre-quiz if enabled
        if self.enable_quizzes:
            _, self.pre_quiz_score = self.current_student.take_quiz(
                task.pre_quiz,
                compute_quiz_score
            )
        else:
            self.pre_quiz_score = 0.0

        # Initial observation - student introduces themselves or waits
        initial_student_utterance = self._generate_initial_student_utterance()

        # Add to dialogue history
        turn = Turn(
            speaker="student",
            utterance=initial_student_utterance,
            turn_index=0,
            action_type=None
        )
        self.dialogue_history.append(turn)

        # Compute initial metrics (mostly zeros)
        metrics = compute_metrics(self.dialogue_history)

        # Build observation
        obs = Observation(
            student_utterance=initial_student_utterance,
            dialogue_summary=self._build_dialogue_summary(),
            metrics=metrics,
            step_index=self.step_count,
            task_id=task.task_id,
            student_id=self.current_student.student_id,
            done=False,
        )

        return obs

    def step(
        self,
        action: PedagogicalAction
    ) -> Tuple[Observation, Optional[float], bool, EpisodeInfo]:
        """
        Execute a pedagogical action and get next observation.

        Args:
            action: PedagogicalAction from the agent

        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Next state
            - reward: Optional reward signal (None for now, could be used for RL later)
            - done: Whether episode is complete
            - info: Additional episode information (populated when done=True)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.step_count += 1

        # Realize action as tutor utterance (could use LLM to polish, but for now use content directly)
        tutor_utterance = action.content

        # Add tutor turn to dialogue history
        tutor_turn = Turn(
            speaker="tutor",
            utterance=tutor_utterance,
            turn_index=len(self.dialogue_history),
            action_type=action.action_type.value,
        )
        self.dialogue_history.append(tutor_turn)

        # Update student knowledge based on teaching action
        self.current_student.update_knowledge_from_teaching(
            tutor_action_type=action.action_type.value,
            tutor_utterance=tutor_utterance,
            concepts_addressed=self.current_task.key_concepts,
        )

        # Check if episode should end
        done = self._check_done(action)

        if done:
            # Terminal step - wrap up episode
            self.episode_done = True

            # Administer post-quiz if enabled
            if self.enable_quizzes:
                _, self.post_quiz_score = self.current_student.take_quiz(
                    self.current_task.post_quiz,
                    compute_quiz_score
                )
            else:
                self.post_quiz_score = 0.0

            # Compute learning gain
            learning_gain = self._compute_learning_gain()

            # Build final observation (no student response needed)
            final_metrics = compute_metrics(self.dialogue_history)

            obs = Observation(
                student_utterance="[Session ended]",
                dialogue_summary=self._build_dialogue_summary(),
                metrics=final_metrics,
                step_index=self.step_count,
                task_id=self.current_task.task_id,
                student_id=self.current_student.student_id,
                done=True,
            )

            # Build episode info
            info = EpisodeInfo(
                pre_quiz_score=self.pre_quiz_score,
                post_quiz_score=self.post_quiz_score,
                learning_gain=learning_gain,
                final_knowledge_state=self.current_student.knowledge.concept_mastery.copy(),
                dialogue_history=[(t.speaker, t.utterance) for t in self.dialogue_history],
                success=(learning_gain > 0.1),  # Arbitrary threshold
            )

            return obs, None, True, info

        else:
            # Get student response
            student_response = self.current_student.respond(tutor_utterance)

            # Add student turn to dialogue history
            student_turn = Turn(
                speaker="student",
                utterance=student_response,
                turn_index=len(self.dialogue_history),
                action_type=None,
            )
            self.dialogue_history.append(student_turn)

            # Compute metrics
            metrics = compute_metrics(self.dialogue_history)

            # Build observation
            obs = Observation(
                student_utterance=student_response,
                dialogue_summary=self._build_dialogue_summary(),
                metrics=metrics,
                step_index=self.step_count,
                task_id=self.current_task.task_id,
                student_id=self.current_student.student_id,
                done=False,
            )

            return obs, None, False, None

    def _generate_initial_student_utterance(self) -> str:
        """Generate initial student utterance to start the session."""
        greetings = [
            "Hi, I'm ready to learn!",
            "Hello! I'm not sure I understand this topic well.",
            "Hi there.",
            "Hey, I need help with this.",
        ]

        # Persona-specific variations
        if self.current_student.persona_type == "struggling":
            return self.random.choice([
                "Hi, I'm really struggling with this topic and need help.",
                "Hello, I find this topic pretty confusing.",
            ])
        elif self.current_student.persona_type == "confident_mistaken":
            return self.random.choice([
                "Hi! I think I understand the basics, but let's see.",
                "Hey, I've seen this before. Let's get started.",
            ])
        elif self.current_student.persona_type == "minimal_talker":
            return self.random.choice([
                "Hi.",
                "Hello.",
                "Hey.",
            ])

        return self.random.choice(greetings)

    def _check_done(self, action: PedagogicalAction) -> bool:
        """Check if episode should end."""
        # End if max turns reached
        if self.step_count >= self.max_turns:
            return True

        # End if tutor explicitly wraps up
        if action.action_type == ActionType.SUMMARIZE_AND_WRAP_UP:
            return True

        return False

    def _compute_learning_gain(self) -> float:
        """
        Compute normalized learning gain.

        Uses: (post - pre) / (1 - pre)
        This accounts for ceiling effects (harder to improve if already high).
        """
        if self.pre_quiz_score >= 0.99:
            # Already at ceiling
            return 0.0

        gain = (self.post_quiz_score - self.pre_quiz_score) / (1 - self.pre_quiz_score)
        return max(0.0, gain)  # Clamp to non-negative

    def _build_dialogue_summary(self) -> str:
        """Build a summary of recent dialogue for context."""
        if not self.dialogue_history:
            return "Session just started."

        # Take last N turns
        recent_turns = self.dialogue_history[-self.summary_window:]

        summary = "Recent conversation:\n"
        for turn in recent_turns:
            summary += f"{turn.speaker.capitalize()}: {turn.utterance}\n"

        return summary

    def get_episode_transcript(self) -> str:
        """Get formatted transcript of full episode."""
        transcript = f"=== Episode: {self.current_task.task_id} ===\n"
        transcript += f"Student: {self.current_student.student_id} ({self.current_student.persona_type})\n"
        transcript += f"Pre-quiz score: {self.pre_quiz_score:.2f}\n\n"

        for turn in self.dialogue_history:
            action_str = f" [{turn.action_type}]" if turn.action_type else ""
            transcript += f"{turn.speaker.capitalize()}{action_str}: {turn.utterance}\n\n"

        if self.episode_done:
            transcript += f"Post-quiz score: {self.post_quiz_score:.2f}\n"
            transcript += f"Learning gain: {self._compute_learning_gain():.2f}\n"

        return transcript

    def render(self):
        """Print current state (for debugging)."""
        print(self.get_episode_transcript())
