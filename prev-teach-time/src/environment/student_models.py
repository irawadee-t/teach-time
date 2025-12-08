"""
Simulated student models for TeachTime environment.

All students are LLM-based with internal knowledge state tracking.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class KnowledgeState:
    """
    Tracks student's knowledge of key concepts.

    Each concept has a mastery level from 0.0 (no understanding) to 1.0 (full mastery).
    """
    concept_mastery: Dict[str, float] = field(default_factory=dict)

    def get_mastery(self, concept: str) -> float:
        """Get mastery level for a concept (default 0.0 if not tracked)."""
        return self.concept_mastery.get(concept, 0.0)

    def update_mastery(self, concept: str, delta: float):
        """Update mastery level (clipped to [0, 1])."""
        current = self.get_mastery(concept)
        self.concept_mastery[concept] = max(0.0, min(1.0, current + delta))

    def get_average_mastery(self) -> float:
        """Get average mastery across all concepts."""
        if not self.concept_mastery:
            return 0.0
        return sum(self.concept_mastery.values()) / len(self.concept_mastery)

    def to_dict(self) -> dict:
        return {"concept_mastery": self.concept_mastery}


class LLMStudent:
    """
    LLM-based student with a specific persona and knowledge state.

    The student responds using an LLM, with the persona and knowledge state
    injected into the prompt to simulate realistic student behavior.
    """

    def __init__(
        self,
        student_id: str,
        persona_type: str,
        task_concepts: List[str],
        llm_client,  # Will be injected
        initial_knowledge: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            student_id: Unique identifier for this student
            persona_type: One of "struggling", "confident_mistaken", "minimal_talker"
            task_concepts: List of concepts for this task (for knowledge tracking)
            llm_client: LLM client for generating responses
            initial_knowledge: Optional dict of concept -> initial mastery level
            random_seed: Random seed for reproducibility
        """
        self.student_id = student_id
        self.persona_type = persona_type
        self.llm_client = llm_client
        self.random = random.Random(random_seed)

        # Initialize knowledge state
        self.knowledge = KnowledgeState()
        if initial_knowledge:
            self.knowledge.concept_mastery = initial_knowledge.copy()
        else:
            # Set default initial knowledge based on persona
            base_level = self._get_base_knowledge_level()
            for concept in task_concepts:
                self.knowledge.concept_mastery[concept] = base_level

        # Dialogue history (for context)
        self.dialogue_history: List[Tuple[str, str]] = []  # (speaker, utterance)

        # Misconceptions (if any)
        self.misconceptions = self._initialize_misconceptions()

    def _get_base_knowledge_level(self) -> float:
        """Get base knowledge level based on persona."""
        persona_configs = {
            "struggling": 0.2,
            "confident_mistaken": 0.5,
            "minimal_talker": 0.3,
        }
        return persona_configs.get(self.persona_type, 0.3)

    def _initialize_misconceptions(self) -> List[str]:
        """Initialize misconceptions based on persona."""
        if self.persona_type == "confident_mistaken":
            return [
                "Tends to apply rules mechanically without understanding",
                "Often makes systematic errors (e.g., sign errors, order of operations)",
            ]
        return []

    def _get_persona_prompt(self) -> str:
        """Get the system prompt that defines the student persona."""
        base_prompt = (
            f"You are a student learning a new math concept. "
            f"Your current knowledge state:\n"
        )

        # Add knowledge state
        for concept, mastery in self.knowledge.concept_mastery.items():
            level_desc = self._mastery_to_description(mastery)
            base_prompt += f"- {concept}: {level_desc} (mastery: {mastery:.2f})\n"

        # Add persona-specific behaviors
        persona_prompts = {
            "struggling": (
                "\n\nYour persona: You are a struggling student who finds math challenging. "
                "You often feel confused and need things explained multiple times. "
                "You ask clarifying questions when you don't understand. "
                "You give longer, more elaborate responses when you're uncertain. "
                "You express confusion openly (e.g., 'I'm not sure', 'I don't understand'). "
                "When you learn something new, it takes multiple explanations to really grasp it."
            ),
            "confident_mistaken": (
                "\n\nYour persona: You are a confident student who thinks you understand, "
                "but you have systematic misconceptions. You answer quickly and assertively, "
                "but sometimes make consistent errors. For example, you might forget to apply "
                "operations to both sides of an equation, or make sign errors. "
                "You don't naturally express confusion unless directly challenged. "
                "You give brief, confident answers. Once a misconception is corrected, "
                "you learn relatively quickly."
            ),
            "minimal_talker": (
                "\n\nYour persona: You are a quiet, reserved student who gives very brief responses. "
                "You typically answer with 1-2 sentences at most. You don't elaborate unless "
                "specifically asked to explain your thinking. You understand concepts at a basic level "
                "but aren't comfortable volunteering detailed explanations. "
                "Your responses are terse: 'yes', 'no', 'I think it's [answer]', etc."
            ),
        }

        base_prompt += persona_prompts.get(self.persona_type, "")

        # Add dialogue context instructions
        base_prompt += (
            "\n\nRespond naturally as this student would. "
            "Stay in character. Your response should reflect your current knowledge level "
            "and persona traits."
        )

        return base_prompt

    def _mastery_to_description(self, mastery: float) -> str:
        """Convert mastery level to description."""
        if mastery < 0.3:
            return "very weak understanding"
        elif mastery < 0.5:
            return "basic understanding"
        elif mastery < 0.7:
            return "moderate understanding"
        elif mastery < 0.9:
            return "good understanding"
        else:
            return "strong understanding"

    def respond(self, tutor_utterance: str) -> str:
        """
        Generate student response to tutor's utterance.

        Args:
            tutor_utterance: What the tutor just said

        Returns:
            Student's response
        """
        # Add to dialogue history
        self.dialogue_history.append(("tutor", tutor_utterance))

        # Build conversation context
        conversation = self._build_conversation_context()

        # Get LLM response
        system_prompt = self._get_persona_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation}
        ]

        # Call LLM (will be implemented via llm_client)
        # For now, return a placeholder that will be replaced when llm_client is integrated
        student_response = self.llm_client.call(
            messages=messages,
            temperature=0.7,
            max_tokens=200,
        )

        # Add to dialogue history
        self.dialogue_history.append(("student", student_response))

        return student_response

    def _build_conversation_context(self) -> str:
        """Build conversation context from recent history."""
        if not self.dialogue_history:
            return "The tutor has just started the session."

        # Include last few turns for context (up to 5)
        recent_history = self.dialogue_history[-5:]
        context = "Recent conversation:\n"
        for speaker, utterance in recent_history:
            context += f"{speaker.capitalize()}: {utterance}\n"

        context += "\nNow respond as the student to the tutor's most recent message."
        return context

    def update_knowledge_from_teaching(
        self,
        tutor_action_type: str,
        tutor_utterance: str,
        concepts_addressed: List[str]
    ):
        """
        Update knowledge state based on tutor's teaching action.

        Args:
            tutor_action_type: The pedagogical action type used
            tutor_utterance: What the tutor said
            concepts_addressed: Which concepts were addressed in this turn
        """
        # Learning rate depends on action type and student persona
        learning_rates = {
            "struggling": {
                "Ask_Open_Question": 0.02,
                "Ask_Check_Understanding": 0.03,
                "Give_Step_By_Step_Explanation": 0.05,
                "Ask_Background": 0.01,
                "Assign_Practice_Problem": 0.04,
                "Summarize_And_Wrap_Up": 0.03,
            },
            "confident_mistaken": {
                "Ask_Open_Question": 0.04,
                "Ask_Check_Understanding": 0.05,
                "Give_Step_By_Step_Explanation": 0.06,
                "Ask_Background": 0.02,
                "Assign_Practice_Problem": 0.07,
                "Summarize_And_Wrap_Up": 0.04,
            },
            "minimal_talker": {
                "Ask_Open_Question": 0.03,
                "Ask_Check_Understanding": 0.04,
                "Give_Step_By_Step_Explanation": 0.05,
                "Ask_Background": 0.02,
                "Assign_Practice_Problem": 0.05,
                "Summarize_And_Wrap_Up": 0.03,
            },
        }

        persona_rates = learning_rates.get(self.persona_type, learning_rates["struggling"])
        learning_rate = persona_rates.get(tutor_action_type, 0.03)

        # Update knowledge for addressed concepts
        for concept in concepts_addressed:
            current_mastery = self.knowledge.get_mastery(concept)

            # Diminishing returns: harder to learn when already know something
            adjusted_rate = learning_rate * (1 - current_mastery * 0.5)

            # Add some noise
            noise = self.random.uniform(-0.01, 0.01)
            delta = adjusted_rate + noise

            self.knowledge.update_mastery(concept, delta)

    def take_quiz(self, questions: List, answer_callback) -> Tuple[List[str], float]:
        """
        Take a quiz and return answers + score.

        Args:
            questions: List of QuizQuestion objects
            answer_callback: Function to grade answers

        Returns:
            Tuple of (answers, score)
        """
        answers = []

        for question in questions:
            # Generate answer based on knowledge
            answer = self._answer_question(question)
            answers.append(answer)

        # Grade using callback
        from .tasks.base import compute_quiz_score
        score = compute_quiz_score(questions, answers)

        return answers, score

    def _answer_question(self, question) -> str:
        """
        Answer a quiz question based on current knowledge.

        Args:
            question: QuizQuestion object

        Returns:
            Student's answer
        """
        concept = question.concept
        mastery = self.knowledge.get_mastery(concept)

        if question.question_type == "multiple_choice":
            # Probability of getting it right based on mastery
            # Baseline 25% (random guess), up to 100% with full mastery
            p_correct = 0.25 + 0.75 * mastery

            if self.random.random() < p_correct:
                return question.correct_answer
            else:
                # Choose a wrong answer
                wrong_options = [opt[0] for opt in question.options
                                if opt[0] != question.correct_answer]
                return self.random.choice(wrong_options) if wrong_options else "A"

        else:  # short_answer
            # Generate answer using LLM with knowledge-conditioned prompt
            system_prompt = (
                f"You are a student with {self._mastery_to_description(mastery)} "
                f"of {concept}. Answer the following question as this student would. "
                f"Give a brief answer (1-2 sentences)."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question.question}
            ]

            answer = self.llm_client.call(
                messages=messages,
                temperature=0.5,
                max_tokens=100,
            )

            return answer.strip()

    def reset_dialogue(self):
        """Reset dialogue history (e.g., for new episode)."""
        self.dialogue_history = []

    def get_state_dict(self) -> dict:
        """Get full state for logging/serialization."""
        return {
            "student_id": self.student_id,
            "persona_type": self.persona_type,
            "knowledge_state": self.knowledge.to_dict(),
            "dialogue_history": self.dialogue_history,
            "misconceptions": self.misconceptions,
        }


def create_student(
    persona_type: str,
    task_concepts: List[str],
    llm_client,
    student_id: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> LLMStudent:
    """
    Factory function to create a student with specified persona.

    Args:
        persona_type: One of "struggling", "confident_mistaken", "minimal_talker"
        task_concepts: List of concepts for knowledge tracking
        llm_client: LLM client instance
        student_id: Optional ID (generated if not provided)
        random_seed: Random seed for reproducibility

    Returns:
        LLMStudent instance
    """
    if student_id is None:
        student_id = f"{persona_type}_{random.randint(1000, 9999)}"

    return LLMStudent(
        student_id=student_id,
        persona_type=persona_type,
        task_concepts=task_concepts,
        llm_client=llm_client,
        random_seed=random_seed,
    )
