"""
Synthetic student agent for Socratic tutoring simulation.

Design principles:
1. Students initially work through problems showing reasoning
2. Students only submit FINAL: when confident (after guidance)
3. Students make realistic mistakes based on cognitive load
4. Proper alternating message structure (no role prefixes in content)

From student's perspective: student=assistant, teacher=user
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from ..config import StudentConfig, DEFAULT_STUDENT_CONFIG
from ..datasets import Question
from ..llm_client import LLMClient


@dataclass
class StudentTurn:
    """Output from a student turn."""
    raw_text: str
    reasoning: str
    final_answer: Optional[str]  # Only present if student submitted FINAL:
    confidence: str  # "low", "medium", "high"
    
    @classmethod
    def from_raw(cls, raw_text: str) -> "StudentTurn":
        """Parse student response."""
        text = raw_text.strip()
        
        reasoning = text
        final_answer = None
        confidence = "medium"
        
        # Extract FINAL: answer if present
        final_match = re.search(r'FINAL:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if final_match:
            candidate = final_match.group(1).strip().rstrip('.,;:')
            if candidate:
                final_answer = candidate
        
        # Detect confidence from language
        low_markers = r'\b(unsure|uncertain|not sure|maybe|I think|probably|guess|confused)\b'
        high_markers = r'\b(definitely|certainly|confident|sure|clearly)\b'
        
        if re.search(low_markers, text, re.IGNORECASE):
            confidence = "low"
        elif re.search(high_markers, text, re.IGNORECASE):
            confidence = "high"
        
        return cls(
            raw_text=raw_text,
            reasoning=reasoning,
            final_answer=final_answer,
            confidence=confidence,
        )
    
    @property
    def has_final_answer(self) -> bool:
        return self.final_answer is not None


@dataclass
class DialogueContext:
    """Context for student dialogue."""
    question: Question
    initial_student_turn: StudentTurn
    history: list[tuple[str, str]]  # (speaker, message) tuples
    
    @property
    def last_teacher_message(self) -> Optional[str]:
        for speaker, msg in reversed(self.history):
            if speaker == "teacher":
                return msg
        return None
    
    @property
    def turn_count(self) -> int:
        return sum(1 for s, _ in self.history if s == "teacher")


def get_student_system_prompt(category: str) -> str:
    """
    Generate student system prompt.
    
    Key design: Student does NOT immediately jump to FINAL.
    They show reasoning, may express uncertainty, and only
    submit FINAL when they feel confident (typically after guidance).
    """
    return f"""You are a student learning {category}. You are working through a problem with a tutor's help.

## How to Respond

1. **Show your thinking**: Work through the problem step by step
2. **Express uncertainty**: If unsure, say so explicitly
3. **Ask clarifying questions**: If something is unclear, ask
4. **Respond to hints**: When the tutor gives guidance, incorporate it
5. **Build on feedback**: Use the tutor's corrections to refine your approach

## When to Submit Final Answer

ONLY use "FINAL: <answer>" when you are confident in your solution.
- First attempts: Show reasoning, express uncertainties
- After guidance: Incorporate feedback, then submit FINAL if confident
- If still unsure: Ask for more help instead of guessing

## Response Format

Keep responses focused and under 200 words. Structure:
- Your current reasoning or approach
- Any uncertainties or questions
- FINAL: <answer> (only when confident)

Remember: It's okay to be wrong or unsure. The goal is learning through dialogue."""


class StudentAgent:
    """
    Student agent that simulates realistic learning behavior.
    
    Key behaviors:
    1. Shows reasoning process, not just answers
    2. Makes realistic mistakes based on problem difficulty
    3. Incorporates teacher feedback
    4. Only submits FINAL when confident
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: StudentConfig = DEFAULT_STUDENT_CONFIG,
    ):
        self.llm = llm_client
        self.config = config
    
    def answer_initial_question(self, question: Question) -> StudentTurn:
        """
        Generate student's first attempt at the problem.
        
        Uses proper message structure:
        [system] Student persona
        [user] The problem to solve
        """
        category = question.category or "STEM"
        
        messages = [
            {"role": "system", "content": get_student_system_prompt(category)},
            {"role": "user", "content": question.question},
        ]
        
        raw = self.llm.chat(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return StudentTurn.from_raw(raw)
    
    def respond_in_dialogue(self, context: DialogueContext) -> StudentTurn:
        """
        Generate student response to tutor feedback.
        
        Uses proper alternating message structure:
        [system] Student persona
        [user] Original problem
        [assistant] Initial attempt
        [user] Tutor feedback 1
        [assistant] Student response 1
        ...
        [user] Latest tutor feedback
        """
        last_msg = context.last_teacher_message
        if not last_msg:
            raise ValueError("No teacher message in history")
        
        category = context.question.category or "STEM"
        messages = self._build_dialogue_messages(context, category)
        
        raw = self.llm.chat(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return StudentTurn.from_raw(raw)
    
    def _build_dialogue_messages(
        self,
        context: DialogueContext,
        category: str,
    ) -> list[dict]:
        """
        Build message list with proper alternating structure.
        
        Clean design: No "Teacher:" or "Student:" prefixes.
        The role field already indicates who is speaking.
        """
        messages = [
            {"role": "system", "content": get_student_system_prompt(category)},
            {"role": "user", "content": context.question.question},
            {"role": "assistant", "content": context.initial_student_turn.raw_text},
        ]
        
        # Add dialogue history
        for speaker, msg in context.history:
            if speaker == "teacher":
                messages.append({"role": "user", "content": msg})
            else:
                messages.append({"role": "assistant", "content": msg})
        
        return messages
    
    @classmethod
    def from_config(cls, config: StudentConfig, llm_client: LLMClient) -> "StudentAgent":
        return cls(llm_client=llm_client, config=config)
