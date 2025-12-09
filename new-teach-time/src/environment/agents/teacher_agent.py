"""
Socratic teacher agent using ReAct reasoning.

Design principles (per ReAct paper, Yao et al. 2022):
1. Thought: Internal reasoning about student state
2. Action: Domain-specific action from action space
3. Observation: Student response + judge verdict (when FINAL submitted)

The teacher:
- Knows the ground truth (for pedagogical guidance)
- Receives judge verdict when student submits FINAL
- Generates Thought → Action → Utterance in strict format
- Never uses END_SUCCESS unless judge confirms correctness
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from ..config import TeacherConfig
from ..datasets import Question
from ..llm_client import LLMClient
from ..react.parser import parse_teacher_response, attempt_repair
from ..react.actions import ACTIONS, is_terminal_action
from .student_agent import StudentTurn


@dataclass
class TeacherTurn:
    """Output from a teacher turn in ReAct format."""
    thought: str
    action: str
    utterance: str
    is_terminal: bool
    raw_output: str
    parse_errors: list[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return len(self.parse_errors) == 0 and self.action in ACTIONS
    
    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "utterance": self.utterance,
            "is_terminal": self.is_terminal,
            "raw_output": self.raw_output,
            "parse_errors": self.parse_errors,
        }


TEACHER_SYSTEM_PROMPT = """You are a Socratic tutor. Guide students to discover answers through hints and questions.

## ReAct Format (STRICT - you MUST follow this exactly)

Every response must have exactly three parts:

Thought: <your reasoning about the student's current state>
Action: <ONE action from the list below>
Utterance: <what you say to the student, 1-3 sentences>

## Action Space (use ONLY these exact names)

GIVE_HINT - Provide a targeted clue
ASK_QUESTION - Ask a guiding question
CORRECT_ERROR - Point out and address a mistake
EXPLAIN_CONCEPT - Explain something the student is missing
CONFIRM_PROGRESS - Acknowledge correct reasoning
END_SUCCESS - Student's FINAL answer is CORRECT (only use when judge confirms)
END_STUCK - Student cannot reach the answer

## Rules

1. NEVER reveal the answer directly (except in END_SUCCESS)
2. Use END_SUCCESS ONLY when the judge has confirmed the student's FINAL answer is correct
3. Use END_STUCK if judge says incorrect and student cannot progress
4. If student hasn't submitted FINAL yet, continue guiding them
5. Always output Thought, Action, Utterance in that order"""


def get_teacher_context(
    question: Question,
    turn_number: int,
    max_turns: int,
    judge_verdict: Optional[bool] = None,
    student_final_answer: Optional[str] = None,
) -> str:
    """Generate context block for teacher including judge verdict."""
    context = f"""---
PROBLEM: {question.question}
CORRECT ANSWER: {question.ground_truth}
[Turn {turn_number}/{max_turns}]"""
    
    # Add judge verdict if student submitted FINAL
    if student_final_answer is not None:
        verdict_str = "CORRECT" if judge_verdict else "INCORRECT"
        context += f"""

STUDENT SUBMITTED: FINAL: {student_final_answer}
JUDGE VERDICT: {verdict_str}

Note: Use END_SUCCESS only if judge says CORRECT. Use CORRECT_ERROR if judge says INCORRECT."""
    
    if turn_number >= max_turns - 1:
        context += "\n⚠️ APPROACHING TURN LIMIT - wrap up soon."
    
    context += "\n---"
    return context


class TeacherAgent:
    """
    ReAct teacher agent that uses external judge verdicts.
    
    When student submits FINAL, the judge verdict is passed to teacher.
    Teacher decides action based on this information.
    """
    
    def __init__(self, llm_client: LLMClient, config: TeacherConfig):
        self.llm = llm_client
        self.config = config
    
    def reset(self):
        """Reset for new session."""
        pass
    
    def step(
        self,
        question: Question,
        initial_student_turn: StudentTurn,
        history: list[tuple[str, str]],
        turn_number: int,
        max_turns: int,
        latest_student_turn: Optional[StudentTurn] = None,
        judge_verdict: Optional[bool] = None,
    ) -> TeacherTurn:
        """
        Execute one teacher turn.
        
        Args:
            question: The problem being worked on
            initial_student_turn: Student's first attempt
            history: Dialogue history
            turn_number: Current turn number
            max_turns: Maximum turns allowed
            latest_student_turn: Most recent student response
            judge_verdict: External judge verdict if student submitted FINAL
        """
        # Extract student's final answer if present
        student_final = None
        if latest_student_turn and latest_student_turn.final_answer:
            student_final = latest_student_turn.final_answer
        
        messages = self._build_messages(
            question=question,
            initial_student_turn=initial_student_turn,
            history=history,
            turn_number=turn_number,
            max_turns=max_turns,
            judge_verdict=judge_verdict,
            student_final_answer=student_final,
        )
        
        raw = self.llm.chat(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Parse ReAct output
        parsed = parse_teacher_response(raw)
        if not parsed.is_valid:
            parsed = attempt_repair(parsed)
        
        # Validate action against judge verdict
        action = parsed.action
        if student_final and judge_verdict is False and action == "END_SUCCESS":
            # Judge says incorrect but teacher trying to end with success
            # Override to CORRECT_ERROR
            action = "CORRECT_ERROR"
            parsed.validation_errors.append("Overridden: END_SUCCESS with incorrect verdict")
        
        # Ensure utterance is never empty
        utterance = parsed.utterance
        if not utterance or not utterance.strip():
            utterance = "Let's think about this more carefully."
        
        return TeacherTurn(
            thought=parsed.thought or "(No thought provided)",
            action=action,
            utterance=utterance,
            is_terminal=is_terminal_action(action),
            raw_output=raw,
            parse_errors=parsed.validation_errors,
        )
    
    def _build_messages(
        self,
        question: Question,
        initial_student_turn: StudentTurn,
        history: list[tuple[str, str]],
        turn_number: int,
        max_turns: int,
        judge_verdict: Optional[bool] = None,
        student_final_answer: Optional[str] = None,
    ) -> list[dict]:
        """Build message list with judge verdict context."""
        context = get_teacher_context(
            question=question,
            turn_number=turn_number,
            max_turns=max_turns,
            judge_verdict=judge_verdict,
            student_final_answer=student_final_answer,
        )
        
        system_content = f"{TEACHER_SYSTEM_PROMPT}\n\n{context}"
        messages = [{"role": "system", "content": system_content}]
        
        # Initial student attempt
        messages.append({"role": "user", "content": initial_student_turn.raw_text})
        
        # Dialogue history
        for speaker, msg in history:
            if speaker == "teacher":
                messages.append({"role": "assistant", "content": msg})
            else:
                messages.append({"role": "user", "content": msg})
        
        return messages
    
    def force_terminal_turn(
        self,
        question: Question,
        initial_student_turn: StudentTurn,
        history: list[tuple[str, str]],
        latest_student_turn: Optional[StudentTurn] = None,
        judge_verdict: Optional[bool] = None,
    ) -> TeacherTurn:
        """Force session end when max turns reached."""
        student_final = None
        if latest_student_turn and latest_student_turn.final_answer:
            student_final = latest_student_turn.final_answer
        
        context = f"""---
PROBLEM: {question.question}
CORRECT ANSWER: {question.ground_truth}
⚠️ TURN LIMIT REACHED - You MUST end this session.
"""
        if student_final:
            verdict_str = "CORRECT" if judge_verdict else "INCORRECT"
            context += f"\nSTUDENT'S LAST ANSWER: {student_final}\nJUDGE VERDICT: {verdict_str}"
        context += "\n---"
        
        messages = [
            {"role": "system", "content": f"{TEACHER_SYSTEM_PROMPT}\n\n{context}"},
            {"role": "user", "content": initial_student_turn.raw_text},
        ]
        
        for speaker, msg in history:
            if speaker == "teacher":
                messages.append({"role": "assistant", "content": msg})
            else:
                messages.append({"role": "user", "content": msg})
        
        messages.append({
            "role": "user",
            "content": "[SYSTEM: Session ending. Use END_SUCCESS if judge said CORRECT, otherwise END_STUCK.]",
        })
        
        raw = self.llm.chat(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        parsed = parse_teacher_response(raw)
        
        # Determine correct terminal action based on judge
        if judge_verdict is True:
            action = "END_SUCCESS"
        else:
            action = "END_STUCK"
        
        utterance = parsed.utterance
        if not utterance or not utterance.strip():
            if action == "END_SUCCESS":
                utterance = "Great work! You've solved the problem correctly."
            else:
                utterance = "Let's stop here. Review the concepts we discussed."
        
        return TeacherTurn(
            thought=parsed.thought or "(Session ended - turn limit)",
            action=action,
            utterance=utterance,
            is_terminal=True,
            raw_output=raw,
            parse_errors=[],
        )
    
    @classmethod
    def from_config(cls, config: TeacherConfig, llm_client: LLMClient) -> "TeacherAgent":
        return cls(llm_client=llm_client, config=config)
