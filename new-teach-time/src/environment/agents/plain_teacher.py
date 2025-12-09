"""
Plain teacher agent WITHOUT ReAct framework.

This teacher generates direct tutoring responses without explicit
Thought → Action → Utterance structure. Used as a baseline to
measure the effect of the ReAct framework.

The teacher:
- Knows the ground truth (for pedagogical guidance)
- Receives judge verdict when student submits FINAL
- Generates natural tutoring responses
- Can be used with base or fine-tuned models
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ..config import TeacherConfig
from ..datasets import Question
from ..llm_client import LLMClient
from .student_agent import StudentTurn


@dataclass
class PlainTeacherTurn:
    """Output from a plain teacher turn (no ReAct)."""
    utterance: str
    is_terminal: bool
    raw_output: str
    action: str = "PLAIN_RESPONSE"  # Default action for plain mode
    thought: Optional[str] = None   # No thought in plain mode
    
    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "utterance": self.utterance,
            "is_terminal": self.is_terminal,
            "raw_output": self.raw_output,
            "parse_errors": [],
        }


PLAIN_TEACHER_SYSTEM_PROMPT = """You are a Socratic tutor. Guide students to discover answers through hints and questions.

## Your Role
- Help the student work through the problem step by step
- Ask guiding questions rather than giving direct answers
- Provide hints when the student is stuck
- Correct errors with clear explanations
- Be encouraging and supportive

## Rules
1. NEVER reveal the answer directly
2. Use questions to lead the student toward understanding
3. If the student submits a FINAL answer and it's CORRECT (judge confirmed), congratulate them and end with "[END_SUCCESS]"
4. If the student is stuck and cannot progress, you may end with "[END_STUCK]"
5. Keep responses concise (2-4 sentences)

## Format
Just respond naturally as a tutor would. No special formatting required."""


def get_plain_teacher_context(
    question: Question,
    turn_number: int,
    max_turns: int,
    judge_verdict: Optional[bool] = None,
    student_final_answer: Optional[str] = None,
) -> str:
    """Generate context block for plain teacher."""
    context = f"""---
PROBLEM: {question.question}
CORRECT ANSWER: {question.ground_truth}
[Turn {turn_number}/{max_turns}]"""
    
    if student_final_answer is not None:
        verdict_str = "CORRECT" if judge_verdict else "INCORRECT"
        context += f"""

STUDENT SUBMITTED: FINAL: {student_final_answer}
JUDGE VERDICT: {verdict_str}

If CORRECT: Congratulate and end with [END_SUCCESS]
If INCORRECT: Help them understand the error."""
    
    if turn_number >= max_turns - 1:
        context += "\n⚠️ APPROACHING TURN LIMIT - wrap up soon."
    
    context += "\n---"
    return context


class PlainTeacherAgent:
    """
    Plain teacher agent without ReAct framework.
    
    Generates natural tutoring responses without explicit
    reasoning structure. Used as baseline for ablation studies.
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
    ) -> PlainTeacherTurn:
        """
        Execute one teacher turn (plain mode).
        """
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
        
        # Check for terminal markers and determine action
        if "[END_SUCCESS]" in raw:
            is_terminal = True
            action = "END_SUCCESS"
        elif "[END_STUCK]" in raw:
            is_terminal = True
            action = "END_STUCK"
        else:
            is_terminal = False
            action = "TUTOR_RESPONSE"
        
        # Clean the response (remove terminal markers from utterance)
        utterance = raw.replace("[END_SUCCESS]", "").replace("[END_STUCK]", "").strip()
        
        # Ensure non-empty utterance
        if not utterance:
            if action == "END_SUCCESS":
                utterance = "Great work! You've solved the problem correctly."
            else:
                utterance = "Let's think about this differently."
        
        return PlainTeacherTurn(
            utterance=utterance,
            is_terminal=is_terminal,
            raw_output=raw,
            action=action,
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
        """Build message list for plain teacher."""
        context = get_plain_teacher_context(
            question=question,
            turn_number=turn_number,
            max_turns=max_turns,
            judge_verdict=judge_verdict,
            student_final_answer=student_final_answer,
        )
        
        system_content = f"{PLAIN_TEACHER_SYSTEM_PROMPT}\n\n{context}"
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
    ) -> PlainTeacherTurn:
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
            {"role": "system", "content": f"{PLAIN_TEACHER_SYSTEM_PROMPT}\n\n{context}"},
            {"role": "user", "content": initial_student_turn.raw_text},
        ]
        
        for speaker, msg in history:
            if speaker == "teacher":
                messages.append({"role": "assistant", "content": msg})
            else:
                messages.append({"role": "user", "content": msg})
        
        messages.append({
            "role": "user",
            "content": "[SYSTEM: Session ending. Provide a brief wrap-up.]",
        })
        
        raw = self.llm.chat(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Determine action based on verdict
        action = "END_SUCCESS" if judge_verdict else "END_STUCK"
        
        # Determine utterance based on verdict
        utterance = raw.replace("[END_SUCCESS]", "").replace("[END_STUCK]", "").strip()
        if not utterance:
            if judge_verdict:
                utterance = "Great work! You've solved the problem correctly."
            else:
                utterance = "Let's stop here. Review the concepts we discussed."
        
        return PlainTeacherTurn(
            utterance=utterance,
            is_terminal=True,
            raw_output=raw,
            action=action,
        )
    
    @classmethod
    def from_config(cls, config: TeacherConfig, llm_client: LLMClient) -> "PlainTeacherAgent":
        return cls(llm_client=llm_client, config=config)

