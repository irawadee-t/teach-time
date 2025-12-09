"""
Multi-turn Socratic Tutoring Interaction Loop

Architecture:
1. Student attempts question
2. If student submits FINAL → Judge evaluates → Verdict passed to Teacher
3. Teacher reasons on observation + verdict → chooses action
4. Loop until terminal action or max turns

Key principle: Teacher receives judge verdict to make informed decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import random
from collections import Counter

from ..config import (
    TeacherConfig, StudentConfig, EnvironmentConfig, ExperimentConfig,
    SamplingConfig,
)
from ..datasets import Question
from ..agents import StudentAgent, TeacherAgent, StudentTurn, TeacherTurn
from ..agents.student_agent import DialogueContext
from ..judge import ExternalJudge, JudgeVerdict, extract_final_answer


@dataclass
class InteractionTurn:
    """A single turn in the tutoring interaction."""
    
    turn_number: int
    speaker: str  # "teacher" or "student"
    
    # Teacher data (ReAct format)
    teacher_thought: Optional[str] = None
    teacher_action: Optional[str] = None
    teacher_utterance: Optional[str] = None
    teacher_is_terminal: bool = False
    
    # Student data
    student_text: Optional[str] = None
    student_final_answer: Optional[str] = None
    
    # External judge verdict (when FINAL submitted)
    judge_verdict: Optional[bool] = None
    judge_raw_response: Optional[str] = None
    
    @classmethod
    def from_teacher(cls, turn_number: int, turn: TeacherTurn) -> "InteractionTurn":
        return cls(
            turn_number=turn_number,
            speaker="teacher",
            teacher_thought=turn.thought,
            teacher_action=turn.action,
            teacher_utterance=turn.utterance,
            teacher_is_terminal=turn.is_terminal,
        )
    
    @classmethod
    def from_student(
        cls,
        turn_number: int,
        turn: StudentTurn,
        judge_verdict: Optional[JudgeVerdict] = None,
    ) -> "InteractionTurn":
        return cls(
            turn_number=turn_number,
            speaker="student",
            student_text=turn.raw_text,
            student_final_answer=turn.final_answer,
            judge_verdict=judge_verdict.is_correct if judge_verdict else None,
            judge_raw_response=judge_verdict.raw_response if judge_verdict else None,
        )


@dataclass
class InteractionSession:
    """Complete record of a tutoring session."""
    
    session_id: str
    timestamp: str
    question: Question
    
    # Initial attempt
    initial_student_text: str
    initial_final_answer: Optional[str]
    initial_judge_verdict: Optional[bool]
    initial_judge_raw: Optional[str]
    
    # Dialogue
    turns: list[InteractionTurn] = field(default_factory=list)
    
    # Outcome
    total_turns: int = 0
    final_action: Optional[str] = None
    final_student_answer: Optional[str] = None
    final_judge_verdict: Optional[bool] = None
    terminated_naturally: bool = False
    forced_termination: bool = False
    
    # Config
    teacher_config_name: Optional[str] = None
    student_config_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "question": {
                "question_id": self.question.question_id,
                "question": self.question.question,
                "ground_truth": self.question.ground_truth,
                "category": self.question.category,
            },
            "initial_attempt": {
                "text": self.initial_student_text,
                "final_answer": self.initial_final_answer,
                "judge_verdict": self.initial_judge_verdict,
                "judge_raw": self.initial_judge_raw,
            },
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "speaker": t.speaker,
                    "teacher_thought": t.teacher_thought,
                    "teacher_action": t.teacher_action,
                    "teacher_utterance": t.teacher_utterance,
                    "teacher_is_terminal": t.teacher_is_terminal,
                    "student_text": t.student_text,
                    "student_final_answer": t.student_final_answer,
                    "judge_verdict": t.judge_verdict,
                    "judge_raw": t.judge_raw_response,
                }
                for t in self.turns
            ],
            "outcome": {
                "total_turns": self.total_turns,
                "final_action": self.final_action,
                "final_student_answer": self.final_student_answer,
                "final_judge_verdict": self.final_judge_verdict,
                "terminated_naturally": self.terminated_naturally,
                "forced_termination": self.forced_termination,
            },
            "config": {
                "teacher": self.teacher_config_name,
                "student": self.student_config_name,
            },
        }


class InteractionLoop:
    """
    Runs multi-turn Socratic tutoring.
    
    Key flow:
    1. Student responds
    2. If FINAL submitted → Judge evaluates
    3. Teacher receives student response + judge verdict
    4. Teacher decides action based on full observation
    """
    
    def __init__(
        self,
        teacher: TeacherAgent,
        student: StudentAgent,
        judge: ExternalJudge,
        config: EnvironmentConfig | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.judge = judge
        self.config = config or EnvironmentConfig()
        self._session_counter = 0
    
    def run(self, question: Question) -> InteractionSession:
        """Run a complete tutoring interaction."""
        self.teacher.reset()
        
        # Session setup
        self._session_counter += 1
        session_id = f"session_{self._session_counter}_{question.question_id}"
        timestamp = datetime.now().isoformat()
        
        # === Phase 1: Student initial attempt ===
        initial_turn = self.student.answer_initial_question(question)
        initial_final = extract_final_answer(initial_turn.raw_text)
        
        # Judge initial attempt if FINAL submitted
        initial_verdict = None
        if initial_final:
            initial_verdict = self.judge.evaluate(
                question=question.question,
                ground_truth=question.ground_truth,
                student_answer=initial_final,
            )
        
        # Initialize session
        session = InteractionSession(
            session_id=session_id,
            timestamp=timestamp,
            question=question,
            initial_student_text=initial_turn.raw_text,
            initial_final_answer=initial_final,
            initial_judge_verdict=initial_verdict.is_correct if initial_verdict else None,
            initial_judge_raw=initial_verdict.raw_response if initial_verdict else None,
            teacher_config_name=self.teacher.config.name,
            student_config_name=self.student.config.name,
        )
        
        # State tracking
        history: list[tuple[str, str]] = []
        latest_student_turn: Optional[StudentTurn] = initial_turn
        latest_final_answer: Optional[str] = initial_final
        latest_judge_verdict: Optional[bool] = (
            initial_verdict.is_correct if initial_verdict else None
        )
        
        # === Phase 2: Multi-turn dialogue ===
        max_turns = self.config.max_teacher_turns
        
        for turn_num in range(1, max_turns + 1):
            # --- Teacher turn ---
            # Teacher receives judge verdict if student submitted FINAL
            current_judge_verdict = None
            if latest_student_turn and latest_student_turn.final_answer:
                current_judge_verdict = latest_judge_verdict
            
            teacher_turn = self.teacher.step(
                question=question,
                initial_student_turn=initial_turn,
                history=history,
                turn_number=turn_num,
                max_turns=max_turns,
                latest_student_turn=latest_student_turn,
                judge_verdict=current_judge_verdict,  # Pass verdict to teacher
            )
            
            # Record teacher turn
            session.turns.append(InteractionTurn.from_teacher(turn_num, teacher_turn))
            session.total_turns = turn_num
            session.final_action = teacher_turn.action
            
            # Update history with utterance
            history.append(("teacher", teacher_turn.utterance))
            
            # Check for natural termination
            if teacher_turn.is_terminal:
                session.terminated_naturally = True
                break
            
            # --- Student turn ---
            context = DialogueContext(
                question=question,
                initial_student_turn=initial_turn,
                history=history,
            )
            
            student_turn = self.student.respond_in_dialogue(context)
            latest_student_turn = student_turn
            
            # Check for FINAL answer
            final_answer = extract_final_answer(student_turn.raw_text)
            
            # Judge if FINAL submitted
            judge_verdict = None
            if final_answer:
                judge_verdict = self.judge.evaluate(
                    question=question.question,
                    ground_truth=question.ground_truth,
                    student_answer=final_answer,
                )
                latest_final_answer = final_answer
                latest_judge_verdict = judge_verdict.is_correct
            
            # Record student turn
            session.turns.append(InteractionTurn.from_student(
                turn_num,
                student_turn,
                judge_verdict=judge_verdict,
            ))
            
            # Update history
            history.append(("student", student_turn.raw_text))
        
        # === Phase 3: Handle max turns ===
        if not session.terminated_naturally:
            session.forced_termination = True
            
            forced_turn = self.teacher.force_terminal_turn(
                question=question,
                initial_student_turn=initial_turn,
                history=history,
                latest_student_turn=latest_student_turn,
                judge_verdict=latest_judge_verdict,  # Pass verdict
            )
            
            session.turns.append(InteractionTurn.from_teacher(
                session.total_turns + 1,
                forced_turn,
            ))
            session.final_action = forced_turn.action
        
        # Record final state
        session.final_student_answer = latest_final_answer
        session.final_judge_verdict = latest_judge_verdict
        
        return session


def create_interaction_loop(
    teacher_config: TeacherConfig,
    student_config: StudentConfig,
    llm_client,
    env_config: EnvironmentConfig | None = None,
) -> InteractionLoop:
    """Factory function to create interaction loop."""
    teacher = TeacherAgent(llm_client=llm_client, config=teacher_config)
    student = StudentAgent(llm_client=llm_client, config=student_config)
    judge = ExternalJudge(llm_client=llm_client)
    
    return InteractionLoop(
        teacher=teacher,
        student=student,
        judge=judge,
        config=env_config,
    )


# =============================================================================
# Question Sampling
# =============================================================================

def sample_questions(
    questions: list[Question],
    config: SamplingConfig,
) -> list[Question]:
    """Sample questions with balanced category representation."""
    rng = random.Random(config.seed)
    
    by_category: dict[str, list[Question]] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)
    
    available_counts = {cat: len(qs) for cat, qs in by_category.items()}
    target_sizes = config.compute_sample_sizes(available_counts)
    
    sampled: list[Question] = []
    for cat, target_n in target_sizes.items():
        if cat not in by_category:
            continue
        cat_questions = by_category[cat].copy()
        rng.shuffle(cat_questions)
        sampled.extend(cat_questions[:min(target_n, len(cat_questions))])
    
    rng.shuffle(sampled)
    return sampled


def get_sampling_summary(questions: list[Question], config: SamplingConfig) -> dict:
    """Get summary of sampled questions."""
    category_counts = Counter(q.category for q in questions)
    return {
        "total_questions": len(questions),
        "num_categories": len(category_counts),
        "questions_per_category": dict(category_counts),
        "config": config.to_dict(),
    }


# =============================================================================
# Experiment Output
# =============================================================================

@dataclass
class ExperimentOutput:
    """Complete experiment output with metadata and sessions."""
    
    config: dict
    sampling_summary: dict
    sessions: list[InteractionSession]
    
    start_time: str
    end_time: str = ""
    total_duration_seconds: float = 0.0
    
    stats: dict = field(default_factory=dict)
    
    def compute_stats(self):
        """Compute summary statistics."""
        if not self.sessions:
            self.stats = {}
            return
        
        n = len(self.sessions)
        
        initial_correct = sum(
            1 for s in self.sessions if s.initial_judge_verdict is True
        )
        final_correct = sum(
            1 for s in self.sessions if s.final_judge_verdict is True
        )
        natural = sum(1 for s in self.sessions if s.terminated_naturally)
        forced = sum(1 for s in self.sessions if s.forced_termination)
        total_turns = sum(s.total_turns for s in self.sessions)
        action_counts = Counter(s.final_action for s in self.sessions)
        
        self.stats = {
            "num_sessions": n,
            "initial_correct": initial_correct,
            "initial_correct_rate": initial_correct / n if n > 0 else 0,
            "final_correct": final_correct,
            "final_correct_rate": final_correct / n if n > 0 else 0,
            "improvement": final_correct - initial_correct,
            "terminated_naturally": natural,
            "forced_termination": forced,
            "avg_turns": total_turns / n if n > 0 else 0,
            "final_action_distribution": dict(action_counts),
            "categories": dict(Counter(s.question.category for s in self.sessions)),
        }
    
    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "config": self.config,
            "sampling_summary": self.sampling_summary,
            "runtime": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration_seconds": self.total_duration_seconds,
            },
            "stats": self.stats,
            "sessions": [s.to_dict() for s in self.sessions],
        }
    
    def save(self, output_dir: Path, experiment_id: str) -> dict[str, str]:
        """Save experiment output to disk."""
        import yaml
        
        experiment_dir = Path(output_dir) / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        summary_path = experiment_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "experiment_id": experiment_id,
                "config": self.config,
                "sampling_summary": self.sampling_summary,
                "runtime": {
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "total_duration_seconds": self.total_duration_seconds,
                },
                "stats": self.stats,
            }, f, indent=2)
        
        results_path = experiment_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return {
            "folder": str(experiment_dir),
            "config": str(config_path),
            "summary": str(summary_path),
            "results": str(results_path),
        }


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs complete experiments with progress tracking."""
    
    def __init__(self, config: ExperimentConfig, llm_client):
        self.config = config
        self.llm_client = llm_client
        
        self.teacher = TeacherAgent(llm_client=llm_client, config=config.teacher_config)
        self.student = StudentAgent(llm_client=llm_client, config=config.student_config)
        self.judge = ExternalJudge(llm_client=llm_client)
        
        self.loop = InteractionLoop(
            teacher=self.teacher,
            student=self.student,
            judge=self.judge,
            config=config.env_config,
        )
    
    def run(
        self,
        questions: list[Question],
        show_progress: bool = True,
    ) -> ExperimentOutput:
        """Run the experiment."""
        from tqdm import tqdm
        
        start_time = datetime.now()
        
        sampling_config = self.config.env_config.sampling
        sampled = sample_questions(questions, sampling_config)
        sampling_summary = get_sampling_summary(sampled, sampling_config)
        
        output = ExperimentOutput(
            config=self.config.to_dict(),
            sampling_summary=sampling_summary,
            sessions=[],
            start_time=start_time.isoformat(),
        )
        
        iterator = tqdm(sampled, desc="Teaching", unit="q") if show_progress else sampled
        
        for question in iterator:
            session = self.loop.run(question)
            output.sessions.append(session)
            
            if show_progress:
                status = "✓" if session.final_judge_verdict else "✗"
                iterator.set_postfix(turns=session.total_turns, correct=status)
        
        end_time = datetime.now()
        output.end_time = end_time.isoformat()
        output.total_duration_seconds = (end_time - start_time).total_seconds()
        output.compute_stats()
        
        return output
