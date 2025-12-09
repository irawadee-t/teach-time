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
from ..agents import PlainTeacherAgent, PlainTeacherTurn, create_teacher
from ..agents.student_agent import DialogueContext
from ..judge import ExternalJudge, JudgeVerdict, extract_final_answer

# Type alias for teacher turn (can be either ReAct or Plain)
from typing import Union
AnyTeacherTurn = Union[TeacherTurn, PlainTeacherTurn]


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
    def from_teacher(cls, turn_number: int, turn: AnyTeacherTurn) -> "InteractionTurn":
        """Create from either TeacherTurn (ReAct) or PlainTeacherTurn."""
        # Handle both ReAct and Plain teacher turns
        thought = getattr(turn, 'thought', None)
        action = getattr(turn, 'action', 'PLAIN_RESPONSE')
        
        return cls(
            turn_number=turn_number,
            speaker="teacher",
            teacher_thought=thought,
            teacher_action=action,
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
    
    def to_judge_format(self) -> dict:
        """
        Convert session to format expected by PedagogicalEvaluator.
        
        Returns a dict with:
        - metadata: session info
        - conversation: list of {"role": "tutor"|"student", "content": "..."}
        """
        conversation = []
        
        # Add initial student attempt
        conversation.append({
            "role": "student",
            "content": self.initial_student_text,
        })
        
        # Add dialogue turns
        for turn in self.turns:
            if turn.speaker == "teacher" and turn.teacher_utterance:
                conversation.append({
                    "role": "tutor",
                    "content": turn.teacher_utterance,
                })
            elif turn.speaker == "student" and turn.student_text:
                conversation.append({
                    "role": "student",
                    "content": turn.student_text,
                })
        
        return {
            "metadata": {
                "session_id": self.session_id,
                "name": f"session_{self.question.question_id}",
                "description": f"Tutoring session for: {self.question.question[:100]}...",
                "domain": self.question.category,
                "topic": self.question.category,
                "question_id": self.question.question_id,
                "ground_truth": self.question.ground_truth,
                "initial_correct": self.initial_judge_verdict,
                "final_correct": self.final_judge_verdict,
                "total_turns": self.total_turns,
                "terminated_naturally": self.terminated_naturally,
            },
            "conversation": conversation,
        }
    
    def save_for_judge(self, output_path: Path) -> Path:
        """
        Save session in judge-compatible format.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.to_judge_format(), f, indent=2)
        
        return output_path


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
        
        # Save individual conversations for judge evaluation
        conversations_dir = experiment_dir / "conversations"
        conversations_dir.mkdir(parents=True, exist_ok=True)
        
        conversation_files = []
        for i, session in enumerate(self.sessions, 1):
            # Create filename from question_id
            safe_id = session.question.question_id.replace("/", "_").replace(" ", "_")
            conv_path = conversations_dir / f"{i:03d}_{safe_id}.json"
            session.save_for_judge(conv_path)
            conversation_files.append(str(conv_path))
        
        return {
            "folder": str(experiment_dir),
            "config": str(config_path),
            "summary": str(summary_path),
            "results": str(results_path),
            "conversations_dir": str(conversations_dir),
            "num_conversations": len(conversation_files),
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


# =============================================================================
# Ablation Runner (Multiple Variants)
# =============================================================================

class AblationRunner:
    """
    Runs ablation experiments with multiple teacher variants.
    
    Each variant is tested on the same set of questions for fair comparison.
    """
    
    def __init__(self, llm_client, student_config: StudentConfig = None):
        from ..config import DEFAULT_STUDENT_CONFIG
        
        self.llm_client = llm_client
        self.student_config = student_config or DEFAULT_STUDENT_CONFIG
        self.student = StudentAgent(llm_client=llm_client, config=self.student_config)
        self.judge = ExternalJudge(llm_client=llm_client)
    
    def run_variant(
        self,
        teacher_config,  # AblationTeacherConfig
        questions: list[Question],
        max_turns: int = 10,
        show_progress: bool = True,
    ) -> ExperimentOutput:
        """
        Run a single variant on the given questions.
        
        Args:
            teacher_config: AblationTeacherConfig with use_react flag
            questions: Pre-sampled questions (same for all variants)
            max_turns: Maximum teacher turns
            show_progress: Show progress bar
            
        Returns:
            ExperimentOutput with results
        """
        from tqdm import tqdm
        
        # Create appropriate teacher based on use_react flag
        use_react = getattr(teacher_config, 'use_react', True)
        teacher = create_teacher(
            config=teacher_config.to_teacher_config(),
            llm_client=self.llm_client,
            use_react=use_react,
        )
        
        # Create environment config
        env_config = EnvironmentConfig(max_teacher_turns=max_turns)
        
        # Create interaction loop
        loop = InteractionLoop(
            teacher=teacher,
            student=self.student,
            judge=self.judge,
            config=env_config,
        )
        
        # Run sessions
        start_time = datetime.now()
        
        output = ExperimentOutput(
            config={
                "teacher": teacher_config.to_dict(),
                "student": {
                    "name": self.student_config.name,
                    "model_id": self.student_config.model_id,
                },
                "max_turns": max_turns,
                "use_react": use_react,
            },
            sampling_summary={"total_questions": len(questions)},
            sessions=[],
            start_time=start_time.isoformat(),
        )
        
        desc = f"{teacher_config.name}"
        iterator = tqdm(questions, desc=desc, unit="q") if show_progress else questions
        
        for question in iterator:
            session = loop.run(question)
            output.sessions.append(session)
            
            if show_progress:
                status = "✓" if session.final_judge_verdict else "✗"
                iterator.set_postfix(turns=session.total_turns, ok=status)
        
        end_time = datetime.now()
        output.end_time = end_time.isoformat()
        output.total_duration_seconds = (end_time - start_time).total_seconds()
        output.compute_stats()
        
        return output
    
    def run_ablation(
        self,
        teacher_configs: list,  # List of AblationTeacherConfig
        questions: list[Question],
        max_turns: int = 10,
        output_dir: Path = None,
        experiment_name: str = "ablation",
        show_progress: bool = True,
    ) -> dict:
        """
        Run full ablation study with multiple variants.
        
        Args:
            teacher_configs: List of AblationTeacherConfig variants
            questions: Pre-sampled questions (same for all variants)
            max_turns: Maximum teacher turns
            output_dir: Where to save results
            experiment_name: Name for the experiment
            show_progress: Show progress bars
            
        Returns:
            Dict with all results and comparison summary
        """
        from datetime import datetime
        
        output_dir = Path(output_dir) if output_dir else Path("experiments/ablation")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_dir / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY: {experiment_name}")
        print(f"  Variants: {len(teacher_configs)}")
        print(f"  Questions: {len(questions)}")
        print(f"  Max turns: {max_turns}")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for i, teacher_config in enumerate(teacher_configs, 1):
            print(f"\n[{i}/{len(teacher_configs)}] Running: {teacher_config.name}")
            print(f"    Model: {teacher_config.model_id}")
            print(f"    ReAct: {teacher_config.use_react}")
            print()
            
            # Run this variant
            output = self.run_variant(
                teacher_config=teacher_config,
                questions=questions,
                max_turns=max_turns,
                show_progress=show_progress,
            )
            
            # Save variant results
            variant_dir = experiment_dir / teacher_config.name
            variant_id = f"{teacher_config.name}_{timestamp}"
            saved = output.save(variant_dir.parent, teacher_config.name)
            
            # Store results
            all_results[teacher_config.name] = {
                "config": teacher_config.to_dict(),
                "stats": output.stats,
                "duration": output.total_duration_seconds,
                "saved_to": saved,
            }
            
            # Print summary
            stats = output.stats
            print(f"    Initial correct: {stats.get('initial_correct_rate', 0)*100:.1f}%")
            print(f"    Final correct:   {stats.get('final_correct_rate', 0)*100:.1f}%")
            print(f"    Improvement:     {stats.get('improvement', 0):+d}")
            print(f"    Avg turns:       {stats.get('avg_turns', 0):.1f}")
        
        # Save comparison summary
        comparison = self._generate_comparison(all_results, questions)
        
        summary_path = experiment_dir / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n{'='*70}")
        print("  ABLATION COMPLETE")
        print(f"  Results saved to: {experiment_dir}")
        print(f"{'='*70}\n")
        
        return {
            "experiment_dir": str(experiment_dir),
            "variants": all_results,
            "comparison": comparison,
        }
    
    def _generate_comparison(self, results: dict, questions: list) -> dict:
        """Generate comparison summary across variants."""
        comparison = {
            "total_questions": len(questions),
            "variants": {},
            "rankings": {},
        }
        
        for name, data in results.items():
            stats = data["stats"]
            comparison["variants"][name] = {
                "model_id": data["config"]["model_id"],
                "use_react": data["config"]["use_react"],
                "is_finetuned": data["config"].get("is_finetuned", False),
                "initial_correct_rate": stats.get("initial_correct_rate", 0),
                "final_correct_rate": stats.get("final_correct_rate", 0),
                "improvement": stats.get("improvement", 0),
                "avg_turns": stats.get("avg_turns", 0),
                "duration_seconds": data["duration"],
            }
        
        # Rank by final correct rate
        ranked = sorted(
            comparison["variants"].items(),
            key=lambda x: x[1]["final_correct_rate"],
            reverse=True
        )
        comparison["rankings"]["by_final_accuracy"] = [
            {"rank": i+1, "name": name, "accuracy": data["final_correct_rate"]}
            for i, (name, data) in enumerate(ranked)
        ]
        
        # Rank by improvement
        ranked = sorted(
            comparison["variants"].items(),
            key=lambda x: x[1]["improvement"],
            reverse=True
        )
        comparison["rankings"]["by_improvement"] = [
            {"rank": i+1, "name": name, "improvement": data["improvement"]}
            for i, (name, data) in enumerate(ranked)
        ]
        
        return comparison
