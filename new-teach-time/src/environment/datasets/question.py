"""
Standard question dataclass for open-ended teaching evaluation.

Key Design Decision:
Questions are stored in OPEN-ENDED format with ground-truth answers.
MCQ options are NOT shown to student or teacher during interaction.
This prevents:
- Student guess/elimination strategies
- Teacher "teaching to the test"
- MCQ-specific reasoning artifacts

The ground truth is used ONLY for:
- Correctness evaluation
- Teacher's internal reference (hidden from student)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class Question:
    """
    Open-ended question format for teaching evaluation.
    
    This is the format used during teaching interactions.
    MCQ options (if originally present) are stripped out.
    """
    
    question_id: str
    question: str                    # The question text (open-ended, no options)
    ground_truth: str                # The correct answer (text form)
    ground_truth_explanation: str    # Explanation/solution if available
    category: str                    # Subject/domain
    difficulty: Optional[str] = None # Easy/Medium/Hard if available
    source_dataset: str = "mmlu_pro" # Original dataset
    
    # Original MCQ data (kept for reference but NOT used in interaction)
    _original_options: Optional[tuple[str, ...]] = None
    _original_answer_index: Optional[int] = None
    
    def check_answer(self, student_answer: str, strict: bool = False) -> bool:
        """
        Check if student's answer is correct.
        
        For open-ended evaluation, we check if the answer matches
        the ground truth semantically (not just string match).
        
        Args:
            student_answer: Student's answer text
            strict: If True, require exact match. If False, allow flexible matching.
            
        Returns:
            True if answer is correct
        """
        if not student_answer:
            return False
        
        # Normalize both answers
        student_norm = self._normalize_answer(student_answer)
        truth_norm = self._normalize_answer(self.ground_truth)
        
        # Exact match
        if student_norm == truth_norm:
            return True
        
        # If not strict, try flexible matching
        if not strict:
            # Check if student answer contains the ground truth
            if truth_norm in student_norm:
                return True
            
            # Check numeric equivalence
            student_num = self._extract_number(student_answer)
            truth_num = self._extract_number(self.ground_truth)
            if student_num is not None and truth_num is not None:
                # Allow small floating point tolerance
                if abs(student_num - truth_num) < 0.01:
                    return True
        
        return False
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        # Lowercase, strip whitespace
        text = text.lower().strip()
        # Remove common prefixes
        for prefix in ["the answer is", "answer:", "answer is", "i think"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        # Remove punctuation at end
        text = text.rstrip(".,;:!?")
        return text
    
    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Extract a number from text if present."""
        # Look for numbers (including negative, decimals, scientific notation)
        match = re.search(r'-?\d+\.?\d*(?:e[+-]?\d+)?', text.lower())
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None
    
    def format_for_student(self) -> str:
        """
        Format question for student - NO OPTIONS shown.
        
        This is what the student sees during the teaching interaction.
        """
        return f"Question: {self.question}"
    
    def format_for_teacher(self, include_ground_truth: bool = True) -> str:
        """
        Format question for teacher.
        
        Teacher sees the question and (optionally) the ground truth answer
        for reference, but NO MCQ options.
        
        Args:
            include_ground_truth: Whether to include the answer (for teacher's reference)
        """
        lines = [f"Question: {self.question}"]
        
        if include_ground_truth:
            lines.append(f"\n[GROUND TRUTH - DO NOT REVEAL]: {self.ground_truth}")
            if self.ground_truth_explanation:
                lines.append(f"[SOLUTION APPROACH]: {self.ground_truth_explanation[:200]}...")
        
        return "\n".join(lines)
    
    @classmethod
    def from_mcq(
        cls,
        question_id: str,
        question_text: str,
        options: tuple[str, ...],
        answer_index: int,
        category: str,
        cot_content: str | None = None,
        source_dataset: str = "mmlu_pro",
    ) -> "Question":
        """
        Convert an MCQ question to open-ended format.
        
        The MCQ options are stripped and the correct answer text
        becomes the ground truth.
        
        Args:
            question_id: Unique identifier
            question_text: The question text
            options: Original MCQ options
            answer_index: Index of correct answer in options
            category: Subject category
            cot_content: Chain-of-thought explanation if available
            source_dataset: Source dataset name
            
        Returns:
            Question in open-ended format
        """
        # Get the correct answer text
        correct_answer_text = options[answer_index]
        
        # Clean the question text (remove any embedded option references)
        clean_question = cls._clean_question_text(question_text)
        
        return cls(
            question_id=question_id,
            question=clean_question,
            ground_truth=correct_answer_text,
            ground_truth_explanation=cot_content or "",
            category=category,
            source_dataset=source_dataset,
            _original_options=options,
            _original_answer_index=answer_index,
        )
    
    @staticmethod
    def _clean_question_text(text: str) -> str:
        """
        Clean question text to remove MCQ artifacts.
        
        Removes things like:
        - "Which of the following..."
        - Option markers (A), (B), etc.
        """
        # Remove "Which of the following" type phrases
        text = re.sub(
            r'\b(which of the following|select the correct|choose the)\b',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Remove trailing option references
        text = re.sub(r'\s*\([A-J]\)\s*', ' ', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()


# Legacy alias for backwards compatibility
QuestionMCQ = Question
