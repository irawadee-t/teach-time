"""
External Judge for Answer Evaluation

Hybrid approach:
1. First: Python numeric comparison (fast, deterministic)
2. Fallback: LLM semantic comparison (handles expressions, sequences, etc.)

Design: Judge outputs ONLY "CORRECT" or "INCORRECT" - no reasoning.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Optional


JUDGE_MODEL_ID = "deepseek-ai/DeepSeek-V3"


@dataclass
class JudgeVerdict:
    """Result from judge evaluation."""
    is_correct: bool
    raw_response: str
    method: str  # "numeric" or "llm"
    model_id: str = JUDGE_MODEL_ID


class ExternalJudge:
    """
    Hybrid judge: numeric comparison first, LLM fallback.
    
    Handles:
    - Numeric equivalence (0.0555 ≈ 5.5e-2)
    - Scientific notation
    - Percentage/decimal (10% = 0.1)
    - Expression evaluation (10π ≈ 31.4)
    - Symbolic equivalence via LLM
    """
    
    def __init__(self, llm_client, tolerance: float = 0.01):
        self.llm_client = llm_client
        self.model_id = JUDGE_MODEL_ID
        self.tolerance = tolerance  # 1% default
    
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        student_answer: str,
    ) -> JudgeVerdict:
        """
        Evaluate if student's answer matches ground truth.
        
        1. Try numeric comparison first (fast, deterministic)
        2. Fall back to LLM for semantic equivalence
        """
        if not student_answer:
            return JudgeVerdict(
                is_correct=False,
                raw_response="No answer provided",
                method="none",
            )
        
        # Step 1: Try numeric comparison
        numeric_result = self._numeric_compare(ground_truth, student_answer)
        if numeric_result is not None:
            return JudgeVerdict(
                is_correct=numeric_result,
                raw_response=f"Numeric comparison: {'CORRECT' if numeric_result else 'INCORRECT'}",
                method="numeric",
            )
        
        # Step 2: Fall back to LLM
        return self._llm_compare(question, ground_truth, student_answer)
    
    def _numeric_compare(self, ground_truth: str, student_answer: str) -> Optional[bool]:
        """
        Try to compare as numbers. Returns None if not comparable numerically.
        """
        gt_num = self._parse_number(ground_truth)
        st_num = self._parse_number(student_answer)
        
        if gt_num is None or st_num is None:
            return None
        
        # Handle zero specially
        if gt_num == 0:
            return abs(st_num) < 1e-6
        
        # Relative tolerance
        rel_diff = abs(gt_num - st_num) / abs(gt_num)
        return rel_diff < self.tolerance
    
    def _parse_number(self, text: str) -> Optional[float]:
        """
        Parse numeric value from text. Handles:
        - Plain numbers: "31.4", "-7.5"
        - Scientific notation: "5.5e-2", "5.5 × 10^-2"
        - Percentages: "10%"
        - Pi expressions: "10π", "10*pi"
        - Common units stripped
        """
        if not text:
            return None
        
        # Clean the text
        s = text.strip().lower()
        
        # Remove common units (order matters - longer first)
        units = ['seconds', 'second', 'meters', 'meter', 'sec', 'ohm', 
                 'mm', 'cm', 'km', 'kg', 'nc', 'μc', 'mc', 'hz',
                 's', 'm', 'g', 'n', 'j', 'w', 'v', 'a', 'ω']
        for unit in units:
            s = re.sub(rf'\s*{unit}\s*$', '', s, flags=re.IGNORECASE)
        
        # Remove $ and other formatting
        s = s.replace('$', '').replace(',', '').replace(' ', '').strip()
        
        # Handle percentage
        if s.endswith('%'):
            try:
                return float(s[:-1]) / 100
            except:
                pass
        
        # Handle pi expressions: 10π, 10*pi, 10pi
        pi_match = re.match(r'^(-?\d*\.?\d*)\s*[*×]?\s*(π|pi)$', s)
        if pi_match:
            coef = pi_match.group(1)
            if coef in ['', '-']:
                coef = coef + '1'
            try:
                return float(coef) * math.pi
            except:
                pass
        
        # Handle scientific notation: 5.5×10^-2, 5.5*10^-2, 5.5e-2
        sci_match = re.match(r'^(-?\d*\.?\d+)\s*[×*]\s*10\^?\s*(-?\d+)$', s)
        if sci_match:
            try:
                base = float(sci_match.group(1))
                exp = int(sci_match.group(2))
                return base * (10 ** exp)
            except:
                pass
        
        # Plain number
        try:
            return float(s)
        except:
            return None
    
    def _llm_compare(self, question: str, ground_truth: str, student_answer: str) -> JudgeVerdict:
        """
        Use LLM for semantic comparison when numeric comparison fails.
        """
        prompt = f"""You are an automatic answer checker.

Question: {question}

Ground truth answer: {ground_truth}

Student's FINAL answer: {student_answer}

Task: Decide if the student's answer is mathematically/semantically equivalent to the ground truth.

Rules:
- Ignore formatting, whitespace, units
- Accept equivalent expressions (e.g., 1+2^n and 3,5,9,17,... for same sequence)
- Accept equivalent numeric forms (0.0555 = 5.5×10^-2)
- Accept reasonable rounding (3.14 ≈ π)
- Only judge the FINAL answer, not reasoning

Return ONLY one word:
CORRECT
or
INCORRECT"""

        response = self.llm_client.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,  # Force short response
        )
        
        response_clean = response.strip().upper()
        is_correct = response_clean.startswith("CORRECT") and not response_clean.startswith("INCORRECT")
        
        return JudgeVerdict(
            is_correct=is_correct,
            raw_response=response.strip(),
            method="llm",
        )
    
    def evaluate_batch(
        self,
        evaluations: list[tuple[str, str, str]],
    ) -> list[JudgeVerdict]:
        """Evaluate multiple answers."""
        return [self.evaluate(q, gt, sa) for q, gt, sa in evaluations]


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract answer from FINAL: format.
    Returns text after the LAST "FINAL:" in the response.
    """
    if not text:
        return None
    
    text_upper = text.upper()
    idx = text_upper.rfind("FINAL:")
    
    if idx == -1:
        return None
    
    after_final = text[idx + 6:].strip()
    first_line = after_final.split('\n')[0].strip()
    first_line = first_line.rstrip('.,;:')
    
    return first_line if first_line else None
