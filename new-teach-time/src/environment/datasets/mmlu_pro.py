"""
MMLU-Pro dataset loader with MCQ → Open-Ended conversion.

STEM-ONLY configuration matching EducationQ methodology.

Key Design Decision:
- Only STEM categories (math, physics, chemistry, biology, CS, engineering)
- Questions converted to OPEN-ENDED format
- MCQ options stripped; ground truth kept for evaluation
- Difficulty stratification via EducationQ's pre-stratified set if available
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .question import Question


# =============================================================================
# Category Definitions (STEM Only)
# =============================================================================

# STEM categories used in EducationQ
STEM_CATEGORIES = (
    "math",
    "physics", 
    "chemistry",
    "biology",
    "computer science",
    "engineering",
)

# Full category list from MMLU-Pro (for reference)
ALL_CATEGORIES = (
    "biology",
    "business", 
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
)


# =============================================================================
# Loaders
# =============================================================================

def load_mmlu_pro_stem(
    from_educationq: bool = True,
    educationq_path: Optional[Path | str] = None,
    questions_per_category: Optional[int] = None,
) -> list[Question]:
    """
    Load MMLU-Pro STEM-only questions in open-ended format.
    
    Priority:
    1. If from_educationq=True, try to load from EducationQ's pre-stratified set
       (which has difficulty-balanced sampling)
    2. Otherwise, load from HuggingFace and filter to STEM
    
    Args:
        from_educationq: Try to load from EducationQ's stratified set first
        educationq_path: Path to EducationQ pretest results
        questions_per_category: Limit questions per category (None = all)
        
    Returns:
        List of Question objects (STEM only, open-ended format)
    """
    if from_educationq:
        # Default path to EducationQ stratified data
        if educationq_path is None:
            educationq_path = Path(
                "EducationQ-main/src/data/output/"
                "EduQ-Bench_Student-llama31-70b-instruct/MMLU-Pro-stratified/"
                "MMLU-Pro-strastified_Student-llama31-70b-instruct_pretest/"
                "MMLU-Pro-stratified_Student-llama31-70b-instruct_zero-shot_pretest-results_1.0.0_212612.json"
            )
        
        if Path(educationq_path).exists():
            questions = _load_from_educationq(educationq_path)
            # Filter to STEM
            questions = [q for q in questions if q.category in STEM_CATEGORIES]
            
            # Limit per category if requested
            if questions_per_category:
                questions = _limit_per_category(questions, questions_per_category)
            
            return questions
    
    # Fall back to HuggingFace
    return load_mmlu_pro_from_huggingface(
        categories=list(STEM_CATEGORIES),
        questions_per_category=questions_per_category,
    )


def _load_from_educationq(filepath: Path | str) -> list[Question]:
    """
    Load from EducationQ's pre-stratified pretest results.
    
    This data has been difficulty-stratified by EducationQ using
    top model accuracy scores (10 bins × 10 questions per bin per subject).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = []
    
    # Navigate to results
    for student_key, student_data in data.items():
        if not isinstance(student_data, dict) or "results" not in student_data:
            continue
        
        results = student_data["results"]
        
        for qid, qdata in results.items():
            if not isinstance(qdata, dict):
                continue
            
            category = qdata.get("category", "")
            responses = qdata.get("responses", [])
            
            if not responses:
                continue
            
            response = responses[0]
            question_text = response.get("question", "")
            options = tuple(response.get("options", []))
            correct_answer = response.get("correct_answer", "")
            answer_index = response.get("correct_answer_index", 0)
            
            if not question_text or not options:
                continue
            
            # Convert MCQ → Open-ended
            questions.append(Question.from_mcq(
                question_id=str(qid),
                question_text=question_text,
                options=options,
                answer_index=answer_index,
                category=category,
                cot_content=None,
                source_dataset="mmlu_pro_educationq_stratified",
            ))
    
    return questions


def _limit_per_category(
    questions: list[Question],
    limit: int,
) -> list[Question]:
    """Limit questions per category."""
    import random
    
    by_category: dict[str, list[Question]] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)
    
    result = []
    rng = random.Random(42)
    
    for category, cat_questions in by_category.items():
        if len(cat_questions) > limit:
            result.extend(rng.sample(cat_questions, limit))
        else:
            result.extend(cat_questions)
    
    return result


def load_mmlu_pro_from_huggingface(
    categories: list[str] | None = None,
    questions_per_category: Optional[int] = None,
    stratified: bool = False,
) -> list[Question]:
    """
    Load MMLU-Pro from HuggingFace in open-ended format.
    
    Note: This does NOT have EducationQ's difficulty stratification.
    For difficulty-balanced sampling, use load_mmlu_pro_stem() with
    from_educationq=True.
    
    Args:
        categories: Filter for specific categories (default: STEM only)
        questions_per_category: Limit per category
        stratified: Apply pseudo-stratification by length (not true difficulty)
        
    Returns:
        List of Question objects in open-ended format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    # Default to STEM categories
    if categories is None:
        categories = list(STEM_CATEGORIES)
    
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    questions = []
    
    for item in dataset:
        category = item["category"]
        
        # Filter to requested categories
        if category not in categories:
            continue
        
        # Filter out N/A options
        options = tuple(opt for opt in item["options"] if opt != "N/A")
        
        if not options:
            continue
        
        # Convert MCQ → Open-ended format
        questions.append(Question.from_mcq(
            question_id=str(item["question_id"]),
            question_text=item["question"],
            options=options,
            answer_index=item["answer_index"],
            category=category,
            cot_content=item.get("cot_content"),
            source_dataset="mmlu_pro_huggingface",
        ))
    
    # Limit per category if requested
    if questions_per_category:
        questions = _limit_per_category(questions, questions_per_category)
    
    return questions


# =============================================================================
# Stratified Sampling (Legacy compatibility)
# =============================================================================

def load_mmlu_pro_stratified(
    dataset_path: Path | str | None = None,
    categories: list[str] | None = None,
    question_ids: list[str] | None = None,
    difficulty_levels: list[int] | None = None,
) -> list[Question]:
    """
    Load MMLU-Pro Stratified dataset.
    
    DEPRECATED: Use load_mmlu_pro_stem() instead for STEM-only with
    proper difficulty stratification from EducationQ.
    """
    # Use STEM loader
    return load_mmlu_pro_stem(
        from_educationq=True,
        questions_per_category=None,
    )


# =============================================================================
# Stats and Info
# =============================================================================

def get_stem_category_stats(questions: list[Question]) -> dict:
    """Get statistics about STEM category distribution."""
    from collections import Counter
    
    cats = Counter(q.category for q in questions)
    
    return {
        "total": len(questions),
        "categories": dict(cats),
        "stem_only": all(q.category in STEM_CATEGORIES for q in questions),
    }


def print_stem_stats(questions: list[Question]):
    """Print STEM category statistics."""
    stats = get_stem_category_stats(questions)
    
    print(f"Total questions: {stats['total']}")
    print(f"STEM-only: {stats['stem_only']}")
    print("By category:")
    for cat, count in sorted(stats['categories'].items()):
        is_stem = "✓" if cat in STEM_CATEGORIES else "✗"
        print(f"  {is_stem} {cat}: {count}")
