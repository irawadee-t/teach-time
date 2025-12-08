"""
Data loading utilities for TutorBench samples.

Supports loading from:
- JSON files
- Hugging Face datasets
- Custom formats
"""

import json
from pathlib import Path
from typing import List, Optional

from .models import (
    Sample,
    Rubric,
    UseCase,
    EvaluationDimension,
    TutoringSkill,
)


def load_samples_from_json(file_path: str) -> List[Sample]:
    """
    Load TutorBench samples from JSON file.

    Expected format:
    [
        {
            "sample_id": "001",
            "use_case": "adaptive",
            "subject": "physics",
            "system_prompt": "...",
            "messages": [...],
            "rubrics": [...],
            "is_multimodal": false
        },
        ...
    ]

    Args:
        file_path: Path to JSON file

    Returns:
        List of Sample objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Parse rubrics
        rubrics = []
        for rubric_data in item.get("rubrics", []):
            rubric = Rubric(
                criterion=rubric_data["criterion"],
                weight=rubric_data["weight"],
                evaluation_dimension=EvaluationDimension(
                    rubric_data["evaluation_dimension"]
                ),
                tutoring_skill=(
                    TutoringSkill(rubric_data["tutoring_skill"])
                    if rubric_data.get("tutoring_skill")
                    else None
                ),
                is_objective=rubric_data.get("is_objective", True),
                is_explicit=rubric_data.get("is_explicit", True),
            )
            rubrics.append(rubric)

        # Create sample
        sample = Sample(
            sample_id=item["sample_id"],
            use_case=UseCase(item["use_case"]),
            subject=item["subject"],
            system_prompt=item.get("system_prompt", ""),
            messages=item["messages"],
            rubrics=rubrics,
            is_multimodal=item.get("is_multimodal", False),
            images=item.get("images"),
        )
        samples.append(sample)

    return samples


def load_samples_from_hf(
    dataset_name: str = "ScaleAI/TutorBench",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Sample]:
    """
    Load TutorBench samples from Hugging Face dataset.

    Args:
        dataset_name: HF dataset identifier
        split: Dataset split to load
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of Sample objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for HF loading. "
            "Install with: pip install datasets"
        )

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Convert to Sample objects
    # Note: This will need to be adapted based on actual HF dataset schema
    samples = []
    for idx, item in enumerate(dataset):
        # This is a placeholder - adapt to actual schema
        sample = _convert_hf_item_to_sample(item, idx)
        if sample:
            samples.append(sample)

    return samples


def _convert_hf_item_to_sample(item: dict, idx: int) -> Optional[Sample]:
    """
    Convert HF dataset item to Sample object.

    Maps ScaleAI/TutorBench schema to our Sample model.

    Args:
        item: Dataset item from HuggingFace
        idx: Index for sample ID

    Returns:
        Sample object or None if conversion fails
    """
    try:
        import json

        # Extract basic info
        sample_id = item.get("TASK_ID", f"sample_{idx:04d}")
        batch = item.get("BATCH", "")
        subject = item.get("SUBJECT", "unknown")

        # Map BATCH to UseCase
        use_case = _map_batch_to_use_case(batch)

        # Build messages from PROMPT and FOLLOW_UP_PROMPT
        messages = []
        if "PROMPT" in item and item["PROMPT"]:
            messages.append({"role": "user", "content": item["PROMPT"]})
        if "FOLLOW_UP_PROMPT" in item and item["FOLLOW_UP_PROMPT"]:
            messages.append({"role": "user", "content": item["FOLLOW_UP_PROMPT"]})

        # Parse rubrics from JSON string
        rubrics = []
        rubrics_json = item.get("RUBRICS", "[]")
        if isinstance(rubrics_json, str):
            rubrics_data = json.loads(rubrics_json)
        else:
            rubrics_data = rubrics_json

        for r in rubrics_data:
            rubric = _parse_rubric(r)
            if rubric:
                rubrics.append(rubric)

        # Check if multimodal
        is_multimodal = "MULTIMODAL" in batch.upper()
        images = [item["IMAGE_URL"]] if item.get("IMAGE_URL") else None

        return Sample(
            sample_id=sample_id,
            use_case=use_case,
            subject=subject,
            system_prompt="",  # Will be set by runner
            messages=messages,
            rubrics=rubrics,
            is_multimodal=is_multimodal,
            images=images,
        )
    except Exception as e:
        print(f"Warning: Failed to convert item {idx}: {e}")
        return None


def _map_batch_to_use_case(batch: str) -> UseCase:
    """Map BATCH field to UseCase enum."""
    batch_upper = batch.upper()
    if "USE_CASE_1" in batch_upper:
        return UseCase.ADAPTIVE
    elif "USE_CASE_2" in batch_upper:
        return UseCase.ASSESSMENT
    elif "USE_CASE_3" in batch_upper:
        return UseCase.ACTIVE_LEARNING
    else:
        # Default to adaptive
        return UseCase.ADAPTIVE


def _parse_rubric(rubric_data: dict) -> Optional[Rubric]:
    """Parse a single rubric from HF dataset format."""
    try:
        attrs = rubric_data.get("attributes", {})

        # Get criterion
        criterion = rubric_data.get("criteria", "")

        # Map severity to weight
        severity = attrs.get("severity", "").lower()
        weight = 5 if severity == "critical" else 1

        # Parse evaluation dimension (may have multiple, take first)
        eval_dim_str = attrs.get("eval_dimension", "truthfulness")
        eval_dim_str = eval_dim_str.split(",")[0].strip()
        eval_dim_str = eval_dim_str.replace(" ", "_").lower()

        # Try to map to EvaluationDimension enum
        try:
            eval_dim = EvaluationDimension(eval_dim_str)
        except ValueError:
            # Default to truthfulness if unknown
            eval_dim = EvaluationDimension.TRUTHFULNESS

        # Parse tutoring skill (optional)
        tutoring_skill = None
        skill_str = attrs.get("tutoring_skill", "").lower()
        if skill_str and skill_str != "not applicable":
            skill_str = skill_str.replace(" ", "_")
            try:
                tutoring_skill = TutoringSkill(skill_str)
            except ValueError:
                # Skip if unknown skill
                pass

        # Parse explicitness and objectivity
        is_explicit = attrs.get("explicitness", "explicit").lower() == "explicit"
        is_objective = attrs.get("objectivity", "objective").lower() == "objective"

        return Rubric(
            criterion=criterion,
            weight=weight,
            evaluation_dimension=eval_dim,
            tutoring_skill=tutoring_skill,
            is_objective=is_objective,
            is_explicit=is_explicit,
        )
    except Exception as e:
        print(f"Warning: Failed to parse rubric: {e}")
        return None


def save_results_to_json(results: List, output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of EvaluationResult objects
        output_path: Path to output file
    """
    # Convert to serializable format
    results_data = []
    for result in results:
        result_dict = {
            "sample_id": result.sample_id,
            "model_name": result.model_name,
            "model_response": result.model_response,
            "weighted_score": result.weighted_score,
            "pass_rate": result.pass_rate,
            "rubric_ratings": [
                {
                    "criterion": r.rubric.criterion,
                    "weight": r.rubric.weight,
                    "passed": r.passed,
                    "explanation": r.explanation,
                }
                for r in result.rubric_ratings
            ],
        }
        results_data.append(result_dict)

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
