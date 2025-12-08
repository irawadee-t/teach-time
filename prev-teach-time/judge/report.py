"""
Report generation for pedagogical evaluation results.

Generates human-readable reports from evaluation components.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .metrics import (
    PESComponents,
    calculate_pes,
    get_pes_category,
    dimension_breakdown,
    layer_breakdown,
)
from .prompts import DIMENSION_PROMPTS


def generate_report(
    components: PESComponents,
    conversation: Optional[List[Dict]] = None,
    output_format: str = "text",
) -> str:
    """
    Generate a comprehensive evaluation report.

    Args:
        components: Evaluation components
        conversation: Original conversation (optional, for context)
        output_format: 'text' or 'json'

    Returns:
        Formatted report string
    """
    pes = calculate_pes(components)
    category = get_pes_category(pes)

    if output_format == "json":
        return _generate_json_report(components, pes, category, conversation)
    else:
        return _generate_text_report(components, pes, category, conversation)


def _generate_text_report(
    components: PESComponents,
    pes: float,
    category: str,
    conversation: Optional[List[Dict]] = None,
) -> str:
    """Generate human-readable text report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("PEDAGOGICAL EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall Score
    lines.append("OVERALL PEDAGOGICAL EFFECTIVENESS SCORE (PES)")
    lines.append("-" * 80)
    lines.append(f"Score: {pes}/100")
    lines.append(f"Category: {category}")
    lines.append(f"Quality: {components.overall_quality}")
    lines.append("")
    lines.append(f"Summary: {components.summary}")
    lines.append("")

    # Key Findings
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)
    lines.append("\nStrengths:")
    for i, strength in enumerate(components.strengths, 1):
        lines.append(f"  {i}. {strength}")

    lines.append("\nAreas for Improvement:")
    for i, area in enumerate(components.areas_for_improvement, 1):
        lines.append(f"  {i}. {area}")

    lines.append("\nRecommendations:")
    for i, rec in enumerate(components.recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("")

    # Layer Breakdown
    layers = layer_breakdown(components)

    lines.append("DETAILED SCORE BREAKDOWN")
    lines.append("-" * 80)

    # Layer 1: 8 Dimensions
    lines.append("\nLAYER 1: 8-Dimension Tutor Response Quality (80% weight)")
    lines.append(f"Layer Score: {layers['layer1_dimensions']['score']:.3f}")
    lines.append(f"Weighted Contribution: {layers['layer1_dimensions']['weighted_contribution']:.2f}/100")
    lines.append("")

    dimensions = layers['layer1_dimensions']['dimensions']
    for dim_key, dim_data in dimensions.items():
        dim_name = DIMENSION_PROMPTS[dim_key]['name']
        score = dim_data['score']
        justification = dim_data['justification']

        lines.append(f"  {dim_name}: {score}/5")
        lines.append(f"    {justification}")
        if dim_data['evidence']:
            lines.append(f"    Evidence: {dim_data['evidence'][0][:100]}...")
        lines.append("")

    # Layer 2: Question Depth
    lines.append("LAYER 2: Question Depth Analysis (10% weight)")
    layer2 = layers['layer2_question_depth']
    lines.append(f"Layer Score: {layer2['score']:.3f}")
    lines.append(f"Weighted Contribution: {layer2['weighted_contribution']:.2f}/100")
    lines.append(f"Raw Score: {layer2['details']['raw_score']}/5")
    lines.append(f"Question Distribution: {layer2['details']['question_count']}")
    lines.append(f"Justification: {layer2['details']['justification']}")
    lines.append("")

    # Layer 3: ICAP Engagement
    lines.append("LAYER 3: ICAP Student Engagement (10% weight)")
    layer3 = layers['layer3_icap']
    lines.append(f"Layer Score: {layer3['score']:.3f}")
    lines.append(f"Weighted Contribution: {layer3['weighted_contribution']:.2f}/100")
    lines.append(f"Raw Score: {layer3['details']['raw_score']}/5")
    lines.append(f"Engagement Distribution: {layer3['details']['distribution']}")
    lines.append(f"Justification: {layer3['details']['justification']}")
    lines.append("")

    # Footer
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def _generate_json_report(
    components: PESComponents,
    pes: float,
    category: str,
    conversation: Optional[List[Dict]] = None,
) -> str:
    """Generate JSON report."""
    report = {
        "pes_score": pes,
        "pes_category": category,
        "overall_quality": components.overall_quality,
        "summary": components.summary,
        "strengths": components.strengths,
        "areas_for_improvement": components.areas_for_improvement,
        "recommendations": components.recommendations,
        "layers": layer_breakdown(components),
        "generated_at": datetime.now().isoformat(),
    }

    if conversation:
        report["conversation_length"] = len(conversation)

    return json.dumps(report, indent=2)


def generate_summary_report(results: List[Dict]) -> str:
    """
    Generate summary report for multiple evaluations.

    Args:
        results: List of evaluation result dicts

    Returns:
        Summary report string
    """
    if not results:
        return "No results to summarize."

    lines = []
    lines.append("=" * 80)
    lines.append("BATCH EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Conversations Evaluated: {len(results)}")
    lines.append("")

    # Calculate statistics
    pes_scores = [r['pes_score'] for r in results]
    avg_pes = sum(pes_scores) / len(pes_scores)
    min_pes = min(pes_scores)
    max_pes = max(pes_scores)

    lines.append("PES SCORE STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Average PES: {avg_pes:.2f}/100")
    lines.append(f"Min PES: {min_pes:.2f}/100")
    lines.append(f"Max PES: {max_pes:.2f}/100")
    lines.append("")

    # Category distribution
    categories = {}
    for r in results:
        cat = r.get('pes_category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1

    lines.append("CATEGORY DISTRIBUTION")
    lines.append("-" * 80)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = (count / len(results)) * 100
        lines.append(f"  {cat}: {count} ({pct:.1f}%)")
    lines.append("")

    # Individual results
    lines.append("INDIVIDUAL RESULTS")
    lines.append("-" * 80)
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('name', f'Conversation {i}')}")
        lines.append(f"   PES: {r['pes_score']:.2f}/100 ({r['pes_category']})")
        lines.append(f"   Summary: {r.get('summary', 'N/A')[:100]}...")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def save_report(
    report: str,
    output_path: Path,
    format: str = "text",
):
    """
    Save report to file.

    Args:
        report: Report content
        output_path: Where to save
        format: 'text' or 'json'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


def save_evaluation_results(
    components: PESComponents,
    conversation: List[Dict],
    output_dir: Path,
    name: str = "evaluation",
):
    """
    Save complete evaluation results including conversation and scores.

    Args:
        components: Evaluation components
        conversation: Original conversation
        output_dir: Output directory
        name: Base name for files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pes = calculate_pes(components)
    category = get_pes_category(pes)

    # Save full results as JSON
    results = {
        "name": name,
        "pes_score": pes,
        "pes_category": category,
        "conversation": conversation,
        "evaluation": {
            "overall_quality": components.overall_quality,
            "summary": components.summary,
            "strengths": components.strengths,
            "areas_for_improvement": components.areas_for_improvement,
            "recommendations": components.recommendations,
            "layers": layer_breakdown(components),
        },
        "generated_at": datetime.now().isoformat(),
    }

    json_path = output_dir / f"{name}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save text report
    text_report = generate_report(components, conversation, output_format="text")
    text_path = output_dir / f"{name}_report.txt"
    with open(text_path, "w") as f:
        f.write(text_report)

    print(f"Results saved:")
    print(f"  - JSON: {json_path}")
    print(f"  - Report: {text_path}")
