"""
Example usage of the LLM-as-a-judge pedagogical evaluation system.

This script demonstrates how to:
1. Evaluate a single conversation
2. Generate and save reports
3. Use the evaluation results
"""

import os
from pathlib import Path
from judge import PedagogicalEvaluator
from judge.metrics import calculate_pes, get_pes_category
from judge.report import generate_report, save_evaluation_results


def main():
    """Example evaluation workflow."""

    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: Please set TOGETHER_API_KEY environment variable")
        print("Example: export TOGETHER_API_KEY='your-key-here'")
        return

    # Initialize evaluator with DeepSeek V3
    print("Initializing evaluator with DeepSeek V3...")
    evaluator = PedagogicalEvaluator(
        api_key=api_key,
        judge_model="deepseek-ai/DeepSeek-V3",
        verbose=True  # Show detailed progress
    )

    # Evaluate the excellent tutoring example
    print("\n" + "="*60)
    print("Evaluating: Excellent Tutoring Example")
    print("="*60)

    test_file = Path("judge/test_conversations/1_excellent_tutoring.json")
    components = evaluator.evaluate_from_file(test_file)

    # Calculate PES
    pes = calculate_pes(components)
    category = get_pes_category(pes)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"PES Score: {pes}/100")
    print(f"Category: {category}")
    print(f"\nOverall Quality: {components.overall_quality}")
    print(f"\nSummary: {components.summary}")

    print("\nStrengths:")
    for i, strength in enumerate(components.strengths, 1):
        print(f"  {i}. {strength}")

    print("\nAreas for Improvement:")
    for i, area in enumerate(components.areas_for_improvement, 1):
        print(f"  {i}. {area}")

    print("\nRecommendations:")
    for i, rec in enumerate(components.recommendations, 1):
        print(f"  {i}. {rec}")

    # Generate and save full report
    print("\n" + "="*60)
    print("Saving detailed report...")

    import json
    with open(test_file, "r") as f:
        data = json.load(f)
        conversation = data["conversation"]

    output_dir = Path("judge/results/examples")
    save_evaluation_results(
        components=components,
        conversation=conversation,
        output_dir=output_dir,
        name="excellent_tutoring_example"
    )

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("  - excellent_tutoring_example_results.json")
    print("  - excellent_tutoring_example_report.txt")


if __name__ == "__main__":
    main()
