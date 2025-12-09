#!/usr/bin/env python3
"""
Run a pilot experiment using STEM-only MMLU-Pro questions.

Clean architecture:
- Student attempts question
- Teacher guides with Socratic hints
- External judge (DeepSeek v3) evaluates FINAL: answers
- No runtime overrides - everything logged for analysis

Usage:
    python scripts/run_pilot.py
    python scripts/run_pilot.py --questions 5
    python scripts/run_pilot.py --categories math physics
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print a section header."""
    print()
    print(f"── {text} " + "─" * (66 - len(text)))


def main():
    parser = argparse.ArgumentParser(
        description="Run a pilot teaching experiment (STEM-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pilot.py                     # Default: 1 per STEM category
  python scripts/run_pilot.py -n 5                # 5 questions per STEM category
  python scripts/run_pilot.py -c math physics     # Specific STEM categories
  python scripts/run_pilot.py -t 10               # 10 max turns per question
        """
    )
    parser.add_argument("-n", "--questions", type=int, default=1,
                        help="Questions per category (default: 1)")
    parser.add_argument("-c", "--categories", nargs="+", default=None,
                        help="STEM categories to include (default: all STEM)")
    parser.add_argument("-t", "--max-turns", type=int, default=10,
                        help="Max teacher turns per question (default: 10)")
    parser.add_argument("--name", type=str, default="pilot_stem",
                        help="Experiment name (default: pilot_stem)")
    args = parser.parse_args()
    
    # Imports (after path setup)
    from src.environment import (
        ExperimentConfig,
        ExperimentRunner,
        EnvironmentConfig,
        SamplingConfig,
        TEACHER_CONFIGS,
        DEFAULT_STUDENT_CONFIG,
    )
    from src.environment.datasets import (
        load_mmlu_pro_from_huggingface,
        STEM_CATEGORIES,
    )
    from src.environment.llm_client import create_llm_client
    from src.environment.judge import JUDGE_MODEL_ID
    
    print_header("STEM TEACHING EXPERIMENT (Clean Architecture)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Configuration")
    
    # Use specified categories or all STEM
    categories = tuple(args.categories) if args.categories else STEM_CATEGORIES
    
    # Validate categories are STEM
    for cat in categories:
        if cat not in STEM_CATEGORIES:
            print(f"  ⚠ Warning: '{cat}' is not a STEM category")
            print(f"  Valid STEM categories: {', '.join(STEM_CATEGORIES)}")
    
    sampling = SamplingConfig(
        questions_per_category=args.questions,
        categories=categories,
    )
    
    config = ExperimentConfig(
        name=args.name,
        description=f"STEM pilot: {args.questions} questions/category from {list(categories)}",
        teacher_config=TEACHER_CONFIGS[0],
        student_config=DEFAULT_STUDENT_CONFIG,
        env_config=EnvironmentConfig(
            max_teacher_turns=args.max_turns,
            sampling=sampling,
        ),
        output_dir=Path("experiments"),
    )
    
    print(f"  Experiment:  {config.name}")
    print(f"  Teacher:     {config.teacher_config.model_id}")
    print(f"  Student:     {config.student_config.model_id}")
    print(f"  Judge:       {JUDGE_MODEL_ID} (external)")
    print(f"  Categories:  {', '.join(categories)} (STEM only)")
    print(f"  Questions:   {args.questions} per category")
    print(f"  Max turns:   {args.max_turns}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load Dataset (STEM Only)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Loading STEM Dataset")
    
    print("  Fetching from HuggingFace...", end=" ", flush=True)
    all_questions = load_mmlu_pro_from_huggingface(
        categories=list(categories),
    )
    print(f"✓ {len(all_questions)} questions")
    
    # Show stats
    from collections import Counter
    cat_counts = Counter(q.category for q in all_questions)
    print()
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")
    
    # Sample questions
    from src.environment.runner import sample_questions
    sampled = sample_questions(all_questions, sampling)
    print(f"\n  Sampled: {len(sampled)} questions for experiment")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Verify API Connection
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Verifying API Connection")
    
    try:
        client = create_llm_client()
        print("  ✓ Client created")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
        print()
        print("  Make sure TOGETHER_API_KEY is set in your .env file")
        return 1
    
    # Test API with timing
    print("  Testing API call...", end=" ", flush=True)
    t0 = time.time()
    try:
        response = client.chat_with_metadata(
            model=config.teacher_config.model_id,
            messages=[{"role": "user", "content": "Reply with exactly: API_OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        t1 = time.time()
        
        if "API_OK" in response.content or "API" in response.content:
            print(f"✓ Response in {t1-t0:.2f}s")
            print(f"  ✓ Model: {response.model}")
            print(f"  ✓ Tokens: {response.prompt_tokens} in, {response.completion_tokens} out")
        else:
            print(f"⚠ Unexpected response: {response.content[:50]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run Experiment
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Running Experiment")
    print()
    print("  Flow: Student → Teacher → [Judge on FINAL:] → repeat")
    print()
    
    runner = ExperimentRunner(config, client)
    output = runner.run(sampled, show_progress=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Saving Results")
    
    experiment_id = config.experiment_id
    saved = output.save(config.output_dir, experiment_id)
    
    print(f"  Folder:  {saved['folder']}")
    print(f"  Config:  config.yaml")
    print(f"  Summary: summary.json")
    print(f"  Results: results.json")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Experiment Summary")
    
    stats = output.stats
    n = stats.get('num_sessions', 0)
    
    print(f"  Sessions:          {n}")
    print(f"  Initial correct:   {stats.get('initial_correct', 0)} ({stats.get('initial_correct_rate', 0)*100:.1f}%)")
    print(f"  Final correct:     {stats.get('final_correct', 0)} ({stats.get('final_correct_rate', 0)*100:.1f}%)")
    print(f"  Improvement:       {stats.get('improvement', 0):+d}")
    print()
    print(f"  Natural end:       {stats.get('terminated_naturally', 0)}")
    print(f"  Forced end:        {stats.get('forced_termination', 0)}")
    print(f"  Avg turns:         {stats.get('avg_turns', 0):.1f}")
    print(f"  Total time:        {output.total_duration_seconds:.1f}s")
    
    print()
    print(f"  Final actions:")
    for action, count in sorted(stats.get('final_action_distribution', {}).items(), key=lambda x: -x[1]):
        print(f"    {action}: {count}")
    
    print_header("EXPERIMENT COMPLETE")
    print(f"  Results saved to: {saved['folder']}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
