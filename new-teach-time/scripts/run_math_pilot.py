#!/usr/bin/env python3
"""
Run a math-only pilot experiment with pedagogical evaluation.

Tests 5 math problems, then evaluates each conversation using the judge.

Usage:
    python scripts/run_math_pilot.py
    python scripts/run_math_pilot.py --questions 5
    python scripts/run_math_pilot.py --skip-judge  # Run without evaluation
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
        description="Run math pilot with pedagogical evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_math_pilot.py                 # 5 math questions + judge
  python scripts/run_math_pilot.py -n 3            # 3 math questions + judge
  python scripts/run_math_pilot.py --skip-judge   # Run without evaluation
        """
    )
    parser.add_argument("-n", "--questions", type=int, default=5,
                        help="Number of math questions (default: 5)")
    parser.add_argument("-t", "--max-turns", type=int, default=8,
                        help="Max teacher turns per question (default: 8)")
    parser.add_argument("--name", type=str, default="math_pilot",
                        help="Experiment name (default: math_pilot)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip pedagogical evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output for judge evaluation")
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
    from src.environment.datasets import load_mmlu_pro_from_huggingface
    from src.environment.llm_client import create_llm_client
    from src.environment.runner import sample_questions
    
    print_header("MATH PILOT WITH PEDAGOGICAL EVALUATION")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Configuration")
    
    sampling = SamplingConfig(
        questions_per_category=args.questions,
        categories=("math",),  # Math only
    )
    
    config = ExperimentConfig(
        name=args.name,
        description=f"Math pilot: {args.questions} math questions for testing",
        teacher_config=TEACHER_CONFIGS[0],  # DeepSeek V3
        student_config=DEFAULT_STUDENT_CONFIG,  # Llama 70B
        env_config=EnvironmentConfig(
            max_teacher_turns=args.max_turns,
            sampling=sampling,
        ),
        output_dir=Path("experiments"),
    )
    
    print(f"  Experiment:  {config.name}")
    print(f"  Teacher:     {config.teacher_config.model_id}")
    print(f"  Student:     {config.student_config.model_id}")
    print(f"  Category:    math only")
    print(f"  Questions:   {args.questions}")
    print(f"  Max turns:   {args.max_turns}")
    print(f"  Judge:       {'Enabled' if not args.skip_judge else 'Disabled'}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Loading Math Dataset")
    
    print("  Fetching from HuggingFace...", end=" ", flush=True)
    all_questions = load_mmlu_pro_from_huggingface(
        categories=["math"],
    )
    print(f"✓ {len(all_questions)} math questions")
    
    # Sample questions
    sampled = sample_questions(all_questions, sampling)
    print(f"  Sampled: {len(sampled)} questions for experiment")
    
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
    
    # Test API
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
        else:
            print(f"⚠ Unexpected response: {response.content[:50]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run Experiment
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Running Tutoring Sessions")
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
    
    print(f"  Folder:         {saved['folder']}")
    print(f"  Config:         config.yaml")
    print(f"  Summary:        summary.json")
    print(f"  Results:        results.json")
    print(f"  Conversations:  {saved['num_conversations']} files in conversations/")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tutoring Summary
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Tutoring Summary")
    
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pedagogical Evaluation (Judge)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_judge:
        print_section("Pedagogical Evaluation (Judge)")
        print()
        print("  Evaluating each conversation with 8-dimension + ICAP analysis...")
        print()
        
        from judge import PedagogicalEvaluator, calculate_pes, get_pes_category
        from judge.report import save_evaluation_results, generate_summary_report
        
        # Initialize evaluator
        evaluator = PedagogicalEvaluator(
            llm_client=client,
            verbose=args.verbose,
        )
        
        # Evaluate each conversation
        conversations_dir = Path(saved['conversations_dir'])
        eval_results_dir = Path(saved['folder']) / "evaluations"
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        for i, session in enumerate(output.sessions, 1):
            conv_data = session.to_judge_format()
            name = conv_data['metadata'].get('name', f'session_{i}')
            
            print(f"  [{i}/{len(output.sessions)}] Evaluating {name}...", end=" ", flush=True)
            
            try:
                components = evaluator.evaluate_conversation(conv_data['conversation'])
                pes = calculate_pes(components)
                category = get_pes_category(pes)
                
                print(f"PES: {pes:.1f}/100 ({category})")
                
                # Save evaluation
                save_evaluation_results(
                    components=components,
                    conversation=conv_data['conversation'],
                    output_dir=eval_results_dir,
                    name=name,
                )
                
                all_results.append({
                    "name": name,
                    "pes_score": pes,
                    "pes_category": category,
                    "summary": components.summary,
                    "question_id": session.question.question_id,
                    "initial_correct": session.initial_judge_verdict,
                    "final_correct": session.final_judge_verdict,
                })
                
            except Exception as e:
                print(f"✗ Error: {e}")
                all_results.append({
                    "name": name,
                    "pes_score": 0,
                    "pes_category": "Error",
                    "summary": str(e),
                })
        
        # Summary
        print_section("Evaluation Summary")
        
        if all_results:
            valid_results = [r for r in all_results if r['pes_category'] != 'Error']
            
            if valid_results:
                avg_pes = sum(r['pes_score'] for r in valid_results) / len(valid_results)
                min_pes = min(r['pes_score'] for r in valid_results)
                max_pes = max(r['pes_score'] for r in valid_results)
                
                print(f"  Evaluated:    {len(valid_results)} conversations")
                print(f"  Average PES:  {avg_pes:.1f}/100")
                print(f"  Min PES:      {min_pes:.1f}/100")
                print(f"  Max PES:      {max_pes:.1f}/100")
                print()
                
                # Category distribution
                from collections import Counter
                categories = Counter(r['pes_category'] for r in valid_results)
                print("  Category Distribution:")
                for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                    pct = (count / len(valid_results)) * 100
                    print(f"    {cat}: {count} ({pct:.1f}%)")
                
                # Save summary
                import json
                summary = {
                    "experiment_id": experiment_id,
                    "num_evaluated": len(valid_results),
                    "avg_pes": avg_pes,
                    "min_pes": min_pes,
                    "max_pes": max_pes,
                    "category_distribution": dict(categories),
                    "results": all_results,
                }
                
                summary_path = eval_results_dir / "evaluation_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                
                print()
                print(f"  Evaluations saved to: {eval_results_dir}")
            else:
                print("  No valid evaluations completed.")
        else:
            print("  No evaluations to summarize.")
    
    print_header("EXPERIMENT COMPLETE")
    print(f"  Results saved to: {saved['folder']}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

