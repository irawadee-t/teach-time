#!/usr/bin/env python3
"""
Re-run pedagogical evaluation on existing conversations.

Usage:
    python scripts/run_judge_only.py experiments/ablation/ablation_20251208_184517
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Re-run pedagogical evaluation on saved conversations")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: {experiment_dir} does not exist")
        return 1
    
    # Import after path setup
    from src.environment.llm_client import create_llm_client
    from judge.evaluator import PedagogicalEvaluator
    from judge.metrics import calculate_pes
    
    print("\n" + "=" * 70)
    print("  PEDAGOGICAL EVALUATION (Re-run)")
    print("=" * 70)
    print(f"\n  Experiment: {experiment_dir.name}")
    
    # Create LLM client for judge
    client = create_llm_client()
    evaluator = PedagogicalEvaluator(llm_client=client, verbose=args.verbose)
    
    # Find all variant directories
    variants = [d for d in experiment_dir.iterdir() if d.is_dir() and (d / "conversations").exists()]
    
    print(f"  Variants found: {len(variants)}")
    for v in variants:
        print(f"    - {v.name}")
    print()
    
    all_results = {}
    
    for variant_dir in sorted(variants):
        variant_name = variant_dir.name
        conv_dir = variant_dir / "conversations"
        conv_files = sorted(conv_dir.glob("*.json"))
        
        print(f"\n  Evaluating {variant_name}...")
        print(f"    Conversations: {len(conv_files)}")
        
        variant_scores = []
        eval_dir = variant_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        for conv_file in conv_files:
            print(f"    - {conv_file.name}...", end=" ", flush=True)
            try:
                components = evaluator.evaluate_from_file(conv_file)
                pes = calculate_pes(components)
                variant_scores.append(pes)
                print(f"PES: {pes:.1f}")
                
                # Save individual evaluation
                eval_result = {
                    "conversation_file": conv_file.name,
                    "pes_score": pes,
                    "dimensions": {
                        "comprehension_probing": {"score": components.comprehension_probing.score, "justification": components.comprehension_probing.justification},
                        "background_knowledge": {"score": components.background_knowledge.score, "justification": components.background_knowledge.justification},
                        "guidance_level": {"score": components.guidance_level.score, "justification": components.guidance_level.justification},
                        "error_feedback": {"score": components.error_feedback.score, "justification": components.error_feedback.justification},
                        "encouragement": {"score": components.encouragement.score, "justification": components.encouragement.justification},
                        "coherence": {"score": components.coherence.score, "justification": components.coherence.justification},
                        "relevance": {"score": components.relevance.score, "justification": components.relevance.justification},
                        "student_talk_ratio": {"score": components.student_talk_ratio.score, "justification": components.student_talk_ratio.justification},
                    },
                    "question_depth": {
                        "score": components.question_depth.score,
                        "question_count": components.question_depth.question_count,
                        "justification": components.question_depth.justification,
                    },
                    "icap": {
                        "score": components.icap_engagement.score,
                        "distribution": components.icap_engagement.engagement_distribution,
                        "justification": components.icap_engagement.justification,
                    },
                    "summary": components.summary,
                }
                
                eval_path = eval_dir / f"{conv_file.stem}_eval.json"
                with open(eval_path, "w") as f:
                    json.dump(eval_result, f, indent=2)
                    
            except Exception as e:
                print(f"ERROR: {e}")
                variant_scores.append(0)
        
        if variant_scores:
            avg_pes = sum(variant_scores) / len(variant_scores)
            all_results[variant_name] = {
                "num_evaluated": len(variant_scores),
                "avg_pes": round(avg_pes, 1),
                "min_pes": round(min(variant_scores), 1),
                "max_pes": round(max(variant_scores), 1),
                "scores": variant_scores,
            }
            print(f"    → Average PES: {avg_pes:.1f}/100")
    
    # Save summary
    summary_path = experiment_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Results saved to: {summary_path}")
    
    # Print comparison
    print("\n  By Average PES Score:")
    sorted_variants = sorted(all_results.items(), key=lambda x: x[1]["avg_pes"], reverse=True)
    for i, (name, data) in enumerate(sorted_variants, 1):
        print(f"    {i}. {name}: {data['avg_pes']}/100")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

