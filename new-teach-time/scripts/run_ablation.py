#!/usr/bin/env python3
"""
Run ablation study comparing different teaching approaches.

Tests combinations of:
- Model capability (DeepSeek V3 vs Qwen 7B vs Llama 70B)
- ReAct framework (with vs without)
- Fine-tuning (base vs fine-tuned) [if model ID provided]

Usage:
    # Test ReAct effect on DeepSeek V3 and Qwen 7B (5 math questions each)
    python scripts/run_ablation.py
    
    # More questions
    python scripts/run_ablation.py -n 10
    
    # Custom variants
    python scripts/run_ablation.py --variants deepseek_v3_plain qwen_7b_react
    
    # Include fine-tuned model
    python scripts/run_ablation.py --finetuned-model YOUR_MODEL_ID
    
    # List available configs
    python scripts/run_ablation.py --list
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    print()
    print(f"── {text} " + "─" * (66 - len(text)))


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on teaching approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablation.py                    # Default: DeepSeek + Qwen, plain vs react
  python scripts/run_ablation.py -n 10              # 10 math questions per variant
  python scripts/run_ablation.py -t 12              # 12 max turns per question
  python scripts/run_ablation.py --list             # Show available configs
  python scripts/run_ablation.py --variants qwen_7b_plain qwen_7b_react
  python scripts/run_ablation.py --finetuned-model ft:together:your-model-id
        """
    )
    
    # Question settings
    parser.add_argument("-n", "--questions", type=int, default=5,
                        help="Number of math questions (default: 5)")
    parser.add_argument("-t", "--max-turns", type=int, default=10,
                        help="Max teacher turns per question (default: 10)")
    
    # Variant selection
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variant names to test (default: all baselines)")
    parser.add_argument("--finetuned-model", type=str, default=None,
                        help="Together AI model ID for fine-tuned model")
    
    # Output
    parser.add_argument("--name", type=str, default="ablation",
                        help="Experiment name (default: ablation)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ablation"),
                        help="Output directory (default: experiments/ablation)")
    
    # Utilities
    parser.add_argument("--list", action="store_true",
                        help="List available teacher configurations")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip pedagogical evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Imports
    from src.environment import (
        AblationRunner,
        AblationTeacherConfig,
        # 3B models
        QWEN_3B_PLAIN,
        QWEN_3B_REACT,
        LLAMA_3B_PLAIN,
        LLAMA_3B_REACT,
        # 7-8B models
        QWEN_7B_PLAIN,
        QWEN_7B_REACT,
        LLAMA_8B_PLAIN,
        LLAMA_8B_REACT,
        # Large models (for reference)
        DEEPSEEK_V3_PLAIN,
        get_available_configs,
        SamplingConfig,
    )
    from src.environment.datasets import load_mmlu_pro_from_huggingface
    from src.environment.llm_client import create_llm_client
    from src.environment.runner import sample_questions
    
    # Handle --list
    if args.list:
        print_header("AVAILABLE TEACHER CONFIGURATIONS")
        configs = get_available_configs()
        print()
        print(f"  {'Name':<25} {'Mode':<8} {'Model ID'}")
        print("  " + "-" * 70)
        for name, cfg in sorted(configs.items()):
            mode = "ReAct" if cfg.use_react else "Plain"
            # Truncate long model IDs
            model_id = cfg.model_id[:45] + "..." if len(cfg.model_id) > 45 else cfg.model_id
            print(f"  {name:<25} {mode:<8} {model_id}")
        print()
        print("  Use --variants <name1> <name2> ... to select specific configs")
        print()
        print("  Recommended ablations:")
        print("    3B models:  --variants qwen_3b_plain qwen_3b_react")
        print("    7B models:  --variants qwen_7b_plain qwen_7b_react llama_8b_plain llama_8b_react")
        print("    Size test:  --variants qwen_3b_react qwen_7b_react")
        return 0
    
    print_header("ABLATION STUDY")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build Teacher Configs
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Configuration")
    
    if args.variants:
        # Use specified variants
        available = get_available_configs()
        teacher_configs = []
        for name in args.variants:
            if name not in available:
                print(f"  ⚠ Unknown variant: {name}")
                print(f"    Available: {sorted(available.keys())}")
                return 1
            teacher_configs.append(available[name])
    else:
        # Default: Qwen 3B and 7B, plain vs react (good for fine-tuning comparison)
        teacher_configs = [
            QWEN_3B_PLAIN,
            QWEN_3B_REACT,
            QWEN_7B_PLAIN,
            QWEN_7B_REACT,
        ]
    
    # Add fine-tuned model if provided
    if args.finetuned_model:
        finetuned_plain = AblationTeacherConfig(
            name="finetuned_plain",
            model_id=args.finetuned_model,
            description="Fine-tuned model, no ReAct",
            use_react=False,
            is_finetuned=True,
            base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        )
        finetuned_react = AblationTeacherConfig(
            name="finetuned_react",
            model_id=args.finetuned_model,
            description="Fine-tuned model, with ReAct",
            use_react=True,
            is_finetuned=True,
            base_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        )
        teacher_configs.extend([finetuned_plain, finetuned_react])
    
    print(f"  Experiment:   {args.name}")
    print(f"  Variants:     {len(teacher_configs)}")
    for cfg in teacher_configs:
        react_str = "ReAct" if cfg.use_react else "Plain"
        ft_str = " [FT]" if cfg.is_finetuned else ""
        print(f"    - {cfg.name}: {react_str}{ft_str}")
    print(f"  Questions:    {args.questions} math questions")
    print(f"  Max turns:    {args.max_turns}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Loading Math Dataset")
    
    print("  Fetching from HuggingFace...", end=" ", flush=True)
    all_questions = load_mmlu_pro_from_huggingface(categories=["math"])
    print(f"✓ {len(all_questions)} math questions")
    
    # Sample questions (same for all variants!)
    sampling = SamplingConfig(questions_per_category=args.questions, categories=("math",))
    sampled = sample_questions(all_questions, sampling)
    print(f"  Sampled: {len(sampled)} questions for ablation")
    
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
    
    # Quick test
    print("  Testing API...", end=" ", flush=True)
    t0 = time.time()
    try:
        response = client.chat_with_metadata(
            model=teacher_configs[0].model_id,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        t1 = time.time()
        print(f"✓ ({t1-t0:.1f}s)")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run Ablation
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Running Ablation Study")
    
    runner = AblationRunner(llm_client=client)
    
    results = runner.run_ablation(
        teacher_configs=teacher_configs,
        questions=sampled,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        experiment_name=args.name,
        show_progress=True,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Comparison Summary
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Comparison Results")
    
    comparison = results["comparison"]
    
    print("\n  By Final Accuracy:")
    for item in comparison["rankings"]["by_final_accuracy"]:
        print(f"    {item['rank']}. {item['name']}: {item['accuracy']*100:.1f}%")
    
    print("\n  By Improvement:")
    for item in comparison["rankings"]["by_improvement"]:
        sign = "+" if item['improvement'] >= 0 else ""
        print(f"    {item['rank']}. {item['name']}: {sign}{item['improvement']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pedagogical Evaluation (Optional)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_judge:
        print_section("Pedagogical Evaluation")
        print()
        print("  Running judge evaluation on all conversations...")
        
        from judge import PedagogicalEvaluator, calculate_pes, get_pes_category
        import json
        
        evaluator = PedagogicalEvaluator(llm_client=client, verbose=args.verbose)
        
        eval_summary = {}
        
        for variant_name, variant_data in results["variants"].items():
            variant_dir = Path(variant_data["saved_to"]["folder"])
            conversations_dir = variant_dir / "conversations"
            
            if not conversations_dir.exists():
                print(f"  ⚠ No conversations found for {variant_name}")
                continue
            
            print(f"\n  Evaluating {variant_name}...")
            
            conv_files = list(conversations_dir.glob("*.json"))
            pes_scores = []
            
            for conv_file in conv_files:
                try:
                    with open(conv_file) as f:
                        data = json.load(f)
                    
                    components = evaluator.evaluate_conversation(data["conversation"])
                    pes = calculate_pes(components)
                    pes_scores.append(pes)
                    
                    if args.verbose:
                        print(f"    {conv_file.name}: PES={pes:.1f}")
                        
                except Exception as e:
                    print(f"    ⚠ Error evaluating {conv_file.name}: {e}")
            
            if pes_scores:
                avg_pes = sum(pes_scores) / len(pes_scores)
                eval_summary[variant_name] = {
                    "num_evaluated": len(pes_scores),
                    "avg_pes": avg_pes,
                    "min_pes": min(pes_scores),
                    "max_pes": max(pes_scores),
                }
                print(f"    Average PES: {avg_pes:.1f}/100")
        
        # Save eval summary
        eval_path = Path(results["experiment_dir"]) / "evaluation_summary.json"
        with open(eval_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        
        print(f"\n  Evaluation saved to: {eval_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────────────────────
    print_header("ABLATION COMPLETE")
    print(f"  Results: {results['experiment_dir']}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

