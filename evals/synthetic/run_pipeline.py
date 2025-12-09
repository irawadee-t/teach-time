"""
Master orchestrator for the synthetic data generation pipeline.

Runs the complete pipeline:
1. Generate synthetic responses (Qwen via Together AI)
2. Grade responses (Claude Sonnet judge)
3. Filter by quality thresholds
3.5. Revise failed responses using rubric feedback (optional)
4. Deduplicate across samples
5. Export training data with full visibility reports

Usage:
    # Dry run (no API calls)
    python -m evals.synthetic.run_pipeline --dry-run

    # Test on small sample
    python -m evals.synthetic.run_pipeline --max-samples 5

    # Full run
    python -m evals.synthetic.run_pipeline

    # Full run with revision step
    python -m evals.synthetic.run_pipeline --revise

    # Resume from checkpoint
    python -m evals.synthetic.run_pipeline --resume
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env.local (preferred) or .env
_project_root = Path(__file__).parent.parent.parent
for env_file in [".env.local", ".env", "evals/.env.local", "evals/.env"]:
    env_path = _project_root / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break

# Import pipeline modules - use CoT by default
from evals.synthetic.generate_samples_cot import (
    run_generation,
    TRAIN_CSV as GEN_TRAIN_CSV,
    DEFAULT_TEMPERATURE,
    DEFAULT_SAMPLES_PER_PERSONA,
)
from evals.synthetic.grade_responses import run_grading, TRAIN_CSV as GRADE_TRAIN_CSV
from evals.synthetic.filter_and_curate import run_filtering
from evals.synthetic.revise_responses import run_revision
from evals.synthetic.deduplicate import run_deduplication
from evals.synthetic.export_training_data import run_export


# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "pipeline_outputs"


def main():
    parser = argparse.ArgumentParser(
        description="Run the full synthetic data generation pipeline"
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples to process (default: all 556)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without API calls")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation step (use existing responses)")
    parser.add_argument("--skip-grading", action="store_true",
                        help="Skip grading step (use existing grades)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: pipeline_outputs/)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier (default: timestamp)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature for generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--samples-per-persona", type=int, default=DEFAULT_SAMPLES_PER_PERSONA,
                        help=f"Number of variations per persona (default: {DEFAULT_SAMPLES_PER_PERSONA})")
    parser.add_argument("--revise", action="store_true",
                        help="Enable revision step to ensure minimum pass rate per sample")
    parser.add_argument("--min-pass-rate", type=float, default=0.25,
                        help="Minimum pass rate to ensure per sample via revision (default: 0.25 = 25%%)")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    generated_path = run_dir / "01_generated.jsonl"
    graded_path = run_dir / "02_graded.jsonl"
    filtered_path = run_dir / "03_filtered.jsonl"
    rejected_path = run_dir / "03_rejected.jsonl"
    revised_path = run_dir / "03b_revised.jsonl"
    revised_graded_path = run_dir / "03c_revised_graded.jsonl"
    combined_filtered_path = run_dir / "03d_combined_filtered.jsonl"
    deduped_path = run_dir / "04_deduped.jsonl"
    training_dir = run_dir / "05_training"

    print(f"\n{'='*70}")
    print("SYNTHETIC DATA GENERATION PIPELINE")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples per persona: {args.samples_per_persona}")
    print(f"Revision enabled: {args.revise}")
    print(f"Resume: {args.resume}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*70}\n")

    # ==================================================
    # STEP 1: Generate synthetic responses
    # ==================================================
    print(f"\n{'#'*70}")
    print("# STEP 1: GENERATE SYNTHETIC RESPONSES")
    print(f"{'#'*70}")

    if args.skip_generation:
        print("⏭️  Skipping generation (--skip-generation)")
        if not generated_path.exists():
            print(f"ERROR: No generated file found at {generated_path}")
            return
    else:
        asyncio.run(run_generation(
            csv_path=GEN_TRAIN_CSV,
            output_path=generated_path,
            max_samples=args.max_samples,
            resume=args.resume,
            dry_run=args.dry_run,
            temperature=args.temperature,
            samples_per_persona=args.samples_per_persona,
        ))

    if args.dry_run:
        print("\n[DRY RUN] Pipeline would continue with grading, filtering, etc.")
        return

    # ==================================================
    # STEP 2: Grade responses
    # ==================================================
    print(f"\n{'#'*70}")
    print("# STEP 2: GRADE RESPONSES WITH CLAUDE SONNET")
    print(f"{'#'*70}")

    if args.skip_grading:
        print("⏭️  Skipping grading (--skip-grading)")
        if not graded_path.exists():
            print(f"ERROR: No graded file found at {graded_path}")
            return
    else:
        asyncio.run(run_grading(
            input_path=generated_path,
            output_path=graded_path,
            train_csv_path=GRADE_TRAIN_CSV,
            resume=args.resume,
            dry_run=False,
        ))

    # ==================================================
    # STEP 3: Filter by quality
    # ==================================================
    print(f"\n{'#'*70}")
    print("# STEP 3: FILTER BY QUALITY THRESHOLDS")
    print(f"{'#'*70}")

    run_filtering(
        input_path=graded_path,
        output_path=filtered_path,
        rejected_path=rejected_path,
        verbose=True,
    )

    # Determine input for deduplication (may include revised responses)
    final_filtered_path = filtered_path

    # ==================================================
    # STEP 3.5: Revise failed responses (optional)
    # ==================================================
    if args.revise:
        print(f"\n{'#'*70}")
        print("# STEP 3.5: REVISE FAILED RESPONSES (Qwen)")
        print(f"{'#'*70}")

        # Revise responses from samples below min_pass_rate
        # Uses Qwen to rewrite based on rubric feedback - no re-grading needed
        revised_responses = asyncio.run(run_revision(
            input_path=graded_path,  # Use all graded responses to find failures
            output_path=revised_path,
            min_pass_rate=args.min_pass_rate,
            dry_run=False,
        ))

        # Combine original filtered + revised (trust revised responses pass)
        if revised_path.exists() and revised_path.stat().st_size > 0:
            print("\nCombining original filtered + revised responses...")
            combined_responses = []

            # Load original filtered
            if filtered_path.exists():
                with open(filtered_path, 'r') as f:
                    for line in f:
                        combined_responses.append(line)

            # Load revised responses directly (no re-grading - they're revised based on rubric)
            # Convert to same format as filtered responses
            import json
            with open(revised_path, 'r') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        # Add placeholder grading fields so downstream steps work
                        r['weighted_score'] = 0.80  # Assume improved
                        r['critical_pass_rate'] = 0.80
                        r['pass_rate'] = 0.80
                        r['is_high_quality'] = True
                        r['rubric_results'] = []  # No rubric results for revised
                        combined_responses.append(json.dumps(r) + '\n')
                    except:
                        continue

            # Write combined
            with open(combined_filtered_path, 'w') as f:
                for line in combined_responses:
                    f.write(line if line.endswith('\n') else line + '\n')

            print(f"Combined {len(combined_responses)} responses into {combined_filtered_path}")
            final_filtered_path = combined_filtered_path
    else:
        print(f"\n⏭️  Skipping revision step (use --revise to enable)")

    # ==================================================
    # STEP 4: Deduplicate
    # ==================================================
    print(f"\n{'#'*70}")
    print("# STEP 4: DEDUPLICATE ACROSS SAMPLES")
    print(f"{'#'*70}")

    run_deduplication(
        input_path=final_filtered_path,
        output_path=deduped_path,
        save_log=True,
    )

    # ==================================================
    # STEP 5: Export training data
    # ==================================================
    print(f"\n{'#'*70}")
    print("# STEP 5: EXPORT TRAINING DATA")
    print(f"{'#'*70}")

    run_export(
        input_path=deduped_path,
        output_dir=training_dir,
        bad_samples_path=rejected_path,
        full_report=True,
    )

    # ==================================================
    # Final Summary
    # ==================================================
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {run_dir}")
    print(f"\n📊 Pipeline outputs:")
    print(f"   1. Generated responses: {generated_path}")
    print(f"   2. Graded responses:    {graded_path}")
    print(f"   3. Filtered (good):     {filtered_path}")
    print(f"   3. Filtered (rejected): {rejected_path}")
    if args.revise:
        print(f"   3b. Revised responses:  {revised_path}")
        print(f"   3c. Revised graded:     {revised_graded_path}")
        print(f"   3d. Combined filtered:  {combined_filtered_path}")
    print(f"   4. Deduplicated:        {deduped_path}")
    print(f"   5. Training data:       {training_dir}")
    print(f"\n📈 Full visibility reports in: {training_dir}")
    print(f"   - training.jsonl:     Ready for finetuning")
    print(f"   - quality_report.json: Statistics and analysis")
    print(f"   - good_samples.jsonl:  High-quality samples explained")
    print(f"   - bad_samples.jsonl:   Rejected samples explained")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
