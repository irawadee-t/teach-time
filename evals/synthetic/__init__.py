"""
Synthetic data generation pipeline for TutorBench finetuning.

Pipeline stages:
1. generate_samples_v2: Generate responses using Qwen via Together AI
2. grade_responses: Grade responses using Claude Sonnet judge
3. filter_and_curate: Apply quality thresholds and diversity selection
4. deduplicate: Remove near-duplicate responses
5. export_training_data: Format for finetuning with full visibility reports

Usage:
    # Run full pipeline
    python -m evals.synthetic.run_pipeline

    # Or run individual stages
    python -m evals.synthetic.generate_samples_v2 --help
    python -m evals.synthetic.grade_responses --help
    python -m evals.synthetic.filter_and_curate --help
    python -m evals.synthetic.deduplicate --help
    python -m evals.synthetic.export_training_data --help
"""
