"""
Example usage of the evaluation framework.

This shows how to use the framework to evaluate your model on TutorBench.
"""

from typing import List
from .experiment import ExperimentConfig
from .run_evaluation import run_evaluation


def example_model_function(system_prompt: str, messages: List[dict]) -> str:
    """
    Example model function - replace this with your actual model API call.

    Args:
        system_prompt: System prompt for the tutoring use case
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Model's response as a string
    """
    # This is where you'd call your model's API
    # For example, with OpenAI:
    #
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "system", "content": system_prompt}] + messages,
    #     temperature=0.0,
    #     seed=42
    # )
    # return response.choices[0].message.content

    # For now, return a placeholder
    raise NotImplementedError("Replace this with your actual model API call")


def main():
    """
    Example: Run evaluation on a model.
    """

    # 1. Configure your experiment
    config = ExperimentConfig(
        model_name="my-model",
        model_version="v1.0",
        temperature=0.0,
        top_p=1.0,
        seed=42,
        use_async=True,
        max_samples=None,  # Use all samples, or set to e.g. 100 for testing
        notes="First evaluation run on TutorBench"
    )

    # 2. Run evaluation
    results = run_evaluation(
        model_fn=example_model_function,
        config=config,
        use_hf=True,  # Load from HuggingFace
        # Or use: samples_path="path/to/samples.json"
        n_runs=3,  # Run 3 times for confidence intervals
        output_dir="results",
        save_to_leaderboard=True,
        verbose=True
    )

    # 3. Results are automatically saved to:
    #    - results/my-model/my-model_TIMESTAMP/run_1.json
    #    - results/my-model/my-model_TIMESTAMP/run_2.json
    #    - results/my-model/my-model_TIMESTAMP/run_3.json
    #    - results/my-model/my-model_TIMESTAMP/final_report.json
    #    - results/leaderboard.json

    # 4. Access specific metrics programmatically
    print(f"\nFinal Score: {results['overall_score_mean']:.2%} ± {results['overall_score_ci_margin']:.2%}")
    print(f"Text-only: {results['text_only_mean']:.2%}")
    print(f"Multimodal: {results['multimodal_mean']:.2%}")


if __name__ == "__main__":
    main()
