# TutorBench Evaluation System

**100% faithful implementation** of the TutorBench benchmark from Scale AI.
Paper: [TutorBench: A Benchmark to Assess Tutoring Capabilities of LLMs](https://arxiv.org/abs/2510.02663)

## Overview

TutorBench evaluates LLMs on **three core tutoring use cases**:

1. **Adaptive Explanation** - Tailoring explanations to student confusion
2. **Assessment & Feedback** - Providing actionable feedback on student work
3. **Active Learning Support** - Guiding with hints without revealing answers

**Key Features:**
- Sample-specific rubrics (3-39 per sample)
- Weighted scoring: critical (±5), non-critical (1)
- LLM-as-judge: Claude Sonnet 4 (F1=0.82 vs human majority)
- Fine-grained analysis by dimension, skill, subject, use case
- Multimodal support (text + images)

## Quick Setup

```bash
# 1. Setup environment (automated)
./setup_env.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="your-key"
# Or create .env file:
echo "ANTHROPIC_API_KEY=your-key" > .env

# 3. Verify installation
python3 -c "from evals import TutorBenchJudge; print('✓ Ready')"
```

## Quick Start

```python
from evals import (
    Sample, Rubric, UseCase, EvaluationDimension,
    TutorBenchJudge, evaluate_model, aggregate_scores
)

# Define your model
def my_model(system_prompt: str, messages: list) -> str:
    return your_llm_api_call(system_prompt, messages)

# Load samples
from evals import load_samples_from_json
samples = load_samples_from_json("samples.json")

# Run evaluation
judge = TutorBenchJudge()
results = evaluate_model(samples, my_model, judge, "my-model", verbose=True)

# Analyze
metrics = aggregate_scores(results)
print(f"Score: {metrics['overall_score']:.1%}")
```

## System Prompts (Exact from Paper)

**Adaptive Explanation:**
```
You are an AI tutor helping a high school student understand a concept.
Answer their question clearly and adjust your explanation based on what
the student says they're confused about.
```

**Assessment & Feedback:**
```
You are an AI tutor reviewing a student's answer to a question.
Evaluate whether it is correct, identify any mistakes, and explain
your reasoning clearly. Provide an assessment of the student incorrect
solution in the first response
```

**Active Learning Support:**
```
You are an AI tutor helping a student who got stuck partway through
a problem. Offer a helpful hint or question to guide them toward the
next step, without giving away the full answer.
```

*(Multimodal variants add: "The image has the student partial solution...")*

## Scoring Formula

**Weighted Average Rubric Rating (ARR_w):**

```
ARR_w = Σ(w_i · r_i) / Σ(w_i · 1_{w_i > 0})
```

Where:
- `w_i ∈ {-5, 1, 5}` - Rubric weight (critical/non-critical/negative)
- `r_i ∈ {0, 1}` - Pass/fail rating
- Only positive weights in denominator

**Example:**
- Rubric 1 (weight +5): PASS → +5 points
- Rubric 2 (weight +1): PASS → +1 points
- Rubric 3 (weight -5): FAIL → 0 points (no penalty)
- Score = (5 + 1) / (5 + 1) = 100%

## Sample Data Format

```json
{
  "sample_id": "001",
  "use_case": "active_learning",
  "subject": "statistics",
  "messages": [
    {"role": "user", "content": "I'm stuck on standard deviation..."}
  ],
  "rubrics": [
    {
      "criterion": "Must NOT reveal the final answer",
      "weight": -5,
      "evaluation_dimension": "instruction_following",
      "tutoring_skill": "step_by_step_help",
      "is_objective": true,
      "is_explicit": true
    }
  ],
  "is_multimodal": false
}
```

## Evaluation Dimensions (8)

- `instruction_following` - Adherence to prompts
- `truthfulness` - Factual accuracy
- `conciseness_relevance` - Direct, on-topic, efficient
- `style_tone` - Clarity, fluency, appropriateness
- `visual_perception` - Image content identification (multimodal)
- `visual_reasoning` - Reasoning from images (multimodal)
- `student_level_calibration` - Accounting for student knowledge
- `emotional_component` - Responding to student emotions

## Tutoring Skills (8)

- `asking_guiding_questions`
- `identifying_core_difficulty`
- `identifying_correct_steps`
- `identifying_incorrect_steps`
- `including_examples`
- `providing_alternative_solutions`
- `stating_knowledge` (definitions, theorems, laws)
- `step_by_step_help`

## Analysis Functions

```python
from evals import aggregate_scores, aggregate_by_dimension, aggregate_by_skill

# Overall metrics
overall = aggregate_scores(results)
# → {overall_score, std_score, median_score, mean_pass_rate, n_samples}

# By dimension
by_dim = aggregate_by_dimension(results)
# → {dimension: {pass_rate, n_rubrics}}

# By skill
by_skill = aggregate_by_skill(results)
# → {skill: {pass_rate, n_rubrics}}
```

## Loading Data

**From JSON:**
```python
from evals import load_samples_from_json
samples = load_samples_from_json("samples.json")
```

**From Hugging Face:**
```python
from evals import load_samples_from_hf
samples = load_samples_from_hf(
    dataset_name="ScaleAI/TutorBench",
    split="train",
    max_samples=100
)
```

**Saving Results:**
```python
from evals import save_results_to_json
save_results_to_json(results, "results.json")
```

## Architecture

```
evals/
├── models.py          # Data models (Sample, Rubric, Results)
├── scoring.py         # ARR_w formula + aggregation
├── judge.py           # LLM judge (Claude Sonnet 4)
├── prompts.py         # System prompts from paper
├── runner.py          # Evaluation orchestration
└── data_loader.py     # JSON/HF loading + real dataset support
```

**Total:** ~1,020 lines of production-ready code

## Performance Benchmarks

From TutorBench paper (16 frontier LLMs):

| Model | Score |
|-------|-------|
| Gemini 2.5 Pro | 55.65% |
| GPT-5 | 55.33% |
| o3 Pro | 54.62% |
| Claude Opus 4.1 (Thinking) | 50.78% |

**Key Findings:**
- No model exceeds 56% (large room for improvement)
- Adaptive explanation most challenging (47.16%)
- Claude models excel at active learning (58% vs 56%)
- <60% pass rate on core tutoring skills

## Paper Compliance

✅ Exact system prompts (Appendix A.5)
✅ ARR_w scoring formula (Section 4.1)
✅ Sample-specific rubrics (Section 2.3)
✅ Weighted scheme: -5, 1, 5 (Section 2.3)
✅ 8 evaluation dimensions (Section 2.4)
✅ 8 tutoring skills (Section 2.4)
✅ Claude Sonnet 4 judge (Section 2.3)
✅ Three use cases (Section 2.1)
✅ Multimodal support (Section 2.2)

## Citation

```bibtex
@article{tutorbench2025,
  title={TutorBench: A Benchmark to Assess Tutoring Capabilities of LLMs},
  author={Srinivasa, Rakshith S and Che, Zora and Zhang, Chen Bo Calvin and others},
  journal={arXiv preprint arXiv:2510.02663},
  year={2025}
}
```

## Troubleshooting

**Import errors:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**API key not found:**
```bash
export ANTHROPIC_API_KEY="your-key"
# Or add to .env file
```

**Permission denied:**
```bash
chmod +x setup_env.sh
```

## Development

```bash
# Install dev tools
pip install black isort mypy ruff pytest pytest-cov

# Run tests
pytest tests/ -v

# Format code
black evals/
isort evals/
```

## Getting API Keys

**Anthropic (Required):**
1. Sign up: https://console.anthropic.com
2. Create API key
3. `export ANTHROPIC_API_KEY="sk-ant-..."`

**Together AI (Optional):**
1. Sign up: https://together.ai
2. Get key from dashboard
3. `export TOGETHER_API_KEY="..."`

---

**Questions?** See `example.py` for complete working example.
**Full methodology:** Read the [TutorBench paper](https://arxiv.org/abs/2510.02663).
