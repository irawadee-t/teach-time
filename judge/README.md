# LLM-as-a-Judge Pedagogical Evaluation System

A comprehensive evaluation framework for assessing tutoring conversations using Claude Sonnet 4.5.

## Overview

This system evaluates tutoring quality across three layers:

### Layer 1: 8-Dimension Tutor Response Quality (80% weight)
Based on Maurya et al. (2025):
1. **Comprehension Probing** - Does the tutor check student understanding?
2. **Background Knowledge** - Does the tutor assess prior knowledge early?
3. **Guidance Level** - Is scaffolding appropriate without giving away answers?
4. **Error Feedback** - Is error feedback constructive and explanatory?
5. **Encouragement** - Is the tone supportive and motivating?
6. **Coherence** - Are responses clear and well-organized?
7. **Relevance** - Does the tutor address student's actual needs?
8. **Student Talk Ratio** - Does the student contribute 50-80% of dialogue?

### Layer 2: Question Depth Analysis (10% weight)
Classifies tutor questions by cognitive depth:
- **Recall**: Factual questions
- **Procedural**: How-to questions
- **Conceptual**: Why/understanding questions
- **Metacognitive**: Thinking about thinking

### Layer 3: ICAP Engagement Classification (10% weight)
Based on Chi & Wylie (2014):
- **Passive**: Student only listens
- **Active**: Student repeats/rehearses
- **Constructive**: Student generates new ideas
- **Interactive**: Collaborative dialogue

## Pedagogical Effectiveness Score (PES)

**Formula:** PES = (Layer1 × 0.8 + Layer2 × 0.1 + Layer3 × 0.1) × 100

**Categories:**
- **Excellent** (85-100): Outstanding pedagogical quality
- **Good** (70-84): Solid tutoring with minor areas for improvement
- **Adequate** (55-69): Acceptable but significant room for growth
- **Poor** (40-54): Major pedagogical issues present
- **Very Poor** (0-39): Serious deficiencies in tutoring approach

## Installation

### Prerequisites
- Python 3.8+
- Anthropic API key with Claude Sonnet 4.5 access

### Setup

1. Install dependencies (if not already installed):
```bash
pip install anthropic>=0.39.0
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Usage

### Command Line Interface

#### Evaluate a Single Conversation

```bash
python -m judge.cli evaluate judge/test_conversations/1_excellent_tutoring.json
```

With custom output directory:
```bash
python -m judge.cli evaluate path/to/conversation.json --output-dir results/eval1
```

With verbose output:
```bash
python -m judge.cli evaluate path/to/conversation.json --verbose
```

#### Batch Evaluate Multiple Conversations

```bash
python -m judge.cli batch judge/test_conversations/
```

This will:
1. Evaluate all JSON files in the directory
2. Generate individual reports for each conversation
3. Create a summary report with statistics

### Python API

```python
from judge import PedagogicalEvaluator
from judge.metrics import calculate_pes, get_pes_category
from judge.report import generate_report
import json

# Initialize evaluator
evaluator = PedagogicalEvaluator(
    anthropic_api_key="your-api-key",
    judge_model="claude-sonnet-4-20250514",
    verbose=True
)

# Load conversation
with open("conversation.json", "r") as f:
    data = json.load(f)
    conversation = data["conversation"]

# Evaluate
components = evaluator.evaluate_conversation(conversation)

# Calculate PES
pes = calculate_pes(components)
category = get_pes_category(pes)

print(f"PES: {pes}/100 ({category})")
print(f"Summary: {components.summary}")

# Generate report
report = generate_report(components, conversation, output_format="text")
print(report)
```

## Conversation JSON Format

Conversations should be formatted as JSON files:

```json
{
  "metadata": {
    "name": "conversation_name",
    "description": "Description of the conversation",
    "domain": "mathematics",
    "topic": "quadratic equations"
  },
  "conversation": [
    {
      "role": "tutor",
      "content": "Tutor's message"
    },
    {
      "role": "student",
      "content": "Student's response"
    }
  ]
}
```

**Required fields:**
- `conversation`: List of turn objects with `role` and `content`
- `role`: Either "tutor" or "student"
- `content`: The message text

**Optional metadata:**
- `name`: Identifier for the conversation
- `description`: What makes this conversation interesting
- `domain`: Subject area (e.g., "mathematics", "physics")
- `topic`: Specific topic being taught

## Test Conversations

Five test conversations are included in `test_conversations/`:

1. **excellent_tutoring.json** (Expected PES: 85-95)
   - Strong Socratic questioning
   - Appropriate scaffolding
   - High student engagement

2. **answer_revealer.json** (Expected PES: 55-65)
   - Excessive guidance
   - Gives away answers
   - Low student cognitive work

3. **passive_engagement.json** (Expected PES: 60-70)
   - Tutor dominates conversation
   - Long explanations
   - Minimal student contribution

4. **incoherent_responses.json** (Expected PES: 40-50)
   - Unclear explanations
   - Contradictory statements
   - Confusing organization

5. **harsh_tone.json** (Expected PES: 50-60)
   - Discouraging language
   - Impatient tone
   - Demotivating feedback

### Running Test Evaluations

```bash
# Evaluate all test conversations
python -m judge.cli batch judge/test_conversations/ --verbose

# Results will be saved to judge/results/
```

## Output Files

After evaluation, you'll find:

```
judge/results/
├── <name>_results.json       # Full evaluation data (JSON)
├── <name>_report.txt         # Human-readable report
└── batch_summary.txt         # Summary of batch evaluation
```

### JSON Results Structure

```json
{
  "name": "conversation_name",
  "pes_score": 87.5,
  "pes_category": "Excellent",
  "conversation": [...],
  "evaluation": {
    "overall_quality": "excellent",
    "summary": "...",
    "strengths": [...],
    "areas_for_improvement": [...],
    "recommendations": [...],
    "layers": {
      "layer1_dimensions": {...},
      "layer2_question_depth": {...},
      "layer3_icap": {...}
    }
  },
  "generated_at": "2025-01-15T10:30:00"
}
```

## Customization

### Using a Different Judge Model

```python
evaluator = PedagogicalEvaluator(
    judge_model="claude-opus-4-20250514",  # Use Opus instead
    verbose=True
)
```

### Adjusting Layer Weights

Edit `judge/metrics.py` and modify the `calculate_pes()` function:

```python
def calculate_pes(components: PESComponents) -> float:
    layer1 = components.layer1_score()
    layer2 = components.layer2_score()
    layer3 = components.layer3_score()

    # Customize weights (must sum to 1.0)
    weighted_score = (layer1 * 0.7) + (layer2 * 0.15) + (layer3 * 0.15)

    return weighted_score * 100
```

## Interpreting Results

### Dimension Scores (1-5 scale)
- **5**: Excellent - Best practices demonstrated
- **4**: Good - Minor improvements possible
- **3**: Adequate - Significant room for improvement
- **2**: Poor - Major issues present
- **1**: Very Poor - Critical deficiencies

### Using DAMR (Desired Annotation Match Rate)

If you have target scores for comparison:

```python
from judge.metrics import calculate_damr

dimension_scores = {
    "comprehension_probing": 4,
    "background_knowledge": 5,
    "guidance_level": 3,
    # ... other dimensions
}

desired_scores = {
    "comprehension_probing": 5,
    "background_knowledge": 5,
    "guidance_level": 4,
    # ... target values
}

damr = calculate_damr(dimension_scores, desired_scores)
print(f"DAMR: {damr}%")  # Percentage of exact matches
```

## API Costs

Using Claude Sonnet 4.5:
- **Input**: ~$3 per million tokens
- **Output**: ~$15 per million tokens

**Estimated cost per evaluation:**
- Single conversation (10 turns): ~$0.15-0.30
- Batch of 50 conversations: ~$7.50-15.00

Costs vary based on conversation length and response detail.

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/teach-time
python -m judge.cli evaluate judge/test_conversations/1_excellent_tutoring.json
```

### API Key Issues

```bash
# Check if API key is set
echo $ANTHROPIC_API_KEY

# Or pass it directly
python -m judge.cli evaluate file.json --api-key your-key-here
```

### JSON Parsing Errors

If the judge returns unparseable JSON, enable verbose mode to see raw responses:
```bash
python -m judge.cli evaluate file.json --verbose
```

## Architecture

```
judge/
├── __init__.py           # Package initialization
├── evaluator.py          # Main evaluation engine
├── prompts.py            # Evaluation prompts for each dimension
├── metrics.py            # PES calculation and scoring
├── report.py             # Report generation
├── cli.py                # Command-line interface
├── test_conversations/   # 5 test conversations
└── results/              # Generated evaluation results (created on first run)
```

## References

- **Maurya et al. (2025)**: 8-dimension tutor response quality taxonomy
- **Chi & Wylie (2014)**: ICAP framework for student engagement
- **Claude Sonnet 4.5**: Anthropic's latest reasoning model for evaluation

## License

Part of the TeachTime tutoring evaluation system.
