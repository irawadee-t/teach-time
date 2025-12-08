# Synthetic Tutoring Data Generation Pipeline

This pipeline generates high-quality synthetic tutoring training data in **OpenAI fine-tuning format**:

1. **Generation** (`generate_samples.py`): Creates 1,000 synthetic tutoring samples using Anthropic's structured outputs
2. **Filtering** (`filter_samples.py`): Keeps only samples passing >80% of critical rubrics
3. **Validation** (`validate_openai_format.py`): Verifies output meets OpenAI's requirements

## Setup

### 1. Install Dependencies

```bash
pip install anthropic pydantic tqdm
```

Or use the requirements file:
```bash
pip install -r requirements_synthetic.txt
```

### 2. Set API Key

Export your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or add it to a `.env` file in the project root:
```bash
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
source .env
```

### 3. Verify Input Data

Ensure `Text Tutoring Prompts.csv` exists in the project root.

## Usage

### Option 1: Test First (Recommended)

Test the pipeline with a small batch (4 samples):

```bash
python test_pipeline.py
```

This creates test files in `synthetic/`:
- `test_samples_generated.jsonl` (4 samples with metadata)
- `test_finetune_openai.jsonl` (filtered subset in OpenAI format)
- `test_filtering_statistics.json` (evaluation stats)

The test also validates the OpenAI format automatically.

### Option 2: Full Pipeline

#### Step 1: Generate 1,000 samples

```bash
python generate_samples.py
```

**Output**: `synthetic/tutoring_samples_generated.jsonl`

**Time estimate**: ~2-4 minutes (1,000 API calls, 50 concurrent)

**Cost estimate**: ~$75-125 USD (using Claude Opus 4.5)

#### Step 2: Filter samples

```bash
python filter_samples.py
```

**Output**:
- `synthetic/tutoring_finetune_openai.jsonl` (~700-900 samples, OpenAI format)
- `synthetic/filtering_statistics.json` (detailed stats)

#### Step 3: Validate output

```bash
python validate_openai_format.py synthetic/tutoring_finetune_openai.jsonl
```

**Output**: Validation report + sample entries

**Time estimate**: ~15-25 minutes (1,000 API calls)

**Cost estimate**: ~$50-75 USD (using Claude Opus 4.5)

### Total Pipeline Cost & Time

- **Time**: ~5-10 minutes (with 50 concurrent requests)
- **Cost**: ~$125-200 USD (using Claude Opus 4.5)
- **Output**: ~700-900 high-quality training samples

**Note**:
- Using Claude Opus 4.5 provides the highest quality generations
- Generation runs 50 concurrent API calls for speed
- If cost is a concern, switch to `claude-sonnet-4-20250514` for ~5x cost reduction (~$25-40 total)

## Output Format

### Generated Samples (`tutoring_samples_generated.jsonl`)

Intermediate format with metadata for filtering:

```json
{
  "id": "seed042_var2",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert tutor helping a student who is confused after your initial explanation."
    },
    {
      "role": "user",
      "content": "## Question\n...\n## My Explanation\n...\n## Student Follow-up\n..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "metadata": {
    "subject": "microeconomics",
    "topic": "price elasticity",
    "misconception_type": "confusing_elastic_vs_inelastic",
    "rubrics": [...]
  }
}
```

### Final Output (`tutoring_finetune_openai.jsonl`)

**OpenAI fine-tuning format** - one JSON object per line:

```json
{"messages": [{"role": "system", "content": "You are an expert tutor. A student asked a question, received an explanation, and now has a follow-up question showing confusion. Address their specific misconception while guiding them toward understanding."}, {"role": "user", "content": "## Question\nWhat is the relationship between price elasticity of demand and total revenue?...\n\n## Initial Explanation\nPrice elasticity of demand (PED) measures...\n\n## Student's Follow-up\nI'm confused - you said if demand is elastic, raising prices decreases revenue..."}, {"role": "assistant", "content": "That's a really common point of confusion, and your intuition about 'more money per unit' is actually correct..."}]}
```

**Critical format requirements:**
- File extension: `.jsonl`
- One JSON object per line (no pretty-printing)
- Each object has exactly one key: `"messages"`
- Messages array has exactly 3 objects: system, user, assistant
- Standardized system prompt across all samples

## Configuration

Edit these variables in the scripts:

### `generate_samples.py`
```python
NUM_SEEDS = 250           # Number of seed examples to use
SAMPLES_PER_SEED = 4      # Variations per seed (total = 250 × 4 = 1,000)
OUTPUT_PATH = "synthetic/tutoring_samples_generated.jsonl"
```

### `filter_samples.py`
```python
CRITICAL_PASS_THRESHOLD = 0.80  # Keep if >80% critical rubrics pass
INPUT_PATH = "synthetic/tutoring_samples_generated.jsonl"
OUTPUT_PATH = "synthetic/tutoring_samples_filtered.jsonl"
```

## Subject Coverage

The pipeline generates samples across 28 subjects, avoiding overlap with TutorBench:

**Economics**: microeconomics, macroeconomics, accounting, finance

**Social Sciences**: psychology, sociology, political_science, philosophy

**History**: world_history, us_history

**Mathematics**: linear_algebra, discrete_math, number_theory, probability

**Chemistry**: organic_chemistry, biochemistry

**Biology**: ecology, genetics

**Physics**: electromagnetism, thermodynamics, fluid_dynamics, quantum_mechanics

**Computer Science**: data_structures, algorithms, machine_learning, databases

**Engineering**: engineering_statics, engineering_dynamics

## Misconception Types

The generator rotates through 8 confusion patterns:

1. Why is this step necessary?
2. How do these concepts relate?
3. Counterexample that seems to contradict
4. Overgeneralizing a rule
5. Confusing similar terms
6. Incorrect assumption
7. Procedural vs conceptual understanding
8. Misapplying formula

## Quality Criteria

Samples are kept if the gold_response passes **>80% of critical rubrics**:

- **Critical rubrics**: weight = 5 or -5
- **Example critical criteria**:
  - Correctly identifies student's misconception
  - Provides accurate content
  - Empathetic and encouraging tone
  - Guides without being condescending

## Troubleshooting

### API Key Issues
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Low Keep Rate (<50%)

If filtering keeps fewer than 50% of samples:
1. Check `filtering_statistics.json` for common failure patterns
2. Consider adjusting critical rubric criteria
3. May need to regenerate with improved prompts

### Generation Failures

If many samples fail to generate:
1. Check API rate limits
2. Verify CSV file format
3. Check for JSON parsing errors in rubrics

## Next Steps

After generation and filtering:

1. **Inspect samples**: Review `tutoring_samples_filtered.jsonl`
2. **Fine-tune model**: Use filtered samples for supervised fine-tuning
3. **Evaluate**: Test on TutorBench or other tutoring benchmarks
4. **Iterate**: Adjust generation prompts based on results

## Files in This Directory

After running the full pipeline:

- `README.md` - This file
- `tutoring_samples_generated.jsonl` - Raw generated samples with metadata (1,000)
- `tutoring_finetune_openai.jsonl` - **FINAL OUTPUT** in OpenAI format (~700-900)
- `filtering_statistics.json` - Detailed evaluation stats
- `tutoring_samples_generated_checkpoint_*.jsonl` - Checkpoints (every 25 seeds)
- `test_*.jsonl` - Test run outputs

## Uploading to OpenAI

Once you have `tutoring_finetune_openai.jsonl`:

1. **Validate the file** (already done if you ran the pipeline)
   ```bash
   python validate_openai_format.py synthetic/tutoring_finetune_openai.jsonl
   ```

2. **Upload to OpenAI**
   - Go to https://platform.openai.com/finetune
   - Upload `tutoring_finetune_openai.jsonl`
   - Select base model (e.g., `gpt-4o-mini-2024-07-18`)
   - Configure hyperparameters
   - Start fine-tuning job

3. **Monitor training**
   - Track loss curves
   - Validate on held-out TutorBench samples
   - Compare to baseline model
