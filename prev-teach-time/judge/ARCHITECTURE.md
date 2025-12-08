# Judge Directory Architecture

## Overview
The `judge/` directory implements a 3-layer LLM-as-a-judge evaluation system for assessing tutoring conversation quality using DeepSeek V3.

## Data Flow

```
conversation.json → evaluator.py → prompts.py → DeepSeek V3 API → metrics.py → report.py → output files
                         ↓                                              ↓
                    cli.py (orchestration)                      PES calculation
```

---

## File-by-File Breakdown

### 1. `__init__.py`
**Purpose**: Package initialization and public API exports

**Exports**:
- `PedagogicalEvaluator` - Main evaluation class
- `calculate_pes()` - PES score calculation
- `PESComponents` - Data structure for scores
- `generate_report()` - Report generation
- `generate_summary_report()` - Batch summary

**Inputs**: None (module import)
**Outputs**: Package namespace with public functions

---

### 2. `prompts.py`
**Purpose**: Contains all evaluation prompts for the judge LLM

**Key Components**:
- `DIMENSION_PROMPTS` - Dict of 8 dimension evaluation prompts
  - Each has: name, description, prompt template with scoring guidelines (1-5)
- `QUESTION_DEPTH_PROMPT` - Template for analyzing question cognitive depth
- `ICAP_ENGAGEMENT_PROMPT` - Template for ICAP framework classification
- `OVERALL_SUMMARY_PROMPT` - Template for synthesizing all scores

**Inputs**: None (static prompt templates)
**Outputs**: Prompt strings with `{conversation}` placeholder

**Used by**: `evaluator.py` (fills in conversation text)

---

### 3. `metrics.py`
**Purpose**: Data structures and PES calculation logic

**Key Classes**:

#### `DimensionScore` (dataclass)
- **Fields**: dimension, score (1-5), justification, evidence
- **Method**: `normalized_score()` → converts 1-5 to 0-1 range

#### `QuestionDepthScore` (dataclass)
- **Fields**: score (1-5), question_count dict, question_examples dict, justification
- **Method**: `normalized_score()` → converts 1-5 to 0-1 range

#### `ICAPScore` (dataclass)
- **Fields**: score (1-5), engagement_distribution dict, turn_classifications list, justification
- **Method**: `normalized_score()` → converts 1-5 to 0-1 range

#### `PESComponents` (dataclass)
- **Fields**: All 8 dimension scores + question_depth + icap_engagement + overall summary
- **Methods**:
  - `layer1_score()` → average of 8 dimensions (0-1)
  - `layer2_score()` → question depth (0-1)
  - `layer3_score()` → ICAP engagement (0-1)

**Key Functions**:

#### `calculate_pes(components: PESComponents) -> float`
- **Input**: PESComponents object
- **Output**: Float (0-100)
- **Formula**: `(layer1 × 0.8 + layer2 × 0.1 + layer3 × 0.1) × 100`

#### `get_pes_category(pes: float) -> str`
- **Input**: PES score (0-100)
- **Output**: Category string ("Excellent", "Good", "Adequate", "Poor", "Very Poor")

#### `calculate_damr(dimension_scores, desired_scores) -> float`
- **Input**: Two dicts mapping dimension names to scores
- **Output**: Percentage of exact matches (0-100)
- **Use case**: Comparing against target scores

#### `dimension_breakdown(components) -> Dict`
- **Input**: PESComponents
- **Output**: Dict with all dimension scores, normalized values, justifications

#### `layer_breakdown(components) -> Dict`
- **Input**: PESComponents
- **Output**: Dict with scores by layer + weighted contributions

---

### 4. `evaluator.py`
**Purpose**: Main evaluation engine - orchestrates the entire evaluation process

**Key Class**: `PedagogicalEvaluator`

#### Constructor
```python
__init__(api_key, judge_model="deepseek-ai/DeepSeek-V3", verbose=False)
```
- **Inputs**:
  - `api_key`: Together AI API key (or reads from TOGETHER_API_KEY env)
  - `judge_model`: Model name (default: DeepSeek V3)
  - `verbose`: Print progress (bool)
- **Initializes**: LLMClient with Together AI configuration

#### Core Methods

**`_format_conversation(conversation: List[Dict]) -> str`**
- **Input**: List of turn dicts `[{"role": "tutor", "content": "..."}, ...]`
- **Output**: Formatted string `"Turn 1 [TUTOR]: ...\n\nTurn 2 [STUDENT]: ..."`
- **Purpose**: Convert JSON conversation to readable text for prompts

**`_call_judge(prompt: str) -> Dict`**
- **Input**: Evaluation prompt string
- **Output**: Parsed JSON response as dict
- **Process**:
  1. Calls DeepSeek V3 via LLMClient
  2. Strips markdown code blocks if present
  3. Parses JSON
  4. Returns dict or default on error

**`evaluate_dimension(dimension_key: str, conversation: List[Dict]) -> DimensionScore`**
- **Input**:
  - `dimension_key`: Key from DIMENSION_PROMPTS (e.g., "comprehension_probing")
  - `conversation`: List of turn dicts
- **Output**: DimensionScore object with score (1-5), justification, evidence
- **Process**:
  1. Gets prompt template from `prompts.py`
  2. Formats conversation
  3. Fills in prompt template
  4. Calls judge LLM
  5. Parses response into DimensionScore

**`evaluate_question_depth(conversation: List[Dict]) -> QuestionDepthScore`**
- **Input**: List of turn dicts
- **Output**: QuestionDepthScore object
- **Process**:
  1. Uses QUESTION_DEPTH_PROMPT
  2. Calls judge to analyze all tutor questions
  3. Returns classification + score

**`evaluate_icap_engagement(conversation: List[Dict]) -> ICAPScore`**
- **Input**: List of turn dicts
- **Output**: ICAPScore object
- **Process**:
  1. Uses ICAP_ENGAGEMENT_PROMPT
  2. Calls judge to classify each student turn
  3. Returns engagement distribution + score

**`evaluate_conversation(conversation: List[Dict]) -> PESComponents`**
- **Input**: List of turn dicts
- **Output**: Complete PESComponents object
- **Process** (main orchestrator):
  1. Evaluate all 8 dimensions (Layer 1)
  2. Evaluate question depth (Layer 2)
  3. Evaluate ICAP engagement (Layer 3)
  4. Generate overall summary
  5. Assemble PESComponents
  6. Calculate and print PES if verbose

**`evaluate_from_file(filepath: Path) -> PESComponents`**
- **Input**: Path to conversation JSON file
- **Output**: PESComponents object
- **Process**:
  1. Load JSON file
  2. Extract conversation array
  3. Call `evaluate_conversation()`

---

### 5. `report.py`
**Purpose**: Generate human-readable and JSON reports from evaluation results

**Key Functions**:

#### `generate_report(components, conversation, output_format="text") -> str`
- **Input**:
  - `components`: PESComponents object
  - `conversation`: Optional list of turn dicts (for context)
  - `output_format`: "text" or "json"
- **Output**: Formatted report string
- **Delegates to**: `_generate_text_report()` or `_generate_json_report()`

#### `_generate_text_report(components, pes, category, conversation) -> str`
- **Input**: PESComponents, PES score, category, conversation
- **Output**: Multi-line text report string
- **Format**:
  ```
  ================================================================================
  PEDAGOGICAL EVALUATION REPORT
  ================================================================================
  OVERALL PES: X/100 (Category)

  KEY FINDINGS
  - Strengths: ...
  - Areas for Improvement: ...
  - Recommendations: ...

  DETAILED SCORE BREAKDOWN
  LAYER 1: 8 dimensions with scores/justifications
  LAYER 2: Question depth analysis
  LAYER 3: ICAP engagement
  ================================================================================
  ```

#### `_generate_json_report(components, pes, category, conversation) -> str`
- **Input**: Same as text report
- **Output**: JSON string with complete evaluation data

#### `generate_summary_report(results: List[Dict]) -> str`
- **Input**: List of result dicts (each with pes_score, pes_category, etc.)
- **Output**: Batch summary report string
- **Includes**:
  - Average/min/max PES
  - Category distribution
  - Individual result summaries

#### `save_evaluation_results(components, conversation, output_dir, name)`
- **Input**:
  - `components`: PESComponents
  - `conversation`: List of turn dicts
  - `output_dir`: Path to save location
  - `name`: Base filename
- **Output**: Creates 2 files:
  - `{name}_results.json` - Full JSON with conversation + evaluation
  - `{name}_report.txt` - Human-readable text report

---

### 6. `cli.py`
**Purpose**: Command-line interface for running evaluations

**Main Functions**:

#### `evaluate_single(filepath, api_key, output_dir, verbose) -> Dict`
- **Input**:
  - `filepath`: Path to conversation JSON
  - `api_key`: Together AI API key
  - `output_dir`: Where to save results
  - `verbose`: Print progress
- **Output**: Result dict with name, pes_score, pes_category, summary
- **Process**:
  1. Load JSON file (extracts conversation + metadata)
  2. Initialize PedagogicalEvaluator
  3. Call `evaluate_conversation()`
  4. Calculate PES
  5. Print summary
  6. Save results
  7. Return result dict

#### `evaluate_batch(directory, api_key, output_dir, verbose)`
- **Input**:
  - `directory`: Path containing multiple JSON files
  - `api_key`: Together AI API key
  - `output_dir`: Where to save results
  - `verbose`: Print progress
- **Output**: None (prints summary, saves files)
- **Process**:
  1. Find all .json files
  2. Call `evaluate_single()` for each
  3. Collect results
  4. Generate batch summary report
  5. Save summary

#### `main()`
- **Input**: Command-line arguments (parsed via argparse)
- **Commands**:
  - `evaluate <file>` - Single evaluation
  - `batch <directory>` - Batch evaluation
- **Process**:
  1. Parse arguments
  2. Get API key (from arg or env)
  3. Route to evaluate_single or evaluate_batch

---

### 7. `example_usage.py`
**Purpose**: Demonstration script showing how to use the Python API

**Input**: Environment variable TOGETHER_API_KEY
**Output**: Prints evaluation results, saves to judge/results/examples/
**Process**:
1. Check for API key
2. Initialize evaluator
3. Evaluate excellent_tutoring example
4. Print results
5. Save detailed report

---

### 8. `test_conversations/` (5 JSON files)
**Purpose**: Test cases with varying pedagogical quality

**JSON Structure**:
```json
{
  "metadata": {
    "name": "conversation_name",
    "description": "...",
    "expected_pes_range": "85-95",
    "domain": "mathematics",
    "topic": "quadratic equations"
  },
  "conversation": [
    {"role": "tutor", "content": "..."},
    {"role": "student", "content": "..."}
  ]
}
```

**Files**:
1. `1_excellent_tutoring.json` - Expected PES: 85-95 (actual: 100)
2. `2_answer_revealer.json` - Expected PES: 55-65
3. `3_passive_engagement.json` - Expected PES: 60-70
4. `4_incoherent_responses.json` - Expected PES: 40-50
5. `5_harsh_tone.json` - Expected PES: 50-60 (actual: 25)

---

## Complete Execution Flow

### Single Evaluation Flow
```
1. USER runs: python -m judge.cli evaluate file.json

2. cli.py main()
   ↓ parses args
   ↓ loads API key
   ↓ calls evaluate_single()

3. evaluate_single()
   ↓ loads JSON file
   ↓ extracts conversation array (NOT metadata)
   ↓ creates PedagogicalEvaluator

4. PedagogicalEvaluator()
   ↓ initializes LLMClient with DeepSeek V3
   ↓ calls evaluate_conversation()

5. evaluate_conversation()
   FOR EACH of 8 dimensions:
     ↓ evaluate_dimension()
       ↓ _format_conversation() → readable text
       ↓ get prompt from prompts.py
       ↓ _call_judge() → DeepSeek V3 API
       ↓ parse JSON response
       ↓ return DimensionScore

   ↓ evaluate_question_depth() → QuestionDepthScore
   ↓ evaluate_icap_engagement() → ICAPScore
   ↓ generate overall summary → summary strings
   ↓ assemble PESComponents

6. Back to evaluate_single()
   ↓ calculate_pes(components) → float (0-100)
   ↓ get_pes_category(pes) → string
   ↓ save_evaluation_results() → creates files

7. report.py generates:
   ↓ {name}_results.json (full data)
   ↓ {name}_report.txt (human-readable)
```

### Batch Evaluation Flow
```
1. USER runs: python -m judge.cli batch directory/

2. cli.py batch command
   ↓ finds all *.json files
   ↓ FOR EACH file:
       ↓ evaluate_single() (see above)
       ↓ collect result dict

3. generate_summary_report()
   ↓ calculates aggregate statistics
   ↓ creates batch_summary.txt
```

---

## Input/Output Summary Table

| File | Input | Output |
|------|-------|--------|
| `__init__.py` | Module import | Package exports |
| `prompts.py` | None (static) | Prompt template strings |
| `metrics.py` | PESComponents | PES score (0-100), category, breakdowns |
| `evaluator.py` | Conversation JSON | PESComponents object |
| `report.py` | PESComponents | Text/JSON reports |
| `cli.py` | Command-line args + JSON files | Saved reports + printed summaries |
| `example_usage.py` | TOGETHER_API_KEY env | Printed results + saved reports |
| `test_conversations/*.json` | None | Sample data for testing |

---

## Key Design Patterns

### 1. Separation of Concerns
- **Prompts** (prompts.py) separate from **Logic** (evaluator.py)
- **Data structures** (metrics.py) separate from **Presentation** (report.py)
- **CLI** (cli.py) separate from **API** (evaluator.py)

### 2. Dataclass-Driven
- All scores stored in typed dataclasses
- Easy serialization to JSON
- Type safety

### 3. Composability
- Can use evaluator directly (Python API)
- Can use CLI for convenience
- Can generate reports independently

### 4. Caching Disabled
- Judge evaluations are NOT cached (enable_cache=False)
- Ensures fresh evaluation each time
- Different from tutoring agent calls

---

## Dependencies

### External APIs
- **Together AI**: DeepSeek V3 model access
- Requires: `TOGETHER_API_KEY` environment variable

### Python Libraries
- `together` - Together AI API client
- `json` - JSON parsing
- `pathlib` - File path handling
- `argparse` - CLI argument parsing
- Standard library imports from parent `src/` (LLMClient)

---

## Extension Points

### Adding New Dimensions
1. Add prompt to `DIMENSION_PROMPTS` in `prompts.py`
2. Add field to `PESComponents` in `metrics.py`
3. Add evaluation call in `evaluate_conversation()` in `evaluator.py`
4. Update `layer1_score()` calculation

### Using Different Judge Models
```python
evaluator = PedagogicalEvaluator(
    judge_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
```

### Custom Scoring Weights
Edit `calculate_pes()` in `metrics.py`:
```python
weighted_score = (layer1 * 0.7) + (layer2 * 0.15) + (layer3 * 0.15)
```
