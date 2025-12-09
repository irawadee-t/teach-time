# Synthetic Data Generation Plan for TutorBench Finetuning

## Overview

Goal: Generate high-quality synthetic tutoring responses to finetune smaller text-only models that achieve strong performance on TutorBench.

**Approach:** Generate-then-Grade (Rejection Sampling) with persona-based diversity conditioning.

**Key Design Decisions:**
- 8 tutoring personas targeting specific skill gaps identified in TutorBench paper (Figure 5)
- Qwen/Qwen3-Next-80B-A3B-Instruct for generation (59.5% on benchmark)
- Claude Sonnet for grading (F1=0.81 vs human experts per TutorBench Section 4.7)
- Text-only focus (Use Case 1: Adaptive Explanation Generation)

---

## Phase 1: Data Preparation

### 1.1 Train/Test Split ✓ COMPLETED

- **Test set:** 100 samples held out for final evaluation (`evals/test.csv`)
- **Train set:** 556 samples for synthetic data generation (`evals/train.csv`)
- **Stratification:** Balanced by SUBJECT (6 categories)

```python
# Executed via evals/split_data.py
train_df, test_df = train_test_split(
    df,
    test_size=100,
    stratify=df['SUBJECT'],
    random_state=42
)
```

### 1.2 Scope Limitations

**What we're covering:**
- Use Case 1: Adaptive Explanation Generation (multi-turn: question → explanation → follow-up)
- Text-only samples across 6 STEM subjects

**What we're NOT covering:**
- Use Case 2: Assessment & Feedback
- Use Case 3: Active Learning/Hints
- Multimodal samples (56% of TutorBench)

This is intentional - we're optimizing for Use Case 1 performance with text-only models.

---

## Phase 2: Synthetic Response Generation

### 2.1 Generation Strategy

**DO NOT include rubrics in generation prompt.** This preserves natural response diversity and prevents "teaching to the test."

### 2.2 Persona-Based Diversity

Generate multiple responses per sample using 8 tutor personas. These personas are designed to target specific tutoring skill gaps identified in TutorBench paper Figure 5:

| TutorBench Skill (Avg Score) | Targeted By Persona |
|------------------------------|---------------------|
| Includes examples/analogy (32.8%) | `analogy_builder` |
| Provides alternative solutions (41.9%) | `alternative_pathfinder` |
| Asks questions to guide (45%) | `socratic_questioner` |
| Step by step help (48%) | `step_by_step_guide` |
| Identifying core misconception (50%) | `error_pattern_expert`, `direct_clarifier` |
| Emotional component (varies) | `empathetic_validator` |

```python
TUTOR_PERSONAS = [
    {
        "name": "empathetic_validator",
        "description": "A patient tutor who always acknowledges the student's feelings and validates their effort before addressing misconceptions. Uses phrases like 'I understand why that's confusing' and 'That's a really thoughtful question.'"
    },
    {
        "name": "socratic_questioner",
        "description": "A Socratic tutor who primarily guides through questions rather than direct explanation. Asks probing questions like 'What do you think would happen if...' and 'Can you walk me through your reasoning?'"
    },
    {
        "name": "direct_clarifier",
        "description": "A clear, direct tutor who efficiently identifies the exact misconception and provides targeted correction. Gets to the point quickly while remaining supportive."
    },
    {
        "name": "analogy_builder",
        "description": "A tutor who excels at relating abstract concepts to concrete, everyday examples. Frequently uses analogies and real-world scenarios to build intuition."
    },
    {
        "name": "step_by_step_guide",
        "description": "A methodical tutor who breaks complex problems into small, manageable steps. Numbers their explanations and checks understanding at each stage."
    },
    {
        "name": "conceptual_connector",
        "description": "A tutor who emphasizes how concepts relate to the bigger picture. Draws connections between current topic and previously learned material."
    },
    {
        "name": "error_pattern_expert",
        "description": "A tutor who specializes in identifying common error patterns. Explicitly names the type of mistake and explains why it's a common pitfall."
    },
    {
        "name": "alternative_pathfinder",
        "description": "A tutor who presents multiple ways to approach problems. When explaining a concept, offers 'another way to think about this...' and shows different solution strategies or perspectives."
    }
]
```

**Why 8 fixed personas instead of combinatorial expansion:** At our scale (556 samples × 8 = 4,448 generations), each persona is used ~556 times - enough to learn meaningful patterns. With 192 combinatorial personas, each would only appear ~23 times, providing insufficient signal.

### 2.3 Generation Prompt Template

```
System prompt:
You are {persona_description}

You are tutoring a high school student in {subject}. Respond to their follow-up question naturally, as an expert tutor would.

---

User prompt:
## Original Question
{PROMPT}

## Initial Explanation Provided
{UC1_INITIAL_EXPLANATION}

## Student's Follow-up Question
{FOLLOW_UP_PROMPT}

---

Provide a helpful tutoring response to the student's follow-up question.
```

### 2.4 Generation Parameters

- **Model:** Qwen/Qwen3-Next-80B-A3B-Instruct via Together AI
- **Generations per sample:** 8 (one per persona)
- **Temperature:** 0.8 (balance quality and diversity)
- **Max tokens:** 1500 (tutoring responses should be focused, not essays)
- **Concurrency:** 50 parallel requests with checkpointing every 50 samples

---

## Phase 3: Quality Grading

### 3.1 Grading Setup

Grade each generated response against the sample's RUBRIC using a separate LLM call. **The grading model should NOT see the generation prompt or persona.**

### 3.2 Rubric Parsing

Each rubric criterion has attributes:
```json
{
    "criteria": "The response must...",
    "severity": "critical" | "not_critical",
    "tutoring_skill": "Identifying Core difficulty/ misconception attribution" | ...,
    "eval_dimension": "instruction_following" | "truthfulness" | ...
}
```

### 3.3 Scoring Formula

Following the TutorBench paper's weighted scoring:

```python
def score_response(rubric_results: list[dict]) -> float:
    """
    rubric_results: list of {criterion, severity, passed: bool}
    """
    total_weight = 0
    earned_weight = 0

    for r in rubric_results:
        # Assign weights based on severity
        if r['severity'] == 'critical':
            # Check if it's a negative criterion (penalizes bad behavior)
            if is_negative_criterion(r['criterion']):
                weight = -5
            else:
                weight = 5
        else:
            weight = 1

        # Only count positive weights in denominator
        if weight > 0:
            total_weight += weight
            if r['passed']:
                earned_weight += weight
        else:
            # Negative weight: subtract if criterion is violated
            if not r['passed']:
                earned_weight += weight  # Adds negative value

    return earned_weight / total_weight if total_weight > 0 else 0
```

### 3.4 Grading Prompt Template

```
System prompt:
You are an expert evaluator assessing tutoring response quality. For each criterion, determine if the response satisfies it. Be strict but fair.

---

User prompt:
## Context
Subject: {subject}
Original Question: {PROMPT}
Initial Explanation: {UC1_INITIAL_EXPLANATION}
Student Follow-up: {FOLLOW_UP_PROMPT}

## Response to Evaluate
{generated_response}

## Evaluation Criteria
{for each criterion in RUBRICS}
Criterion {i}: {criterion_text}
Severity: {severity}
{end for}

---

For each criterion, respond with:
- Criterion {i}: PASS or FAIL
- Brief justification (1 sentence)

Then provide overall score.
```

### 3.5 Detecting Negative Criteria

Negative criteria (weight -5) typically contain phrases like:
- "must NOT"
- "should NOT"
- "must not reveal"
- "must not give away"
- "should avoid"

```python
NEGATIVE_PATTERNS = [
    r"must\s+not",
    r"should\s+not",
    r"must\s+avoid",
    r"should\s+avoid",
    r"must\s+not\s+reveal",
    r"must\s+not\s+give\s+away",
    r"must\s+not\s+state\s+.*\s+answer"
]

def is_negative_criterion(criterion_text: str) -> bool:
    return any(re.search(p, criterion_text, re.IGNORECASE) for p in NEGATIVE_PATTERNS)
```

---

## Phase 4: Filtering and Curation

### 4.1 Quality Threshold

**Two-tier filtering strategy:**
1. **Hard requirement:** 100% pass rate on critical criteria (no exceptions)
2. **Soft requirement:** Overall weighted score >= 0.80 (80%)

This ensures:
- Zero tolerance for critical failures (wrong answers, giving away solutions, ignoring misconceptions)
- Flexibility on non-critical criteria (formatting, analogies, etc.) to preserve tutoring style diversity
- More training data than a blanket 90% filter

**Expected pass rate:** ~40-45% → ~60K high-quality examples

### 4.2 Filtering Implementation

```python
def filter_response(response: dict) -> bool:
    """
    response: {
        response_text: str,
        weighted_score: float,
        rubric_results: list[{criterion, severity, passed}]
    }
    """
    # Calculate critical-only pass rate
    critical_criteria = [r for r in response['rubric_results'] if r['severity'] == 'critical']
    if critical_criteria:
        critical_pass_rate = sum(1 for r in critical_criteria if r['passed']) / len(critical_criteria)
    else:
        critical_pass_rate = 1.0

    # Must pass ALL critical criteria
    if critical_pass_rate < 1.0:
        return False

    # Overall score at least 80%
    if response['weighted_score'] < 0.80:
        return False

    return True
```

### 4.3 Diversity Preservation

For each source sample, keep up to 3 passing responses **from different personas** to preserve tutoring style diversity.

```python
def select_diverse_responses(sample_id: str, responses: list[dict]) -> list[dict]:
    """
    responses: list of {response_text, score, persona}
    """
    # Filter by quality threshold
    passing = [r for r in responses if filter_response(r)]

    # Group by persona, take best from each
    by_persona = {}
    for r in passing:
        persona = r['persona']
        if persona not in by_persona or r['weighted_score'] > by_persona[persona]['weighted_score']:
            by_persona[persona] = r

    # Sort by score and take top 3 different personas
    sorted_responses = sorted(by_persona.values(), key=lambda x: x['weighted_score'], reverse=True)
    return sorted_responses[:3]
```

**Why persona-based selection instead of clustering:** Simpler to implement, no dependency on sentence-transformers, and achieves the same goal - ensuring stylistic diversity by construction rather than post-hoc analysis.

### 4.4 Cross-Sample Deduplication

Remove near-duplicate responses across the entire dataset to prevent repetitive patterns:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_dataset(responses: list[dict], threshold: float = 0.9) -> list[dict]:
    """Remove responses that are >90% similar to earlier responses."""
    if not responses:
        return responses

    vectorizer = TfidfVectorizer(max_features=5000)
    texts = [r['response_text'] for r in responses]
    tfidf_matrix = vectorizer.fit_transform(texts)

    unique = [responses[0]]
    unique_indices = [0]

    for i in range(1, len(responses)):
        # Compare against all kept responses
        sims = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[unique_indices]).flatten()
        if sims.max() < threshold:
            unique.append(responses[i])
            unique_indices.append(i)

    return unique
```

**Why TF-IDF instead of MinHash LSH:** At our scale (~1,600 final responses), O(n²) pairwise comparison is fast enough, and TF-IDF is simpler with no additional dependencies.

### 4.5 Anti-Gaming Filter

Reject responses that appear to literally quote rubric language (suggests gaming if rubric somehow leaked):

```python
def check_for_rubric_parroting(response: str, rubric_criteria: list[str]) -> bool:
    """Returns True if response appears to parrot rubric language."""
    for criterion in rubric_criteria:
        # Extract key phrases from criterion (skip common words)
        key_phrases = extract_key_phrases(criterion)
        matches = sum(1 for phrase in key_phrases if phrase.lower() in response.lower())
        if matches >= 3:  # Too many exact matches is suspicious
            return True
    return False
```

---

## Phase 5: Training Data Formatting

### 5.1 Output Format

Format for instruction finetuning:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an expert tutor helping a high school student."
        },
        {
            "role": "user",
            "content": "## Question\n{PROMPT}\n\n## Previous Explanation\n{UC1_INITIAL_EXPLANATION}\n\n## Student's Follow-up\n{FOLLOW_UP_PROMPT}"
        },
        {
            "role": "assistant",
            "content": "{high_quality_generated_response}"
        }
    ],
    "metadata": {
        "source_sample_id": "...",
        "subject": "...",
        "bloom_taxonomy": "...",
        "generation_persona": "...",
        "quality_score": 0.85
    }
}
```

### 5.2 Data Mixing (Collapse Prevention)

Following "Is Model Collapse Inevitable?" - accumulate, don't replace:

- If you have any human-written gold responses, include them
- Mix synthetic data with original benchmark context
- During finetuning, consider mixing with general instruction data (e.g., 70-80% tutoring, 20-30% general)

**Note:** Specific mixing ratios should be determined empirically during training. The key principle is to never train solely on synthetic data.

---

## Phase 6: Evaluation

### 6.1 Held-Out Test Set

Evaluate finetuned model on the 100 held-out samples using:
1. Same grading pipeline as curation (LLM judge + rubrics)
2. Compare against baseline model (pre-finetune)
3. Compare against Opus and other frontier models

### 6.2 Metrics

- **Primary:** Weighted Average Rubric Rating (ARR_w) - same as TutorBench paper
- **Secondary breakdowns:**
  - By subject
  - By bloom taxonomy level
  - By tutoring skill (from rubric tags)
  - By evaluation dimension

### 6.3 Ablation Studies

Consider testing:
1. Impact of persona diversity (single persona vs. multi-persona)
2. Quality threshold sensitivity (60% vs 70% vs 80%)
3. Number of training examples (learning curves)

---

## Expected Outputs

| Phase | Output | Estimated Count |
|-------|--------|-----------------|
| 1 | Train/test split | 556 train / 100 test |
| 2 | Raw generations | ~4,448 responses (556 × 8 personas) |
| 3 | Graded responses | ~4,448 with scores |
| 4 | Filtered high-quality (100% critical + 80% overall) | ~1,800-2,000 responses |
| 4b | After deduplication | ~1,600-1,800 responses |
| 5 | Training JSONL | ~1,600-1,800 examples |

---

## Implementation Checklist

- [x] Parse and validate CSV structure
- [x] Implement stratified train/test split (`evals/split_data.py`)
- [x] Set up generation pipeline with persona rotation (`evals/synthetic/generate_samples_v2.py`)
- [ ] Add 8th persona (alternative_pathfinder) to generation script
- [ ] Implement rubric parser (handle JSON in RUBRICS column)
- [ ] Build grading pipeline with weighted scoring
- [ ] Implement quality filtering logic
- [ ] Implement cross-sample deduplication
- [ ] Format output for finetuning
- [ ] Run generation (batched, with checkpointing)
- [ ] Run grading (batched, with checkpointing)
- [ ] Filter, deduplicate, and export training data
- [ ] Evaluate baseline on test set
- [ ] Finetune model
- [ ] Evaluate finetuned model on test set

---

## Cost Estimation

### Qwen generation + Claude Sonnet grading (Recommended)

**Generation phase (Together AI - Qwen 80B):**
- 556 samples × 8 personas = ~4,448 generations
- Cost: ~$5-15 (Together pricing is very low)

**Grading phase (Claude Sonnet - $3/M input, $15/M output):**
- 4,448 responses to grade
- Input: 4,448 × 2.5K tokens = ~11M tokens → $33
- Output: 4,448 × 300 tokens = ~1.3M tokens → $20
- **Subtotal: ~$55**

**Total estimated: ~$65-70**
**With batch pricing (50% off): ~$35-40**

---

## References

- TutorBench paper: Weighted scoring formula, rubric structure
- Persona paper: Diversity through persona conditioning
- Model collapse papers: Data accumulation strategy
- Best practices survey: Quality filtering over quality conditioning
