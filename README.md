# TeachTime: A ReAct Framework for Metric-Guided Language Model Tutors

**A research framework for building and evaluating LLM tutors that explicitly reason over teaching metrics using the ReAct paradigm.**

---

## Overview

TeachTime combines **ReAct** (Reason + Act interleaving) with **teaching quality metrics** from tutoring literature to create more effective AI tutors. Instead of generating free-form responses, our ReAct-Teacher agent:

1. **Observes** teaching metrics (student talk ratio, questions asked, understanding checks, etc.) as part of the environment state
2. **Reasons** explicitly about these metrics in its chain-of-thought
3. **Acts** by choosing from discrete pedagogical primitives (Ask_Open_Question, Give_Explanation, etc.)

We compare three agent architectures:
- **Baseline-CoT**: Standard helpful-tutor with chain-of-thought
- **Metric-CoT**: CoT with metrics in prompt, but no action space
- **ReAct-Teacher**: Full ReAct loop with metrics + discrete pedagogical actions

## Key Features

- 🎯 **Metric-guided tutoring**: Observable teaching quality metrics (talk ratio, questions, checks)
- 🔄 **ReAct framework**: Thought → Pedagogical_Action → Observation loop
- 🧠 **LLM-based student simulation**: 3 personas with internal knowledge states
- 📊 **AppBench-style evaluation**: Structured scoring across pedagogy, learning, and quality
- 🔬 **Full experiment suite**: 4 main experiments + human pilot interface
- 💾 **Reproducible**: Seeded randomness + LLM response caching

---

## Installation

### Prerequisites
- Python 3.9+
- Together AI API key (get one at https://api.together.xyz)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd teach-time
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
```

4. Verify installation:
```bash
python -c "import src; print('✓ TeachTime installed successfully')"
```

---

## Quick Start

### Run a Single Experiment

```bash
# Run Experiment 1: Pedagogical Metrics Match
python experiments/run_experiment.py --config exp1_metrics_match

# Run with verbose output
python experiments/run_experiment.py --config exp2_learning_gains --verbose
```

### Run Full Experiment Suite

```bash
# Run all 4 experiments (Experiments 1-4)
python experiments/run_experiment_suite.py
```

### Human Pilot Study

```bash
# Interact with a tutor agent yourself
python experiments/run_human_pilot.py
```

### Analyze Results

```bash
# Launch Jupyter to run analysis notebooks
jupyter notebook notebooks/
```

---

## Architecture

### Core Components

#### 1. **Environment** (`src/env/`)

- **TeachingEnv**: Gym-like environment with pre/post quizzes
- **Metrics**: Computes teaching quality metrics from dialogue
- **Tasks**: Math domain (algebra, functions, fractions, probability)
- **Student Models**: 3 LLM-based personas:
  - **Struggling**: Low prior knowledge, high confusion
  - **Confident but Mistaken**: Systematic misconceptions
  - **Minimal Talker**: Terse responses, challenges talk ratio

#### 2. **Agents** (`src/agents/`)

- **CoTTutorAgent**: Baseline chain-of-thought tutor
- **MetricCoTTutorAgent**: CoT + metrics awareness
- **ReActTeacherAgent**: Full ReAct with discrete actions

#### 3. **Pedagogical Actions** (`src/env/actions.py`)

Six discrete teaching moves:
- `Ask_Open_Question`: Encourage elaboration
- `Ask_Check_Understanding`: Verify comprehension
- `Give_Step_By_Step_Explanation`: Provide guidance
- `Ask_Background`: Probe prior knowledge
- `Assign_Practice_Problem`: Practice application
- `Summarize_And_Wrap_Up`: Reinforce learning

#### 4. **Teaching Metrics** (`src/env/metrics.py`)

Observable metrics derived from tutoring research:
- `student_talk_ratio`: Student tokens / total tokens
- `num_questions_asked`: Questions by tutor
- `asked_background`: Has prior knowledge been probed?
- `checks_of_understanding_last_k_turns`: Recent comprehension checks
- Plus: confusion indicators, practice problems, etc.

#### 5. **Scoring System** (`src/eval/`)

AppBench-style structured evaluation:
- **Pedagogy Score** (40%): Adherence to teaching best practices
- **Learning Score** (40%): Pre→post quiz improvement
- **Quality Score** (20%): Conversation quality (LLM-as-judge)

---

## Experiments

### Experiment 1: Pedagogical Metrics Match

**Goal**: Compare agents on ability to match target teaching behaviors

- **Agents**: Baseline-CoT, Metric-CoT, ReAct-Teacher
- **Metrics**: student talk ratio, questions, background probing, understanding checks
- **Expected Result**: ReAct-Teacher > Metric-CoT > Baseline-CoT

```bash
python experiments/run_experiment.py --config exp1_metrics_match
```

### Experiment 2: Learning Gains

**Goal**: Measure learning outcomes on simulated students

- **Setup**: Pre-quiz → Tutoring → Post-quiz
- **Metrics**: Normalized learning gain, knowledge state improvement
- **Expected Result**: ReAct-Teacher shows higher mean learning gain

```bash
python experiments/run_experiment.py --config exp2_learning_gains
```

### Experiment 3: Persona Robustness

**Goal**: Test adaptation to diverse student behaviors

- **Personas**: Struggling, Confident-but-Mistaken, Minimal-Talker
- **Analysis**: Agent × persona performance matrix
- **Expected Result**: ReAct-Teacher degrades less across personas

```bash
python experiments/run_experiment.py --config exp3_persona_robustness
```

### Experiment 4: Ablation Study

**Goal**: Decompose contribution of ReAct vs. metrics

- **Variants**:
  - ReAct-Teacher (full)
  - ReAct-Teacher without metrics
  - Baseline-CoT (reference)
- **Expected Result**: Full > No Metrics > Baseline

```bash
python experiments/run_experiment.py --config exp4_react_ablation
```

---

## Configuration

All experiments are configured via YAML files in `configs/`:

### `configs/model.yaml`
```yaml
default_model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
temperature: 0.7
max_tokens: 300
enable_cache: true
```

### `configs/env.yaml`
```yaml
max_turns: 10
enable_quizzes: true
default_domain: "mathematics"
```

### `configs/scoring.yaml`
```yaml
weights:
  pedagogy: 0.4
  learning: 0.4
  quality: 0.2

pedagogy:
  student_talk_ratio_range: [0.5, 0.8]
  target_questions_per_session: 5
  background_asked_by_turn: 3
  target_understanding_checks: 3
```

---

## Results

After running experiments, results are saved to:

- `results/raw/` - JSONL trajectories (full episode logs)
- `results/processed/` - Aggregated CSV summaries
- `results/plots/` - Generated figures

### Example Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv("results/processed/exp1_metrics_match_summary.csv")

# Compare student talk ratios
df.groupby("agent_type")["metric_student_talk_ratio"].mean()
```

---

## Reproducibility

TeachTime is designed for full reproducibility:

1. **Seeded randomness**: All experiments use fixed random seeds
2. **LLM caching**: Responses cached in SQLite (`.cache/llm_cache.db`)
3. **Config snapshots**: Full config saved with each experiment run
4. **Version tracking**: Environment version recorded in configs

To reproduce paper results:
```bash
# Clear cache (optional, to start fresh)
rm -rf .cache/

# Run full suite with default seeds
python experiments/run_experiment_suite.py
```

---

## Development

### Project Structure

```
teach-time/
├── configs/              # YAML configurations
│   ├── experiments/      # Per-experiment configs
│   ├── env.yaml
│   ├── model.yaml
│   └── scoring.yaml
├── src/
│   ├── env/              # Environment, tasks, students, metrics
│   ├── agents/           # Tutor agent implementations
│   ├── llm/              # LLM client + prompts
│   ├── eval/             # Scoring + evaluation loops
│   └── utils/            # Config, logging, random utils
├── experiments/          # Experiment runners
├── notebooks/            # Analysis notebooks
├── results/              # Experiment outputs
└── paper/                # Paper outline + figures
```

### Adding a New Agent

```python
from src.agents.base_agent import BaseTutorAgent

class MyAgent(BaseTutorAgent):
    def act(self, observation):
        # Your logic here
        return PedagogicalAction(...)
```

### Adding a New Task

```python
from src.env.tasks.base import TaskSpec, QuizQuestion

task = TaskSpec(
    task_id="my_task",
    topic="My Topic",
    # ... define learning objectives, quizzes, etc.
)
```

---

## Citation

If you use TeachTime in your research, please cite:

```bibtex
@article{teachtime2025,
  title={TeachTime: A ReAct Framework for Metric-Guided Language Model Tutors},
  author={[Your Name]},
  year={2025},
  note={Research project}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Contact & Contributing

- **Issues**: https://github.com/[your-repo]/teach-time/issues
- **Discussions**: For questions about using TeachTime
- **Pull Requests**: Contributions welcome!

---

## Acknowledgments

- **ReAct**: [Yao et al., 2022] - Reasoning and Acting framework
- **TeachLM**: [Sonkar et al., 2024] - LLM tutoring research
- **AppBench**: [Mao et al., 2024] - Structured evaluation approach
- **Together AI**: For providing LLM inference infrastructure
