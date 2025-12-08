# TeachTime Quick Start Guide - SocraticLM Edition

## ✅ Installation Complete!

You've successfully migrated to **SocraticLM** - a fine-tuned model specifically designed for Socratic teaching and mathematical problem solving.

---

## 🚀 Running Your First Experiment

### Option 1: Run a Single Experiment

```bash
# Experiment 1: Pedagogical Metrics Match
python experiments/run_experiment.py --config exp1_metrics_match

# Experiment 2: Learning Gains
python experiments/run_experiment.py --config exp2_learning_gains

# Experiment 3: Persona Robustness
python experiments/run_experiment.py --config exp3_persona_robustness

# Experiment 4: ReAct Ablation
python experiments/run_experiment.py --config exp4_react_ablation
```

### Option 2: Run with Verbose Output

```bash
# See detailed output including model loading and generation
python experiments/run_experiment.py --config exp1_metrics_match --verbose
```

### Option 3: Run All Experiments

```bash
# Run the full experiment suite (all 4 experiments)
python experiments/run_experiment_suite.py
```

---

## 📊 What to Expect

### First Run (Model Download)
- **Download size**: ~14GB (SocraticLM model)
- **Download time**: 5-10 minutes (depending on internet speed)
- **Storage**: Model is cached at `~/.cache/huggingface/`
- **Status**: You'll see "Downloading..." messages from HuggingFace

### Subsequent Runs (Using Cached Model)
- **Load time**: 30-60 seconds (loading model into memory)
- **Per episode**: ~10-30 seconds (depends on conversation length)
- **Full experiment**: ~30-60 minutes (depends on num_episodes_per_condition)

### Performance Notes
- ✅ **Your system**: Mac with MPS (Metal) GPU acceleration detected
- ✅ **Speed**: Should be reasonably fast with GPU acceleration
- ⚠️ **Memory**: If you encounter memory issues, see Troubleshooting below

---

## 📁 Results Location

After running experiments, find your results here:

```
results/
├── raw/                    # Full episode trajectories (JSONL format)
│   └── exp1_*.jsonl
├── processed/              # Aggregated summaries (CSV format)
│   └── exp1_*_summary.csv
└── plots/                  # Generated visualizations
    └── *.png
```

---

## 🔬 Understanding the Experiments

### Experiment 1: Pedagogical Metrics Match
**Goal**: Compare how well different agents match target teaching behaviors

**Agents Tested**:
- Baseline-CoT (standard chain-of-thought)
- Metric-CoT (CoT with metrics awareness)
- ReAct-Teacher (full ReAct with discrete actions)

**Metrics Evaluated**:
- Student talk ratio (target: 0.5-0.8)
- Questions asked per session (target: ~5)
- Background probing timing (target: by turn 3)
- Understanding checks (target: ~3 per session)

```bash
python experiments/run_experiment.py --config exp1_metrics_match
```

### Experiment 2: Learning Gains
**Goal**: Measure actual learning outcomes on simulated students

**Setup**: Pre-quiz → Tutoring (10 turns) → Post-quiz

**Metrics**:
- Normalized learning gain
- Knowledge state improvement
- Quiz score delta

```bash
python experiments/run_experiment.py --config exp2_learning_gains
```

### Experiment 3: Persona Robustness
**Goal**: Test how agents adapt to different student types

**Student Personas**:
1. **Struggling**: Low prior knowledge, frequent confusion
2. **Confident but Mistaken**: High confidence, systematic misconceptions
3. **Minimal Talker**: Very terse responses (challenges talk ratio)

**Analysis**: Agent × persona performance matrix

```bash
python experiments/run_experiment.py --config exp3_persona_robustness
```

### Experiment 4: Ablation Study
**Goal**: Decompose the contribution of ReAct framework vs. metrics awareness

**Variants**:
- ReAct-Teacher (full: ReAct + metrics)
- ReAct-Teacher-No-Metrics (ReAct only)
- Baseline-CoT (neither)

```bash
python experiments/run_experiment.py --config exp4_react_ablation
```

---

## 📈 Analyzing Results

### Using Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Open analysis notebooks (if available)
# Or create your own analysis
```

### Quick Analysis with Python

```python
import pandas as pd

# Load experiment results
df = pd.read_csv("results/processed/exp1_metrics_match_summary.csv")

# Compare student talk ratios by agent
print(df.groupby("agent_type")["metric_student_talk_ratio"].describe())

# Compare learning gains
df2 = pd.read_csv("results/processed/exp2_learning_gains_summary.csv")
print(df2.groupby("agent_type")["learning_gain"].mean())
```

---

## 🎯 Using SocraticLM for Both Teacher and Student

**Good news!** Your setup now uses SocraticLM for **both**:

1. **Teacher Agents** (CoT, MetricCoT, ReAct)
   - Generates pedagogical responses
   - Uses Socratic questioning techniques
   - Adapts to student knowledge state

2. **Simulated Students** (all 3 personas)
   - Responds based on internal knowledge state
   - Exhibits persona-specific behaviors
   - Simulates realistic student interactions

This is ideal because SocraticLM was specifically fine-tuned for educational dialogue!

---

## ⚙️ Configuration

All experiments use these config files:

### Model Configuration (`configs/model.yaml`)
```yaml
default_model: "CogBase-USTC/SocraticLM"
temperature: 0.7
max_tokens: 300
enable_cache: true
```

### Environment Configuration (`configs/env.yaml`)
```yaml
max_turns: 10
enable_quizzes: true
default_domain: "mathematics"
```

### Experiment Configs (`configs/experiments/*.yaml`)
Each experiment has its own config file specifying:
- Which agents to compare
- Which tasks to use
- Which student personas to test
- Number of episodes to run
- Random seed for reproducibility

---

## 🐛 Troubleshooting

### Model Download Issues

**Problem**: Download fails or times out

**Solution**:
```bash
# Manually download the model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('CogBase-USTC/SocraticLM')
AutoModelForCausalLM.from_pretrained('CogBase-USTC/SocraticLM')
"
```

### Memory Issues

**Problem**: "Out of memory" errors

**Solutions**:

1. **Reduce max_tokens** in `configs/model.yaml`:
```yaml
max_tokens: 150  # Reduced from 300
```

2. **Use CPU instead of GPU** (slower but uses less memory):
Edit `src/llm/llm_client.py` line 185:
```python
device_map="cpu",  # Instead of "auto"
```

3. **Close other applications** to free up memory

### Slow Generation

**Problem**: Generation is very slow

**Check**:
```bash
# Verify GPU is being used
python experiments/run_experiment.py --config exp1_metrics_match --verbose

# Look for: "Model loaded successfully on device: mps"
# If you see "device: cpu", GPU is not being used
```

**Solutions**:
- Ensure no other heavy processes are using GPU
- Reduce `max_tokens` in config
- Use smaller batch sizes (if applicable)

### Cache Not Working

**Problem**: Same prompts generating different responses

**Check**:
```bash
# Verify cache is enabled
cat configs/model.yaml | grep enable_cache
# Should show: enable_cache: true

# Check cache database exists
ls -lh .cache/llm_cache.db
```

**Clear cache** (if needed):
```bash
rm -rf .cache/llm_cache.db
```

---

## 📚 Key Differences from Previous Setup

| Aspect | Before (Together AI) | Now (SocraticLM) |
|--------|---------------------|------------------|
| **Model Location** | Cloud API | Local on your machine |
| **Cost** | Pay per API call | Free (one-time download) |
| **Privacy** | Data sent to API | Fully private/local |
| **Internet** | Required for each call | Only for initial download |
| **Model Size** | 70B parameters | 7B parameters (fine-tuned) |
| **Specialization** | General instruction | Socratic teaching + math |
| **API Key** | Required | Not needed |

---

## 🎓 About SocraticLM

**Model**: CogBase-USTC/SocraticLM
**Base**: Qwen2.5-Math-7B-Instruct
**Training**: Fine-tuned on SocraTeach dataset
**Specialization**:
- Socratic-style teaching guidance
- Mathematical problem solving
- Heuristic questioning techniques

**Paper/Source**: Implementation of SocraticLM methodology for educational AI

---

## 📞 Need Help?

1. **Check logs**: Look at verbose output with `--verbose` flag
2. **Verify installation**: Run `python -c "import src; print('OK')"`
3. **Check GPU**: Run `python -c "import torch; print(torch.backends.mps.is_available())"`
4. **Review configs**: Ensure YAML files are properly formatted

---

## 🚦 Quick Health Check

Run this to verify everything is working:

```bash
# Test imports
python -c "import src; from src.llm.llm_client import create_llm_client; print('✓ All imports OK')"

# Check GPU availability
python -c "import torch; print(f'✓ MPS GPU: {torch.backends.mps.is_available()}')"

# List experiment configs
ls configs/experiments/
```

---

## 🎉 You're Ready!

Everything is set up. Start with:

```bash
python experiments/run_experiment.py --config exp1_metrics_match --verbose
```

This will:
1. Download SocraticLM (first time only, ~14GB)
2. Load the model into memory (~30-60 seconds)
3. Run Experiment 1 with all agent comparisons
4. Save results to `results/` directory
5. Print summary statistics

**Enjoy experimenting with SocraticLM!** 🎓✨
