# AutoSkill: Automatic Skill Discovery and Steering (work in progress)

Code repo for *AutoSkill: Automatic Skill Discovery and Steering for Reasoning LLMs*.

**Author:** Feiyang Kang (`fyk@vt.edu`)
Capstone Milestone Report, Advanced Machine Learning, Spring 2026 (Instructor: Prof. Ming Jin)

## Abstract

As Large Language Models (LLMs) evolve toward complex reasoning tasks with extended solution traces, optimizing their post-training data and inference strategies remains a critical challenge. Current approaches often rely on manual "skill profiling" to curate datasets, but these human-defined taxonomies frequently fail to align with the model's internal representations. In this work, we introduce **AutoSkill**, a fully automated framework for discovering and steering latent reasoning capabilities.

Motivated by the **Linear Representation Hypothesis**, we demonstrate that reasoning skills are encoded as orthogonal directions within the model's high-dimensional activation space. By decomposing activation patterns from mathematical reasoning trajectories, AutoSkill uncovers latent directions with distinct semantic interpretations (e.g., distinguishing *natural language analysis* from *symbolic derivation*). Crucially, we introduce a utility metric that correlates activation strength along these directions with verifiable solution correctness, allowing us to distinguish beneficial skills from failure modes.

This enables steering via two mechanisms:
1. **Targeted SFT data selection** — curating fine-tuning data aligned with beneficial skill directions.
2. **Inference-time activation injection** — permanently modifying model weights with steering vectors.

In initial results, empirical evaluations on Llama3-8B demonstrate that AutoSkill substantially improves mathematical reasoning: **Pass@1 accuracy on MATH-500 and AMC benchmarks are increased by 15% and 87% respectively**, relative to standard baselines. By automating the discovery of effective reasoning granularities, AutoSkill offers a scalable, task-agnostic path for optimizing agentic behaviors.

> *The project is under active development. In the following stage, we will investigate: (i) generalizing the results and findings to different models, (ii) quantifying the predictability of steering outcomes, (iii) comparing the empirical effectiveness of automatically discovered skills with manually-curated ones.*

---

## Repository Structure

```
AutoSkill/
├── activations.ipynb                  # Step 1: Extract activation patterns
├── skills_interp.ipynb                # Step 2: Discover & interpret skills via PCA + LLM
├── steering_via_model_edits.ipynb     # Step 3: Steer models via weight editing
├── evaltask-v6.py                     # Evaluation pipeline (7 benchmarks, Pass@k)
├── l8b-nemo-math-25k-5xlr-e1.yaml    # SFT config: Llama-3-8B
├── q3b-nemo-25k.yaml                 # SFT config: Qwen2.5-3B
└── readme.md
```

---

## Pipeline

### Step 1. Extracting Activations on Reasoning Examples
> `activations.ipynb`

Extracts hidden-state activations from a language model over math reasoning trajectories.

- Processes examples from the [Llama-Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) (SFT/math split)
- Extracts intermediate hidden states from all transformer layers (mean-pooled over tokens), discarding the first and last layers
- Produces an activation matrix of shape `(N_samples, D_hidden)` — e.g., 50,000 samples x 30,720 dimensions
- Output is saved as a pickled gradient matrix for downstream analysis

### Step 2. Discovering Skills from Activations and Semantic Interpretations
> `skills_interp.ipynb`

Discovers latent skill directions via PCA and obtains human-readable interpretations.

- **PCA decomposition:** Reduces the high-dimensional activation space into orthogonal principal components (e.g., 10 directions), each representing an independent reasoning skill axis
- **Semantic labeling:** For each skill direction, identifies the most positively and negatively correlated training examples, then uses contrastive analysis (via LLM) to generate interpretable skill descriptions
- Example discovered axis: *"Exploratory qualitative reasoning"* vs. *"Structured symbolic calculus"*

### Step 3. Steering the Target Model towards an Identified Skill Direction
> `steering_via_model_edits.ipynb`

Implements permanent model steering by injecting skill vectors into model weights.

- For each discovered skill direction, creates both positive (+α) and negative (-α) steered model variants
- **Steering mechanism:** Adds scaled steering vectors as bias offsets to the MLP `down_proj` layers across intermediate transformer layers
- Default steering strength: α = 0.50
- Output: 2K steered model variants (K skills × 2 polarities), saved and ready for evaluation

---

## SFT Training Configurations

Training configs for supervised fine-tuning using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

| Config | Model | Dataset | Max Samples | Seq Len | LR | Method |
|--------|-------|---------|-------------|---------|-----|--------|
| `l8b-nemo-math-25k-5xlr-e1.yaml` | Llama-3-8B-Instruct | Nemotron SFT/math | 25,000 | 8,192 | 5e-5 | Full fine-tuning (DeepSpeed ZeRO-3) |
| `q3b-nemo-25k.yaml` | Qwen2.5-3B | Nemotron SFT/math | 25,000 | 8,192 | 5e-5 | Full fine-tuning (DeepSpeed ZeRO-3) |

Both configs use: 1 epoch, cosine LR schedule, 10% warmup, bfloat16 precision, Flash Attention 2, effective batch size 2048.

---

## Evaluation Pipeline
> `evaltask-v6.py`

Comprehensive evaluation across 7 mathematical reasoning benchmarks with Pass@k scoring.

### Supported Benchmarks

| Task ID | Dataset | Description |
|---------|---------|-------------|
| `math500` | MATH-500 | Competition math (lighteval) |
| `aime1k` | AIME 1983–2024 | AME/AIME problems (1,000 questions) |
| `gsm8k` | GSM8K | Grade school math (8,000 questions) |
| `aime25` | AIME 2025 | Latest AIME competition |
| `amc` | AMC | AMC validation problems |
| `olympiad` | OlympiadBench | International math olympiad problems |
| `minerva` | MinervaAuth | Minerva math dataset |

### Usage

```bash
python evaltask-v6.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tasks math500 aime1k gsm8k aime25 amc olympiad minerva \
  --rollouts 256 \
  --temperature 1.0 \
  --gen_len 8000 \
  --max_sample 500 \
  --output_path l3b8-v-8k-t100 \
  --full_logs True \
  --sys_prompt rl_prompt \
  --devices '0,1,2,3,5,6,7' \
  --seed 42 \
  --verbose True
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--tasks` | *required* | Benchmark names (space-separated) |
| `--rollouts` | 64 | Number of rollouts per question (must be power of 2) |
| `--temperature` | 1.0 | Sampling temperature |
| `--gen_len` | *required* | Max generation length in tokens |
| `--max_sample` | -1 | Max samples per task (-1 = full dataset) |
| `--output_path` | *required* | Output directory for results |
| `--full_logs` | False | Save complete generated traces |
| `--sys_prompt` | `no_prompt` | System prompt: `no_prompt` or `rl_prompt` (step-by-step reasoning) |
| `--devices` | `0,1,2,3,4,5,6,7` | CUDA device indices (comma-separated) |
| `--seed` | 42 | Random seed |
| `--verbose` | False | Output per-sample accuracy and full traces |

### Output Files

- `{output_path}_EvalRes.txt` — Pass@k scores and entropy metrics
- `Verb_crr_{output_path}_EvalCrrs.txt` — Per-sample accuracy matrix (if `--verbose`)
- `Verb_trace_{output_path}_EvalTexts.txt` — Full Q&A traces (if `--full_logs`)
