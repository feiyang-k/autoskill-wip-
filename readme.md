# AutoSkill: Automatic Skill Discovery and Steering
Code repo for *AutoSkill: Automatic Skill Discovery and Steering for Reasoning LLMs*.

**Author:**
  Feiyang Kang (\texttt{fyk@vt.edu})\\
  Capstone Milestone Report\\
  Advanced Machine Learning, Spring 2026 (Instructor: Prof. Ming Jin) \\
  Report date: 3/17/2026. \\

**Abstract:**

As Large Language Models (LLMs) evolve toward complex reasoning tasks with extended solution traces, optimizing their post-training data and inference strategies remains a critical challenge. Current approaches often rely on manual "skill profiling" to curate datasets, but these human-defined taxonomies frequently fail to align with the model's internal representations. In this work, we introduce AutoSkill, a fully automated framework for discovering and steering latent reasoning capabilities. Motivated by the Linear Representation Hypothesis, we demonstrate that reasoning skills are encoded as orthogonal directions within the model's high-dimensional activation space. By decomposing activation patterns from mathematical reasoning trajectories, AutoSkill uncovers latent directions with distinct semantic interpretations (e.g., distinguishing *natural language analysis* from *symbolic derivation*). Crucially, we introduce a utility metric that correlates activation strength along these directions with verifiable solution correctness, allowing us to distinguish beneficial skills from failure modes. This enables steering via two mechanisms: (1) targeted selection of Supervised Fine-Tuning (SFT) data, and (2) inference-time activation injection. In initial results, empirical evaluations on Llama3-8B demonstrate that AutoSkill substantially improves mathematical reasoning: Pass@1 accuracy on MATH-500 and AMC benchmark are increased by 15% and 87% respectively, relative to standard baselines. By automating the discovery of effective reasoning granularities, AutoSkill offers a scalable, task-agnostic path for optimizing agentic behaviors. *The project is under active development.  In the following stage, we will investigate: i. generalizing the results and findings to different models. ii. quantifying the predictability of steering outcomes. iii. comparing the empirical effectiveness of automatically discovered skills with manually-curated ones*. Codes are open-sourced.

---

### 1. Extracting Activations on Reasoning Examples
> /activisions.ipynb

### 2. Discovering Skills from Activations and Semantic Interpretations
> /skills_interp.ipynb

### 3. Steering the Target Model towards an Identified Skill Direction
> /steering_via_model_edits.ipynb

- Model Steering via Editing: Adding the Steering Vector to MLP Bias Parameters at Each Layer as an Offset

---
### SFT Training Configurations
> /sft_configs/*

---
### Evaluation Pipeline
> /evaltask-v6.py

**Example usage:** python evaltask-v6.py --model meta-llama/Meta-Llama-3-8B-Instruct --tasks math500 aime1k gsm8k aime25 amc olympiad minerva --rollouts 256 --temperature 1.0 --gen_len 8000 --max_sample 500 --output_path l3b8-v-8k-t100 --full_logs True --sys_prompt rl_prompt --devices '0,1,2,3,5,6,7,8' --seed 42 --verbose True
