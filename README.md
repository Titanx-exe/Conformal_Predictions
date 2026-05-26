<div align="center">

# Conformal Prediction for Reliable Candidate Generation in Entity Linking

Official repository for the paper: **"Conformal Prediction for Reliable Candidate Generation in Entity Linking"**.

![Conformal Predictions Logo](asset/logo.jpeg)

</div>

This repository contains the complete implementation of Conformal Prediction methods (Minimax, Softmax, and Margin) and Conformal Baselines (Top-K, Score-Threshold, and Platt Scaling) applied to autoregressive decoders (Llama-3) and dense embeddings (Qwen3, E5, and Biencoder) for candidate generation in Entity Linking (EL).

---

## Table of Contents
1. [Abstract Summary](#abstract-summary)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Models & Datasets](#models--datasets)
5. [Quick Start: Model Downloading](#quick-start-model-downloading)
6. [Running Evaluation and Searches](#running-evaluation-and-searches)
   - [Baseline Evaluator](#1-baseline-evaluator)
   - [Conformal Predictor Evaluator](#2-conformal-predictor-evaluator)
   - [Using Bash Automation Scripts](#3-using-bash-automation-scripts)
7. [Finetuning with LoRA & DHNM](#finetuning-with-lora--dhnm)
8. [Citing & References](#citing--references)

---

## Abstract Summary
Candidate generation is a crucial first step in Entity Linking. While conventional techniques select a fixed number of candidates (e.g., Top-K), they lack mathematical guarantees regarding the inclusion of the correct entity. In this work, we introduce Conformal Prediction (CP) to dynamically generate candidate sets of variable size that are guaranteed to contain the target entity with a user-defined confidence level ($1-\epsilon$). We evaluate these methods across 8 standard datasets, demonstrating that CP significantly improves candidate generation reliability while maintaining compact set sizes.

---

## Repository Structure
The project is organized as follows:
```text
Conformal_Predictions/
├── Finetuning/                # LoRA finetuning scripts & datasets
│   ├── Datasets/              # Sample dataset splits
│   ├── entity_update_blink.py # Map entities and fetch descriptions via SPARQL
│   └── finetuning_dhnm_blink.py # LoRA + Dynamic Hard Negative Mining (DHNM) training
├── models/                    # Model wrappers for E5, Llama, and Qwen decoders
│   ├── E5.py                  # E5 ranker model interface
│   ├── llama_decoder.py       # Llama-based Look-Both-Ways (LBW) decoder ranker
│   └── qwen3_decoder.py       # Qwen-based Look-Both-Ways (LBW) decoder ranker
├── scripts/                   # Automated bash execution scripts
│   ├── download_all_models.sh # Download base models (llama, qwen, e5)
│   ├── run_ablation_calibration.sh # Calibration split ablation study script
│   ├── run_ablation_epsilon.sh     # Epsilon ablation study script
│   ├── run_baseline_llama.sh  # Llama baseline runs on all 9 datasets
│   ├── run_baseline_qwen.sh   # Qwen baseline runs on all 9 datasets
│   ├── run_conformal_llama.sh # Llama conformal runs on all datasets and methods
│   └── run_conformal_qwen.sh  # Qwen conformal runs on all datasets and methods
├── download_model.py          # HuggingFace model downloader script
├── create_test_splits.py      # Split NIF datasets into test10/test20 calibration sets
├── generic_training.py        # Core training and single evaluation harness
├── parameters.py              # Configuration defaults and flag definitions
├── run_biencoder_conformal.py # Conformal Prediction runner for Biencoder / E5
├── run_lbw_hyperparameter_search_conformal.py # Conformal hyperparameter search
└── run_lbw_hyperparameter_search_conformal_baseline.py # Baseline hyperparameter search
```

---

## Environment Setup
It is highly recommended to use a Python virtual environment:
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Models & Datasets

### Fine-Tuned Checkpoints Restriction
> [!IMPORTANT]
> The final fine-tuned decoder models (LoRA adapters) and encoder checkpoints are currently withheld from this public release because they are utilized in ongoing concurrent research papers. The complete checkpoints will be published open-source on Hugging Face as soon as the associated papers are published. 
>
> To ensure reproducibility, reviewers and researchers can evaluate the entire pipeline using open-source base models (**Llama-3.1-1B**, **Qwen3-Embedding-0.6B**, and **E5-base-v2**), which are fully supported out-of-the-box.

### Dataset Retrieval
Datasets can be retrieved from the public repositories associated with the paper:
* *Contextual Augmentation for Entity Linking using Large Language Models*

For local execution, place your extracted datasets in the `data/aida/wikidata/` directory.

The 9 benchmark datasets utilized in our experiments are:
* `ace2004`, `aida`, `aquaint`, `iitb-fix`, `kore50`, `msnbc`, `n3reuters128`, `n3rss500`, `spotlight`

---

## Quick Start: Model Downloading
Use the provided `download_model.py` script to fetch base models from Hugging Face. We support shorthand aliases to automate output directory routing:

```bash
# Option 1: Download all models at once
bash scripts/download_all_models.sh

# Option 2: Download individually using shorthand aliases
python3 download_model.py --model_id llama
python3 download_model.py --model_id qwen
python3 download_model.py --model_id e5
```

---

## Running Evaluation and Searches

All python runner scripts support automatic architecture configuration detection based on `--model_id`. If `--found_model` or `--type_optimization` are not provided, the scripts will auto-deduce them based on whether `llama` or `qwen` is in your model ID path.

### 1. Baseline Evaluator
Evaluates the three baseline candidate predictors (`topk`, `score-threshold`, and `platt-scaling`) over a grid search of Look-Both-Ways layer configurations:
```bash
# Run Qwen3 baseline search on ace2004
python3 run_lbw_hyperparameter_search_conformal_baseline.py \
    --gpu_id 0 \
    --model_id ./models_local/qwen3-embedding-0.6B \
    --dataset ace2004
```

### 2. Conformal Predictor Evaluator
Evaluates the conformal prediction methods (`minmax`, `softmax`, or `margin`) given a confidence level ($1-\epsilon$):
```bash
# Run Llama base conformal search with minmax method on ace2004
python3 run_lbw_hyperparameter_search_conformal.py \
    --gpu_id 0 \
    --model_id ./models_local/llama-3.2-1B \
    --dataset ace2004 \
    --conformal_method minmax \
    --conformal_epsilon 0.05 \
    --conformal_coverage entity \
    --conformal_K 40
```

### 3. Using Bash Automation Scripts
We provide modular shell scripts under `scripts/` to run full evaluations across all 9 benchmark datasets sequentially:

```bash
# Run Llama baseline evaluations on GPU 0
bash scripts/run_baseline_llama.sh 0

# Run Qwen baseline evaluations on GPU 0
bash scripts/run_baseline_qwen.sh 0

# Run Llama conformal evaluations (minmax, softmax, margin) on GPU 0
bash scripts/run_conformal_llama.sh 0

# Run Qwen conformal evaluations (minmax, softmax, margin) on GPU 0
bash scripts/run_conformal_qwen.sh 0
```

### 4. Ablation Studies
We provide automated bash scripts to run the Epsilon and Calibration Split ablation studies:

#### Epsilon Ablation Study
Varies epsilon across `0.05`, `0.1`, and `0.15`:
```bash
# Run Epsilon Ablation for Qwen on GPU 0
bash scripts/run_ablation_epsilon.sh qwen 0

# Run Epsilon Ablation for Llama on GPU 0
bash scripts/run_ablation_epsilon.sh llama 0
```

#### Calibration Split Ablation Study
Varies the size of the calibration set (`test10` and `test20` - representing 10% and 20% of documents removed, respectively).

Before running the calibration ablation script, you must generate the `test10` and `test20` NIF splits using the provided `create_test_splits.py` script:
```bash
# Generate test10 & test20 splits (outputting to the respective splits folder)
python3 create_test_splits.py \
    data/aida/wikidata/ace2004_splits/ACE2004_testb \
    --out-dir data/aida/wikidata/ace2004_splits
```
*(Make sure to run this for all datasets you wish to evaluate before running the ablation script).*

Once splits are generated, run the ablation study:
```bash
# Run Calibration Set Ablation for Qwen on GPU 0
bash scripts/run_ablation_calibration.sh qwen 0

# Run Calibration Set Ablation for Llama on GPU 0
bash scripts/run_ablation_calibration.sh llama 0
```
*Note: To run tasks concurrently in the background, you can edit the scripts to uncomment the `setsid ... &` lines.*

---

## Finetuning with LoRA & DHNM
To perform LoRA finetuning on the BLINK dataset with Dynamic Hard Negative Mining (DHNM):
```bash
# Single GPU
python3 Finetuning/finetuning_dhnm_blink.py

# Multi-GPU training via Accelerate
CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --num_processes 3 Finetuning/finetuning_dhnm_blink.py
```
For further tuning and loss configuration details, please consult `parameters.py` and `Finetuning/finetuning_dhnm_blink.py`.

---

## Citing & References
If you use the datasets or code in this repository, please cite:

```bibtex
@inproceedings{vollmers-etal-2025-contextual,
  title = "Contextual Augmentation for Entity Linking using Large Language Models",
  author = "Vollmers, Daniel and Zahera, Hamada and Moussallem, Diego and Ngonga Ngomo, Axel-Cyrille",
  booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
  pages = "8535--8545",
  year = "2025",
  address = "Abu Dhabi, UAE",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.coling-main.570/"
}
```

### Authors
* **Daniel Vollmers**, **Hamada M. Zahera**, **Diego Moussallem**, **Axel-Cyrille Ngonga Ngomo**
* Data Science Group, Paderborn University, Germany
* Contact: `{daniel.vollmers, hamada.zahera, diego.moussallem, axel.ngonga}@uni-paderborn.de`
