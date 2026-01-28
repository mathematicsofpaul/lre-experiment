# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LRE Experiment is a Python research project exploring Linear Relational Embeddings (LRE) to understand how transformer language models encode relational knowledge. The technique learns linear transformations (W, b) that map subject hidden states to predicted outputs: `z_predicted = W * h_subject + b`.

## Commands

```bash
# Install dependencies (uv recommended)
uv sync

# Run Jupyter notebooks for interactive experiments
uv run jupyter notebook

# Run distributed Ray experiments
python ray_experiments.py --config experiments_config.json

# Create experiment config template
python ray_experiments.py --create-config
```

## Architecture

### Core Components

- **`lre/lre.py`**: `LREModel` class - loads transformer models, extracts hidden states via baukit's TraceDict, trains linear regression operators, and evaluates faithfulness
- **`data_utils.py`**: Dataset loading, model initialization, layer testing utilities, and plotting functions
- **`ray_experiments.py`**: Distributed experiment execution across multiple GPUs using Ray workers

### Data Organization

Datasets in `data/` are JSON files organized by category:
- `bias/`: Gender, age, religion stereotypes
- `commonsense/`: Object properties, sentiment
- `factual/`: Knowledge relations (capitals, families, etc.)
- `linguistic/`: Antonyms, past tense, letter patterns

Each dataset contains `prompt_templates` (with `{}` placeholder), `samples` (subject/object pairs), and metadata.

### Configuration Files

- `model_config.json`: Available HuggingFace model identifiers
- `model_layer_info.json`: Layer counts for supported models
- `.env`: Contains `HF_TOKEN` for HuggingFace access

### Supported Models

Google Gemma 3, Mistral, Meta Llama 3.2, and Qwen 3 variants (0.6B to 8B parameters).

## Key Patterns

- Device configuration: `"cpu"`, `"cuda"`, or `"mps"` (Apple Silicon)
- Data splits: 60/40 train/test with reproducible seeding (seed=42)
- Layer naming: Varies by model (e.g., `transformer.h.15` for GPT-2, `model.layers.8` for Llama)
- Results: Faithfulness scores as percentage of correct predictions
