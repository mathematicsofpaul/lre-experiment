# LRE Experiment - Linear Relational Embeddings

A Python implementation exploring **Linear Relational Embeddings (LRE)** using transformer language models to analyze how models encode relational knowledge. This experiment specifically examines gender bias stereotypes in academic fields and professions across multiple model architectures.

## Overview

This project implements the Linear Relational Embedding technique to understand how transformer models internally represent relationships between concepts. The experiment includes implementations for:

- **GPT-2** (124M parameters)
- **GPT-2-XL** (1.5B parameters)
- **Llama 3.2** (1B parameters)
- **Qwen 2.5** (0.6B parameters)
- **Gemma 2** (2.6B parameters)

Each implementation:

1. **Extracts hidden states** from specific transformer layers when processing subject prompts
2. **Trains a linear regression** to map subject representations to their associated outputs
3. **Evaluates faithfulness** - tests whether the linear approximation accurately predicts the model's behavior

### What is LRE?

Linear Relational Embeddings (LRE) is a technique for understanding how language models encode factual knowledge. The key insight is that many relationships can be approximated by a linear transformation:

```
z_predicted = W * h_subject + b
```

Where:
- `h_subject` is the hidden state at a specific layer for the subject entity
- `W` and `b` are learned linear parameters
- `z_predicted` approximates the model's output representation

This lets us test whether the model's reasoning about relationships is fundamentally linear.

## Dataset

The experiment uses `data/data_sample.json` containing 38 academic fields/professions paired with gendered associations (based on stereotypical model predictions):

```json
{
  "subject": "engineering",
  "object": "men"
}
```

**Note:** This dataset examines *biases in the model's outputs*, not factual or normative statements about fields. The data is split 60/40 for training and testing across all experiments.

## Requirements

- Python 3.12+
- `uv` (recommended package manager) or `pip`
- 4GB+ GPU memory recommended (or use CPU with `device="cpu"`)
- ~3GB disk space for GPT-2-XL model

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or navigate to the project directory
cd LRE-Experiment

# Install dependencies
uv sync
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install transformers torch scikit-learn accelerate ipykernel
pip install git+https://github.com/davidbau/baukit.git
```

## Usage

### Running the Jupyter Notebooks (Recommended)

The project includes multiple experiment notebooks for different models. Choose the model you want to explore:

```bash
# Using uv
uv run jupyter notebook

# Or with standard jupyter
jupyter notebook
```

Available experiment notebooks:

1. **`lre-experiment-gpt2-xl.ipynb`** - GPT-2-XL (1.5B parameters, 48 layers)
   - Tests multiple layers: early (layer 5), middle (layer 15), and late (layer 35)
   - Recommended starting point for most users

2. **`lre-experiment-gpt2.ipynb`** - GPT-2 (124M parameters, 12 layers)
   - Faster and lighter weight
   - Good for testing on limited hardware

3. **`lre-experiment-llama-3.2-1b.ipynb`** - Llama 3.2 (1B parameters)
   - Modern architecture comparison

4. **`lre-experiment-qwen3-0.6b.ipynb`** - Qwen 2.5 (0.6B parameters)
   - Efficient model for quick experiments

5. **`lre-experiment-gemma3-270m.ipynb`** - Gemma 2 (2.6B parameters)
   - Larger model for more capable representations

6. **`model-layer-inspect.ipynb`** - Utility notebook
   - Inspect layer names and architecture of any transformer model
   - Useful for configuring experiments with new models

Each notebook walks through:
1. Loading and splitting the dataset (60/40 train/test)
2. Initializing the model
3. Training LRE operators at different layers
4. Evaluating faithfulness on test data
5. Comparing layer performance

### Running Programmatically

You can also use the `LREModel` class directly in your own scripts:

```python
from lre import LREModel
import json
import random

# Load data
with open("data/data_sample.json", "r") as f:
    data = json.load(f)

# Split data (60/40 train/test)
random.seed(42)
random.shuffle(data)
split_idx = int(len(data) * 0.6)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Initialize model
lre = LREModel(model_name="gpt2-xl", device="mps")  # Use "cpu" or "cuda" as needed

# Train LRE operator
layer_name = "transformer.h.15"  # Middle layer of GPT-2-XL
template = "{} students are typically"
operator = lre.train_lre(train_data, layer_name, template)

# Evaluate
lre.evaluate(operator, test_data, layer_name, template)
```

### Device Configuration

Change the device parameter based on your hardware:

```python
# For Apple Silicon (M1/M2/M3)
lre = LREModel(model_name="gpt2-xl", device="mps")

# For NVIDIA GPUs
lre = LREModel(model_name="gpt2-xl", device="cuda")

# For CPU (slower but works everywhere)
lre = LREModel(model_name="gpt2-xl", device="cpu")
```

## Project Structure

```
LRE-Experiment/
├── lre/                                    # LRE package
│   ├── __init__.py                        # Package initialization
│   └── lre.py                             # Main LREModel class implementation
├── data/
│   └── data_sample.json                   # Dataset of field → gender associations
├── lre-experiment-gpt2.ipynb              # GPT-2 (124M) experiment notebook
├── lre-experiment-gpt2-xl.ipynb           # GPT-2-XL (1.5B) experiment notebook
├── lre-experiment-llama-3.2-1b.ipynb      # Llama 3.2 (1B) experiment notebook
├── lre-experiment-qwen3-0.6b.ipynb        # Qwen 2.5 (0.6B) experiment notebook
├── lre-experiment-gemma3-270m.ipynb       # Gemma 2 (2.6B) experiment notebook
├── model-layer-inspect.ipynb              # Tool for inspecting model architectures
├── pyproject.toml                         # Project dependencies and metadata
├── uv.lock                                # Locked dependency versions
└── README.md                              # This file
```

## How It Works

### 1. Hidden State Extraction

```python
h = self.get_hidden_state(prompt, layer_name, subject)
```

Extracts the hidden representation at layer 15 (for GPT-2-XL) at the last token position.

### 2. Linear Regression Training

```python
operator = lre.train_lre(training_data, layer_name, template)
```

Learns a linear mapping `W` and bias `b` such that:
- Input: hidden state of subject (e.g., "engineering")
- Output: embedding direction of expected object (e.g., "men")

### 3. Evaluation

```python
lre.evaluate(operator, test_data, layer_name, template)
```

Tests whether the learned linear transformation accurately predicts the model's actual outputs on unseen examples.

## Expected Output

```
--- Evaluation ---
Subject: nursing             | Expected: women      | LRE Pred: women       | ✓
Subject: engineering         | Expected: men        | LRE Pred: men         | ✓
Subject: computer science    | Expected: men        | LRE Pred: men         | ✓
...
Faithfulness: 28/38 (73.68%)
```

The faithfulness score indicates how well the linear approximation captures the model's actual behavior.

## Configuration Options

### Model Selection

```python
# Smaller, faster (fewer parameters)
lre = LREModel(model_name="gpt2", device="cpu")

# Larger, more capable (default)
lre = LREModel(model_name="gpt2-xl", device="mps")

# Even larger (requires more memory)
lre = LREModel(model_name="EleutherAI/gpt-j-6b", device="cuda")
```

### Layer Selection

Different layers capture different types of information:
- **Early layers (0-5)**: Syntax and basic patterns
- **Middle layers (10-20)**: Semantic relationships (recommended)
- **Late layers (25-47)**: Task-specific processing

Adjust the `LAYER_NAME` variable:
```python
LAYER_NAME = "transformer.h.15"  # For GPT-2-XL (48 layers total)
```

### Prompt Template

Modify the template to explore different relationships:
```python
TEMPLATE = "{} students are typically"  # Current: gender stereotypes
TEMPLATE = "The capital of {} is"       # Geography relations
TEMPLATE = "{} is located in"           # Location relations
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Use a smaller model: `gpt2` instead of `gpt2-xl`
2. Switch to CPU: `device="cpu"`
3. Reduce batch processing (modify the code to process one sample at a time)

### Slow Execution

- First run downloads the model (~3GB) - this is a one-time operation
- CPU inference is significantly slower than GPU
- Consider using MPS (Apple) or CUDA (NVIDIA) acceleration

### Module Not Found

Ensure all dependencies are installed:
```bash
uv sync  # Or pip install -r requirements.txt
```

## References

- [Baukit Library](https://github.com/davidbau/baukit) - Tools for probing neural networks
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- Original LRE research explores how linear transformations can represent factual knowledge in language models

## License

This is an experimental research project. Check individual dependency licenses for production use.

## Notes

- This experiment examines model biases, not factual accuracy
- Results depend heavily on model choice, layer selection, and prompt template
- The linear approximation provides insight into how models encode relationships

