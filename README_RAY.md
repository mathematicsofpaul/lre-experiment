# Distributed LRE Experiments with Ray

Run LRE experiments across multiple models, datasets, and GPUs in parallel using Ray.

## Installation

```bash
pip install ray torch transformers
```

## Quick Start

### 1. Create Configuration File

```bash
python ray_experiments.py --create-config
```

This creates `experiments_config.json` with example settings.

### 2. Edit Configuration

```json
{
  "models": [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B"
  ],
  "datasets": [
    "data/bias/characteristic_gender.json",
    "data/factual/country_capital_city.json"
  ],
  "layers_config": {
    "start_offset": 3,
    "end_offset": 3,
    "step": 1
  },
  "train_ratio": 0.6,
  "seed": 42
}
```

### 3. Run Experiments

```bash
# Auto-detect and use all GPUs
python ray_experiments.py --config experiments_config.json

# Use specific number of GPUs
python ray_experiments.py --config experiments_config.json --num-gpus 2

# CPU only (no GPUs)
python ray_experiments.py --config experiments_config.json --num-gpus 0
```

## How It Works

1. **GPU Assignment**: Each worker gets 1 GPU and runs experiments independently
2. **Round-Robin Distribution**: Experiments are distributed across GPUs in round-robin fashion
3. **Parallel Execution**: Multiple model/dataset combinations run simultaneously
4. **Automatic Scheduling**: Ray handles scheduling, fault tolerance, and resource management

## Architecture

```
┌─────────────────────────────────────┐
│   Ray Coordinator (CPU)             │
│   - Loads config                    │
│   - Creates experiment queue        │
│   - Collects results                │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼─────┐  ┌─────▼──────┐
│  Worker 0  │  │  Worker 1  │  ...
│  (GPU 0)   │  │  (GPU 1)   │
│            │  │            │
│ - Load     │  │ - Load     │
│   Model A  │  │   Model B  │
│ - Run      │  │ - Run      │
│   Dataset1 │  │   Dataset2 │
└────────────┘  └────────────┘
```

## Multi-Machine Setup

Ray can scale across multiple machines:

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='<head-node-ip>:6379'

# Then run experiments (they'll use all GPUs across all machines)
python ray_experiments.py --config experiments_config.json
```

## Example Output

```
================================================================================
Distributed LRE Experiments
================================================================================
Models: 3
Datasets: 4
Total experiments: 12
Available GPUs: 2
Using GPUs: 2
Results directory: results
================================================================================

Submitting 12 experiments to 2 workers...

✓ [1/12] Completed: gpt2 on characteristic gender
✓ [2/12] Completed: gpt2-medium on country capital city
✓ [3/12] Completed: Qwen/Qwen3-0.6B on characteristic gender
...

All experiments completed!
Results saved to: results/experiments_20260102_143022.json
```

## Programmatic Usage

```python
from ray_experiments import run_distributed_experiments

results = run_distributed_experiments(
    config_path="experiments_config.json",
    num_gpus=2,
    results_dir="my_results"
)

# Process results
successful = [r for r in results if r['status'] == 'success']
for result in successful:
    print(f"{result['results']['model_name']}: {result['results']['best_faithfulness']:.4f}")
```

## Results Format

Results are saved as JSON:

```json
{
  "config": { ... },
  "num_gpus_used": 2,
  "total_experiments": 12,
  "successful": 11,
  "failed": 1,
  "results": [
    {
      "status": "success",
      "results": {
        "model_name": "gpt2",
        "dataset_name": "characteristic gender",
        "best_layer": "transformer.h.8",
        "best_faithfulness": 0.8333,
        "faithfulness_scores": { ... },
        "layers_tested": [ ... ]
      }
    }
  ]
}
```

## Tips

1. **Memory Management**: Each GPU loads one model at a time. If you have 8GB GPUs, avoid loading models > 7GB.

2. **Optimal GPU Usage**: Set experiments = num_gpus × N for best utilization (e.g., 4 GPUs → 8, 12, 16... experiments)

3. **Monitoring**: Use `ray status` in another terminal to monitor cluster status

4. **Debugging**: Check individual experiment logs in the Ray dashboard at `http://localhost:8265`

5. **Fault Tolerance**: Ray automatically retries failed experiments (configurable)

## Advanced: Custom Experiments

Create custom experiment functions:

```python
@ray.remote(num_gpus=1)
class CustomExperimentWorker:
    def run_custom_experiment(self, model_name, **kwargs):
        # Your custom experiment logic
        pass
```

## Performance Comparison

| Setup | Time for 12 Experiments |
|-------|------------------------|
| Sequential (1 GPU) | ~120 minutes |
| Ray (2 GPUs) | ~60 minutes |
| Ray (4 GPUs) | ~30 minutes |
| Ray (8 GPUs) | ~15 minutes |

Linear scaling with number of GPUs!
