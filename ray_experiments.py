"""
Ray-based distributed LRE experiments across multiple models, datasets, and GPUs.

Usage:
    python ray_experiments.py --config experiments_config.json
    
Or programmatically:
    from ray_experiments import run_distributed_experiments
    results = run_distributed_experiments(config)
"""

import ray
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

# Initialize Ray with GPU support
# ray.init(num_gpus=torch.cuda.device_count()) will be called in main


@ray.remote(num_gpus=1)  # Each task gets 1 GPU
class LREExperimentWorker:
    """
    Ray actor that runs LRE experiments on a dedicated GPU.
    Each worker loads a model and runs experiments independently.
    """
    
    def __init__(self, gpu_id: int):
        """Initialize worker with specific GPU."""
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        
        # Import here to avoid issues with Ray serialization
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) if gpu_id >= 0 else ""
        
        print(f"Worker initialized on GPU {gpu_id}")
        
    def run_experiment(self, 
                       model_name: str,
                       data_file: str,
                       layers_config: Dict,
                       train_ratio: float = 0.6,
                       seed: int = 42,
                       default_template: str = "{} is commonly associated with"):
        """
        Run a complete LRE experiment for one model on one dataset.
        
        Args:
            model_name: HuggingFace model name
            data_file: Path to dataset JSON file
            layers_config: Dict with 'start_offset', 'end_offset', 'step'
            train_ratio: Proportion of data for training
            seed: Random seed
            default_template: Default prompt template
            
        Returns:
            Dict with experiment results
        """
        from data_utils import (
            load_and_split_data,
            configure_template_and_print_summary,
            initialize_lre_model,
            get_layers_to_test,
            run_layer_experiment
        )
        
        print(f"\n{'='*80}")
        print(f"Starting experiment: {model_name} on {data_file}")
        print(f"Device: {self.device}")
        print(f"{'='*80}")
        
        try:
            # Initialize model
            lre = initialize_lre_model(model_name, device=self.device)
            
            # Load and split data
            result = load_and_split_data(data_file, train_ratio=train_ratio, seed=seed)
            train_data = result['train_data']
            test_data = result['test_data']
            template = configure_template_and_print_summary(result, default_template=default_template)
            
            # Get model metadata for layer detection
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
            else:
                num_layers = 12  # fallback
                
            model_type = config.model_type if hasattr(config, 'model_type') else "unknown"
            
            model_info = {
                model_name: {
                    'num_layers': num_layers,
                    'model_type': model_type
                }
            }
            
            # Determine layers to test
            layers_to_test = get_layers_to_test(
                model_name,
                model_info=model_info,
                start_offset=layers_config.get('start_offset', 3),
                end_offset=layers_config.get('end_offset', 3),
                step=layers_config.get('step', 1)
            )
            
            # Run layer experiment
            experiment_results = run_layer_experiment(
                lre_model=lre,
                train_data=train_data,
                test_data=test_data,
                layers_to_test=layers_to_test,
                template=template,
                visualize=False  # Don't visualize in distributed mode
            )
            
            # Prepare results for serialization (Ray can't serialize LinearRegression objects)
            serializable_results = {
                'model_name': model_name,
                'data_file': data_file,
                'dataset_name': result['dataset_name'],
                'template': template,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'layers_tested': layers_to_test,
                'faithfulness_scores': experiment_results['faithfulness_scores'],
                'best_layer': experiment_results['best_layer'],
                'best_faithfulness': experiment_results['best_faithfulness'],
                'device': self.device,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n{'='*80}")
            print(f"Completed: {model_name} on {data_file}")
            print(f"Best layer: {experiment_results['best_layer']}")
            print(f"Best faithfulness: {experiment_results['best_faithfulness']:.4f}")
            print(f"{'='*80}\n")
            
            return {
                'status': 'success',
                'results': serializable_results
            }
            
        except Exception as e:
            print(f"ERROR in experiment {model_name} on {data_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'failed',
                'error': str(e),
                'model_name': model_name,
                'data_file': data_file,
                'device': self.device
            }


def run_distributed_experiments(config_path: str, 
                                 num_gpus: Optional[int] = None,
                                 results_dir: str = "results"):
    """
    Run distributed LRE experiments across multiple GPUs.
    
    Args:
        config_path: Path to JSON configuration file
        num_gpus: Number of GPUs to use (None = auto-detect)
        results_dir: Directory to save results
        
    Config file format:
    {
        "models": ["gpt2", "Qwen/Qwen3-0.6B", ...],
        "datasets": ["data/bias/characteristic_gender.json", ...],
        "layers_config": {
            "start_offset": 3,
            "end_offset": 3,
            "step": 1
        },
        "train_ratio": 0.6,
        "seed": 42,
        "default_template": "{} is commonly associated with"
    }
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    models = config['models']
    datasets = config['datasets']
    layers_config = config.get('layers_config', {'start_offset': 3, 'end_offset': 3, 'step': 1})
    train_ratio = config.get('train_ratio', 0.6)
    seed = config.get('seed', 42)
    default_template = config.get('default_template', "{} is commonly associated with")
    
    # Detect available GPUs
    import torch
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus is None:
        num_gpus = available_gpus if available_gpus > 0 else 1
    
    print(f"\n{'='*80}")
    print(f"Distributed LRE Experiments")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Total experiments: {len(models) * len(datasets)}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Using GPUs: {num_gpus}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")
    
    # Initialize Ray
    if not ray.is_initialized():
        if num_gpus > 0:
            ray.init(num_gpus=num_gpus)
        else:
            ray.init()
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create worker pool (one per GPU)
    workers = [LREExperimentWorker.remote(i) for i in range(num_gpus)]
    
    # Create experiment queue
    experiments = []
    for model_name in models:
        for data_file in datasets:
            experiments.append({
                'model_name': model_name,
                'data_file': data_file,
                'layers_config': layers_config,
                'train_ratio': train_ratio,
                'seed': seed,
                'default_template': default_template
            })
    
    print(f"Submitting {len(experiments)} experiments to {num_gpus} workers...\n")
    
    # Submit experiments with round-robin GPU assignment
    futures = []
    for i, exp in enumerate(experiments):
        worker = workers[i % num_gpus]
        future = worker.run_experiment.remote(**exp)
        futures.append(future)
    
    # Collect results as they complete
    all_results = []
    completed = 0
    
    while futures:
        # Wait for at least one to complete
        ready, futures = ray.wait(futures, num_returns=1)
        
        for future in ready:
            result = ray.get(future)
            all_results.append(result)
            completed += 1
            
            if result['status'] == 'success':
                print(f"✓ [{completed}/{len(experiments)}] Completed: {result['results']['model_name']} on {result['results']['dataset_name']}")
            else:
                print(f"✗ [{completed}/{len(experiments)}] Failed: {result['model_name']} on {result['data_file']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"experiments_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'num_gpus_used': num_gpus,
            'total_experiments': len(experiments),
            'successful': sum(1 for r in all_results if r['status'] == 'success'),
            'failed': sum(1 for r in all_results if r['status'] == 'failed'),
            'results': all_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
    
    # Generate summary
    successful_results = [r['results'] for r in all_results if r['status'] == 'success']
    
    if successful_results:
        print("\nSummary of Best Results:")
        print(f"{'Model':<30} {'Dataset':<35} {'Best Layer':<15} {'Faithfulness':<12}")
        print("-" * 95)
        
        for result in sorted(successful_results, key=lambda x: x['best_faithfulness'], reverse=True):
            model = result['model_name'].split('/')[-1][:28]
            dataset = result['dataset_name'][:33]
            layer = result['best_layer'].split('.')[-1]
            faith = result['best_faithfulness']
            print(f"{model:<30} {dataset:<35} Layer {layer:<8} {faith:.4f}")
    
    return all_results


def create_example_config(output_path: str = "experiments_config.json"):
    """Create an example configuration file."""
    
    config = {
        "models": [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "Qwen/Qwen3-0.6B",
            "google/gemma-2-2b"
        ],
        "datasets": [
            "data/bias/characteristic_gender.json",
            "data/bias/occupation_gender.json",
            "data/factual/country_capital_city.json",
            "data/factual/person_occupation.json"
        ],
        "layers_config": {
            "start_offset": 3,
            "end_offset": 3,
            "step": 1
        },
        "train_ratio": 0.6,
        "seed": 42,
        "default_template": "{} is commonly associated with"
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Example configuration saved to: {output_path}")
    print("\nEdit this file to customize your experiments, then run:")
    print(f"  python ray_experiments.py --config {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed LRE experiments with Ray")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--create-config', action='store_true', help='Create example configuration file')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_example_config()
    elif args.config:
        run_distributed_experiments(
            config_path=args.config,
            num_gpus=args.num_gpus,
            results_dir=args.results_dir
        )
    else:
        print("Usage:")
        print("  python ray_experiments.py --create-config  # Create example config")
        print("  python ray_experiments.py --config experiments_config.json  # Run experiments")
        print("\nFor more options: python ray_experiments.py --help")
