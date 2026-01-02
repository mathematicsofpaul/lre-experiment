"""
Extended Ray experiments with plot generation for eigenvalue spectra.

This script runs distributed LRE experiments and generates:
1. Layer-by-layer faithfulness plots (per experiment)
2. Eigenvalue spectrum plots (for best operators)
3. Comparison plots across models/datasets

Usage:
    python ray_plot_experiments.py --config qwen_experiments_config.json --plots-dir plots
"""

import ray
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime


@ray.remote(num_gpus=1)
class LREPlotExperimentWorker:
    """
    Ray actor that runs LRE experiments and generates plots.
    """
    
    def __init__(self, gpu_id: int):
        """Initialize worker with specific GPU."""
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) if gpu_id >= 0 else ""
        
        print(f"Worker initialized on GPU {gpu_id}")
        
    def run_experiment_with_plots(self, 
                                   model_name: str,
                                   data_file: str,
                                   layers_config: Dict,
                                   plots_dir: str = "plots",
                                   train_ratio: float = 0.6,
                                   seed: int = 42,
                                   default_template: str = "{} is commonly associated with"):
        """
        Run experiment and save plots.
        
        Returns:
            Dict with results and plot paths
        """
        from data_utils import (
            load_and_split_data,
            configure_template_and_print_summary,
            initialize_lre_model,
            get_layers_to_test,
            train_layer_with_loo,
            plot_operator_eigenvalue_spectrum
        )
        from transformers import AutoConfig
        
        print(f"\n{'='*80}")
        print(f"Starting: {model_name} on {data_file}")
        print(f"{'='*80}")
        
        try:
            # Create plots directory
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize model
            lre = initialize_lre_model(model_name, device=self.device)
            
            # Load data
            result = load_and_split_data(data_file, train_ratio=train_ratio, seed=seed)
            train_data = result['train_data']
            test_data = result['test_data']
            template = configure_template_and_print_summary(result, default_template=default_template)
            
            # Get layers to test
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
            else:
                num_layers = 12
                
            model_type = config.model_type if hasattr(config, 'model_type') else "unknown"
            
            # Determine layer naming pattern
            if model_type in ['gpt2', 'gpt_neo', 'gpt_neox']:
                layer_pattern = "transformer.h"
            elif model_type in ['llama', 'mistral', 'qwen2']:
                layer_pattern = "model.layers"
            else:
                layer_pattern = "model.layers"
            
            start = layers_config.get('start_offset', 3)
            end = num_layers - layers_config.get('end_offset', 3)
            step = layers_config.get('step', 1)
            
            layers_to_test = [f"{layer_pattern}.{i}" for i in range(start, end + 1, step)]
            
            # Run layer-by-layer experiment
            print(f"\nTesting layers: {layers_to_test}")
            
            results = {}
            faithfulness_scores = {}
            
            for layer in layers_to_test:
                print(f"\n--- Testing Layer: {layer} ---")
                operator, faithfulness = train_layer_with_loo(
                    lre, train_data, test_data, layer, template
                )
                results[layer] = operator
                faithfulness_scores[layer] = faithfulness
            
            # Find best layer
            best_layer = max(faithfulness_scores, key=faithfulness_scores.get)
            best_faithfulness = faithfulness_scores[best_layer]
            best_operator = results[best_layer]
            
            print(f"\nBest Layer: {best_layer}")
            print(f"Best Faithfulness: {best_faithfulness:.4f}")
            
            # Generate sanitized filename prefix
            model_safe = model_name.replace('/', '_').replace('.', '_')
            dataset_safe = Path(data_file).stem
            file_prefix = f"{model_safe}_{dataset_safe}"
            
            # Plot 1: Layer-by-layer faithfulness
            fig, ax = plt.subplots(figsize=(12, 6))
            layer_numbers = [int(l.split('.')[-1]) for l in layers_to_test]
            faithfulness_values = [faithfulness_scores[l] for l in layers_to_test]
            
            ax.plot(layer_numbers, faithfulness_values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Layer Number', fontsize=12)
            ax.set_ylabel('Faithfulness Score', fontsize=12)
            ax.set_title(f'Layer-wise Faithfulness: {model_name}\n{dataset_safe}', fontsize=13)
            ax.grid(alpha=0.3)
            ax.axhline(y=best_faithfulness, color='red', linestyle='--', alpha=0.5, 
                      label=f'Best: Layer {best_layer.split(".")[-1]}')
            ax.legend()
            
            layer_plot_path = f"{plots_dir}/{file_prefix}_layers.png"
            plt.savefig(layer_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved layer plot: {layer_plot_path}")
            
            # Plot 2: Eigenvalue spectrum for best operator
            eigenvalues = np.linalg.eigvals(best_operator.coef_)
            eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Subplot 1: All eigenvalues (log scale)
            ax1 = axes[0, 0]
            ax1.plot(eigenvalues_sorted, 'o-', color='steelblue', markersize=3, linewidth=1)
            ax1.set_yscale('log')
            ax1.set_xlabel('Index', fontsize=11)
            ax1.set_ylabel('Eigenvalue Magnitude (log)', fontsize=11)
            ax1.set_title('All Eigenvalues', fontsize=12, fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # Subplot 2: Top 50 eigenvalues
            ax2 = axes[0, 1]
            top_50 = eigenvalues_sorted[:min(50, len(eigenvalues_sorted))]
            ax2.bar(range(len(top_50)), top_50, color='coral', alpha=0.8)
            ax2.set_xlabel('Index', fontsize=11)
            ax2.set_ylabel('Magnitude', fontsize=11)
            ax2.set_title(f'Top {len(top_50)} Eigenvalues', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3, axis='y')
            
            # Subplot 3: Cumulative energy
            ax3 = axes[1, 0]
            total_energy = np.sum(eigenvalues_sorted**2)
            cumulative_energy = np.cumsum(eigenvalues_sorted**2) / total_energy
            ax3.plot(cumulative_energy, linewidth=2, color='green')
            ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
            ax3.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
            ax3.set_xlabel('Number of Eigenvalues', fontsize=11)
            ax3.set_ylabel('Cumulative Energy', fontsize=11)
            ax3.set_title('Energy Explained', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Subplot 4: Eigenvalue ratios
            ax4 = axes[1, 1]
            if len(eigenvalues_sorted) > 1:
                ratios = eigenvalues_sorted[:-1] / eigenvalues_sorted[1:]
                plot_range = min(100, len(ratios))
                ax4.plot(ratios[:plot_range], 'o-', color='purple', markersize=3, linewidth=1)
                ax4.set_xlabel('Index', fontsize=11)
                ax4.set_ylabel('λ_i / λ_{i+1}', fontsize=11)
                ax4.set_title('Consecutive Ratios', fontsize=12, fontweight='bold')
                ax4.grid(alpha=0.3)
            
            plt.suptitle(f'Eigenvalue Spectrum: {model_name}\n{dataset_safe} (Layer {best_layer})', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            eigen_plot_path = f"{plots_dir}/{file_prefix}_eigenvalues.png"
            plt.savefig(eigen_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved eigenvalue plot: {eigen_plot_path}")
            
            # Return results (can't pickle operators, save parameters)
            return {
                'status': 'success',
                'model_name': model_name,
                'data_file': data_file,
                'dataset_name': dataset_safe,
                'best_layer': best_layer,
                'best_faithfulness': best_faithfulness,
                'faithfulness_scores': faithfulness_scores,
                'layers_tested': layers_to_test,
                'best_operator_params': {
                    'coef_shape': best_operator.coef_.shape,
                    'intercept_shape': best_operator.intercept_.shape,
                    'coef_mean': float(np.mean(np.abs(best_operator.coef_))),
                    'coef_std': float(np.std(best_operator.coef_))
                },
                'eigenvalue_stats': {
                    'num_eigenvalues': len(eigenvalues_sorted),
                    'largest': float(eigenvalues_sorted[0]),
                    'smallest': float(eigenvalues_sorted[-1]),
                    'condition_number': float(eigenvalues_sorted[0] / eigenvalues_sorted[-1]),
                    'effective_rank_90': int(np.argmax(cumulative_energy >= 0.90) + 1),
                    'effective_rank_95': int(np.argmax(cumulative_energy >= 0.95) + 1)
                },
                'plots': {
                    'layer_faithfulness': layer_plot_path,
                    'eigenvalue_spectrum': eigen_plot_path
                }
            }
            
        except Exception as e:
            import traceback
            return {
                'status': 'failed',
                'model_name': model_name,
                'data_file': data_file,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


def run_distributed_plot_experiments(config_path: str,
                                     num_gpus: Optional[int] = None,
                                     plots_dir: str = "plots",
                                     results_dir: str = "results"):
    """
    Run distributed experiments with plot generation.
    
    Args:
        config_path: Path to JSON config file
        num_gpus: Number of GPUs to use (None = auto-detect)
        plots_dir: Directory for plots
        results_dir: Directory for results JSON
    """
    import torch
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    models = config['models']
    datasets = config['datasets']
    layers_config = config.get('layers_config', {})
    train_ratio = config.get('train_ratio', 0.6)
    seed = config.get('seed', 42)
    
    # Determine GPU count
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus)
    
    # Create output directories
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Distributed LRE Experiments with Plots")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Total experiments: {len(models) * len(datasets)}")
    print(f"GPUs: {num_gpus}")
    print(f"Plots directory: {plots_dir}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")
    
    # Create workers
    workers = [LREPlotExperimentWorker.remote(i) for i in range(num_gpus)] if num_gpus > 0 else \
              [LREPlotExperimentWorker.remote(-1)]  # CPU worker
    
    # Create experiment queue
    experiments = []
    for model in models:
        for dataset in datasets:
            experiments.append({
                'model_name': model,
                'data_file': dataset,
                'layers_config': layers_config,
                'plots_dir': plots_dir,
                'train_ratio': train_ratio,
                'seed': seed
            })
    
    # Submit experiments (round-robin across workers)
    print(f"Submitting {len(experiments)} experiments to {len(workers)} workers...\n")
    
    futures = []
    for i, exp in enumerate(experiments):
        worker_idx = i % len(workers)
        future = workers[worker_idx].run_experiment_with_plots.remote(**exp)
        futures.append(future)
    
    # Collect results
    results = []
    completed = 0
    while len(futures) > 0:
        ready, futures = ray.wait(futures, num_returns=1)
        result = ray.get(ready[0])
        results.append(result)
        completed += 1
        
        if result['status'] == 'success':
            print(f"✓ [{completed}/{len(experiments)}] {result['model_name']} on {result['dataset_name']}")
            print(f"    Best: {result['best_layer']} (faithfulness: {result['best_faithfulness']:.4f})")
        else:
            print(f"✗ [{completed}/{len(experiments)}] {result['model_name']} on {result['data_file']} FAILED")
            print(f"    Error: {result['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/experiments_with_plots_{timestamp}.json"
    
    output = {
        'config': config,
        'num_gpus_used': num_gpus,
        'total_experiments': len(experiments),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'plots_directory': plots_dir,
        'results': results,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Results: {results_file}")
    print(f"Plots: {plots_dir}/")
    print(f"{'='*80}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        # Sort by faithfulness
        successful.sort(key=lambda x: x['best_faithfulness'], reverse=True)
        
        print(f"\n{'Model':<30} {'Dataset':<25} {'Best Layer':<12} {'Faithfulness':<12} {'Rank'}")
        print("-" * 95)
        for r in successful:
            model_short = r['model_name'].split('/')[-1] if '/' in r['model_name'] else r['model_name']
            print(f"{model_short:<30} {r['dataset_name']:<25} {r['best_layer']:<12} "
                  f"{r['best_faithfulness']:<12.4f} Rank={r['eigenvalue_stats']['effective_rank_90']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed LRE experiments with plots")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory for plots")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    run_distributed_plot_experiments(
        config_path=args.config,
        num_gpus=args.num_gpus,
        plots_dir=args.plots_dir,
        results_dir=args.results_dir
    )
