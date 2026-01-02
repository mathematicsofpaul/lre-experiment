"""
Utility functions for loading and managing data files.
"""
import os
import json
import ipywidgets as widgets


def load_model_options(config_file="model_config.json"):
    """
    Load model options from JSON configuration file.
    
    Args:
        config_file: Path to the model configuration JSON file
        
    Returns:
        list: List of model names
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get('model_options', [])
    except FileNotFoundError:
        print(f"Warning: {config_file} not found. Using default models.")
        return ["Qwen/Qwen3-0.6B"]


def load_json_files(data_root="data"):
    """
    Load all JSON files from subdirectories under data_root.
    
    Args:
        data_root: Root directory to search for JSON files
        
    Returns:
        list: Sorted list of relative paths to JSON files
    """
    json_files = []
    
    if os.path.exists(data_root):
        for subdir, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.json'):
                    # Store relative path from data root
                    rel_path = os.path.relpath(os.path.join(subdir, file), data_root)
                    json_files.append(rel_path)
    
    json_files.sort()  # Sort for consistent ordering
    print(f"Found {len(json_files)} relation files:")
    
    return json_files


def create_data_file_dropdown(json_files):
    """
    Create a dropdown widget for selecting data files.
    
    Args:
        json_files: List of JSON file paths to display in dropdown
        
    Returns:
        ipywidgets.Dropdown: Dropdown widget for file selection
    """
    return widgets.Dropdown(
        options=json_files,
        value=json_files[0] if json_files else None,
        description='Data File:',
        style={'description_width': 'initial'}
    )


def create_model_dropdown(default_model="Qwen/Qwen3-0.6B"):
    """
    Create a dropdown widget for selecting models from config file.
    
    Args:
        default_model: Default model to select
        
    Returns:
        ipywidgets.Dropdown: Dropdown widget for model selection
    """
    model_options = load_model_options()
    
    # Validate default model is in options
    if default_model not in model_options and model_options:
        default_model = model_options[0]
    
    return widgets.Dropdown(
        options=model_options,
        value=default_model,
        description='Model:',
        style={'description_width': 'initial'}
    )


def fetch_model_layer_info(models, output_file=None, verbose=True):
    """
    Fetch layer count and metadata for a list of models.
    
    Args:
        models: List of model names to fetch info for
        output_file: Optional JSON file path to save results
        verbose: Whether to print progress information
        
    Returns:
        dict: Dictionary mapping model names to their metadata
    """
    from transformers import AutoConfig
    
    model_info = {}
    
    for model_name in models:
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Different models use different attribute names for layer count
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
            elif hasattr(config, 'num_layers'):
                num_layers = config.num_layers
            else:
                num_layers = None
                if verbose:
                    print(f"  WARNING: Could not determine layer count for {model_name}")
            
            model_info[model_name] = {
                "num_layers": num_layers,
                "model_type": config.model_type if hasattr(config, 'model_type') else "unknown"
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: Failed to load {model_name}: {str(e)}")
            model_info[model_name] = {
                "num_layers": None,
                "error": str(e)
            }
    
    # Display summary
    if verbose:
        for model_name, info in model_info.items():
            layers = info.get('num_layers', 'Unknown')
            model_type = info.get('model_type', 'unknown')
            print(f"{model_name:<40} | Layers: {layers:<4} | Type: {model_type}")
    
    # Save to JSON file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        if verbose:
            print(f"\nModel information saved to: {output_file}")
    
    return model_info


def load_and_split_data(data_file, train_ratio=0.6, n_train=None, seed=42):
    """
    Load JSON data and split into train/test sets.
    
    Args:
        data_file: Path to the JSON data file
        train_ratio: Proportion of data to use for training (default: 0.6 means 60% train, 40% test)
        n_train: Exact number of training samples to use. If specified, overrides train_ratio (default: None)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        dict: Dictionary containing:
            - 'train_data': Training samples
            - 'test_data': Test samples
            - 'template': Prompt template (if available)
            - 'dataset_name': Dataset name (if available)
    """
    import random
    
    with open(data_file, "r") as f:
        data_json = json.load(f)
    
    # Handle new format with "samples" key and metadata
    if isinstance(data_json, dict) and "samples" in data_json:
        data = data_json["samples"]
        template = data_json["prompt_templates"][0] if "prompt_templates" in data_json and data_json["prompt_templates"] else None
        dataset_name = data_json.get('name', 'Unknown')
    else:
        # Legacy format: assume data_json is directly the samples list
        data = data_json
        template = None
        dataset_name = 'Unknown'
    
    # Split data
    random.seed(seed)
    random.shuffle(data)
    
    # Use exact number of training samples if specified, otherwise use train ratio
    if n_train is not None:
        split_idx = min(n_train, len(data))  # Don't exceed available data
    else:
        split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'template': template,
        'dataset_name': dataset_name
    }


def configure_template_and_print_summary(result, default_template=None):
    """
    Configure template from data result and print dataset summary.
    
    Args:
        result: Dictionary returned by load_and_split_data()
        default_template: Default template to use if none in data file (default: None)
        
    Returns:
        str: The selected template (from file or default)
    """
    # Use template from data file if available, otherwise use default
    template = result['template'] if result['template'] else default_template
    
    # Print summary
    print(f"Dataset: {result['dataset_name']}")
    print(f"Template: {template}")
    print(f"Data: {len(result['train_data'])} train, {len(result['test_data'])} test")
    
    return template


def initialize_lre_model(model_name, device="mps"):
    """
    Initialize an LRE model with the given model name and device.
    Automatically retrieves HF_TOKEN from environment variables if available.
    
    Args:
        model_name (str): HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
        device (str): Device to run the model on (default: "mps" for Apple Silicon)
    
    Returns:
        LREModel: Initialized LRE model instance
    """
    from lre import LREModel
    import os
    
    # Get HuggingFace token from environment (if available)
    hf_token = os.getenv('HF_TOKEN', None)
    
    print(f"Selected model: {model_name}")
    if hf_token:
        print("Using HuggingFace token from environment")
    else:
        print("No HF_TOKEN found - proceeding without authentication")
    
    # Initialize and return LRE model
    lre = LREModel(
        model_name=model_name,
        device=device,
        token=hf_token
    )
    
    return lre


def get_layers_to_test(model_name, model_info=None, start_offset=3, end_offset=3, step=1):
    """
    Automatically determine which layers to test based on model metadata.
    Uses model architecture info to determine layer naming pattern and range.
    
    Args:
        model_name (str): HuggingFace model name
        model_info (dict): Optional model metadata from fetch_model_layer_info()
        start_offset (int): How many layers to skip from the start (default: 3)
        end_offset (int): How many layers to skip from the end (default: 3)
        step (int): Step size for layer selection (default: 1 = test all layers)
    
    Returns:
        list: Layer names to test (e.g., ["model.layers.3", "model.layers.4", ...])
    """
    from transformers import AutoConfig
    
    # If model_info not provided, fetch it
    if model_info is None or model_name not in model_info:
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Get layer count
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
            elif hasattr(config, 'num_layers'):
                num_layers = config.num_layers
            else:
                raise ValueError(f"Could not determine layer count for {model_name}")
            
            model_type = config.model_type if hasattr(config, 'model_type') else "unknown"
        except Exception as e:
            raise ValueError(f"Failed to load model config for {model_name}: {str(e)}")
    else:
        # Use provided model_info
        info = model_info[model_name]
        num_layers = info.get('num_layers')
        model_type = info.get('model_type', 'unknown')
        
        if num_layers is None:
            raise ValueError(f"No layer count available for {model_name}")
    
    # Determine layer naming pattern based on model type
    # GPT-2, GPT-Neo, GPT-J use "transformer.h"
    # Most others (BERT, LLaMA, Mistral, Qwen, Gemma) use "model.layers"
    if model_type in ['gpt2', 'gpt_neo', 'gptj', 'gpt_neox']:
        layer_prefix = "transformer.h"
    elif model_type in ['bert', 'roberta', 'distilbert']:
        layer_prefix = "encoder.layer"
    else:
        # Default for modern models (LLaMA, Mistral, Qwen, Gemma, etc.)
        layer_prefix = "model.layers"
    
    # Calculate layer range
    start_layer = max(0, start_offset)
    end_layer = max(start_layer + 1, num_layers - end_offset)
    
    # Generate layer names
    layers_to_test = [f"{layer_prefix}.{i}" for i in range(start_layer, end_layer, step)]
    
    print(f"Model: {model_name}")
    print(f"  Type: {model_type}")
    print(f"  Total layers: {num_layers}")
    print(f"  Layer prefix: {layer_prefix}")
    print(f"  Testing layers {start_layer} to {end_layer-1} (step={step})")
    print(f"  Total layers to test: {len(layers_to_test)}")
    
    return layers_to_test


def train_layer_with_loo(lre_model, train_data, test_data, layer_name, template):
    """
    Train a single layer using Leave-One-Out Cross Validation and evaluate on test set.
    
    Args:
        lre_model: Initialized LRE model instance
        train_data: Training samples (list of dicts with 'subject' and 'object')
        test_data: Test samples for evaluation
        layer_name: Layer name to train (e.g., "model.layers.12")
        template: Prompt template string (e.g., "{} is commonly associated with")
    
    Returns:
        dict: {
            'averaged_operator': Averaged LinearRegression operator,
            'faithfulness': Faithfulness score on test set,
            'eval_results': Full evaluation results
        }
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    layer_num = layer_name.split(".")[-1]
    print(f"\n{'='*80}")
    print(f"TESTING LAYER {layer_num}")
    print(f"{'='*80}")
    
    # Train with Leave-One-Out Cross Validation
    loo_temp = LeaveOneOut()
    layer_operators_list = []
    layer_bias_list = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(loo_temp.split(train_data)):
        fold_train = [train_data[i] for i in train_idx]
        
        # Create few-shot template for this fold
        few_shot_examples = "\n".join([
            template.format(sample['subject']) + f" {sample['object']}."
            for sample in fold_train
        ])
        few_shot_template = few_shot_examples + "\n" + template
        
        # Train operator on this fold
        operator_fold = lre_model.train_lre(fold_train, layer_name, few_shot_template)
        
        # Store operator weights and bias
        layer_operators_list.append(operator_fold.coef_)
        layer_bias_list.append(operator_fold.intercept_)
    
    # Average the operators for this layer
    avg_coef_layer = np.mean(layer_operators_list, axis=0)
    avg_bias_layer = np.mean(layer_bias_list, axis=0)
    
    # Create averaged operator
    averaged_operator_layer = LinearRegression()
    averaged_operator_layer.coef_ = avg_coef_layer
    averaged_operator_layer.intercept_ = avg_bias_layer
    
    # Create few-shot template with all training examples for evaluation
    few_shot_examples_full = "\n".join([
        template.format(sample['subject']) + f" {sample['object']}."
        for sample in train_data
    ])
    few_shot_template_full = few_shot_examples_full + "\n" + template
    
    # Evaluate on test_data
    print(f"\nEvaluating Layer {layer_num} on test set:")
    eval_results = lre_model.evaluate(averaged_operator_layer, test_data, layer_name, few_shot_template_full)
    
    return {
        'averaged_operator': averaged_operator_layer,
        'faithfulness': eval_results.get('faithfulness', 0),
        'eval_results': eval_results
    }


def run_layer_experiment(lre_model, train_data, test_data, layers_to_test, template, visualize=True):
    """
    Run layer-by-layer experiment to find the best layer for LRE.
    
    Args:
        lre_model: Initialized LRE model instance
        train_data: Training samples
        test_data: Test samples
        layers_to_test: List of layer names to test
        template: Prompt template string
        visualize: Whether to create visualization (default: True)
    
    Returns:
        dict: {
            'results': Dict mapping layer names to averaged operators,
            'faithfulness_scores': Dict mapping layer names to faithfulness scores,
            'best_layer': Name of best performing layer,
            'best_faithfulness': Faithfulness score of best layer,
            'layer_numbers': List of layer numbers (for visualization),
            'faithfulness_values': List of faithfulness values (for visualization)
        }
    """
    results = {}
    faithfulness_scores = {}
    
    # Train and evaluate each layer
    for layer_name in layers_to_test:
        layer_results = train_layer_with_loo(lre_model, train_data, test_data, layer_name, template)
        results[layer_name] = layer_results['averaged_operator']
        faithfulness_scores[layer_name] = layer_results['faithfulness']
    
    # Find the best layer based on test performance
    best_layer = max(faithfulness_scores, key=faithfulness_scores.get)
    best_faithfulness = faithfulness_scores[best_layer]
    
    print(f"\n{'='*80}")
    print(f"BEST LAYER: {best_layer} with test faithfulness score: {best_faithfulness:.4f}")
    print(f"{'='*80}")
    
    # Prepare data for visualization
    layer_numbers = [int(layer.split(".")[-1]) for layer in layers_to_test]
    faithfulness_values = [faithfulness_scores[layer] for layer in layers_to_test]
    
    # Create visualization if requested
    if visualize:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(layer_numbers, faithfulness_values, color='steelblue')
        ax.set_xlabel('Faithfulness Score (Test Set)')
        ax.set_ylabel('Layer Number')
        ax.set_title('LRE Faithfulness by Layer (Test Set Performance)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\n{'='*80}")
    print("LAYER COMPARISON SUMMARY (TEST SET)")
    print(f"{'='*80}")
    for layer in layers_to_test:
        layer_num = layer.split(".")[-1]
        print(f"Layer {layer_num}: Test Faithfulness = {faithfulness_scores[layer]:.4f}")
    
    return {
        'results': results,
        'faithfulness_scores': faithfulness_scores,
        'best_layer': best_layer,
        'best_faithfulness': best_faithfulness,
        'layer_numbers': layer_numbers,
        'faithfulness_values': faithfulness_values
    }


def plot_operator_eigenvalue_spectrum(operator, title="Operator Eigenvalue Spectrum"):
    """
    Plot the eigenvalue spectrum of an operator's coefficient matrix.
    
    Parameters:
    -----------
    operator : LinearRegression
        The trained linear regression operator
    title : str
        Title for the plot
        
    Returns:
    --------
    eigenvalues_sorted : numpy array
        Sorted eigenvalues in descending order by magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(operator.coef_)
    eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All eigenvalues (log scale)
    ax1 = axes[0, 0]
    ax1.plot(eigenvalues_sorted, 'o-', color='steelblue', markersize=3, linewidth=1)
    ax1.set_yscale('log')
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue Magnitude (log scale)', fontsize=11)
    ax1.set_title('All Eigenvalues', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top 50 eigenvalues
    ax2 = axes[0, 1]
    top_50 = eigenvalues_sorted[:min(50, len(eigenvalues_sorted))]
    ax2.bar(range(len(top_50)), top_50, color='coral', alpha=0.8)
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=11)
    ax2.set_title(f'Top {len(top_50)} Eigenvalues', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Cumulative energy
    ax3 = axes[1, 0]
    total_energy = np.sum(eigenvalues_sorted**2)
    cumulative_energy = np.cumsum(eigenvalues_sorted**2) / total_energy
    ax3.plot(cumulative_energy, linewidth=2, color='green')
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% energy')
    ax3.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% energy')
    ax3.set_xlabel('Number of Eigenvalues', fontsize=11)
    ax3.set_ylabel('Cumulative Energy Ratio', fontsize=11)
    ax3.set_title('Cumulative Energy Explained', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Ratio of consecutive eigenvalues
    ax4 = axes[1, 1]
    if len(eigenvalues_sorted) > 1:
        eigenvalue_ratios = eigenvalues_sorted[:-1] / eigenvalues_sorted[1:]
        plot_range = min(100, len(eigenvalue_ratios))
        ax4.plot(eigenvalue_ratios[:plot_range], 'o-', color='purple', markersize=3, linewidth=1)
        ax4.set_xlabel('Index', fontsize=11)
        ax4.set_ylabel('λ_i / λ_{i+1}', fontsize=11)
        ax4.set_title(f'Ratio of Consecutive Eigenvalues (First {plot_range})', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("EIGENVALUE SPECTRUM STATISTICS")
    print(f"{'='*60}")
    print(f"Total number of eigenvalues: {len(eigenvalues_sorted)}")
    print(f"Largest eigenvalue: {eigenvalues_sorted[0]:.6e}")
    print(f"Smallest eigenvalue: {eigenvalues_sorted[-1]:.6e}")
    print(f"Condition number: {eigenvalues_sorted[0] / eigenvalues_sorted[-1]:.6e}")
    
    print(f"\nTop 10 eigenvalues:")
    for i in range(min(10, len(eigenvalues_sorted))):
        print(f"  λ_{i+1}: {eigenvalues_sorted[i]:.6e}")
    
    # Effective rank
    effective_rank_90 = np.argmax(cumulative_energy >= 0.90) + 1
    effective_rank_95 = np.argmax(cumulative_energy >= 0.95) + 1
    print(f"\nEffective rank (90% energy): {effective_rank_90}")
    print(f"Effective rank (95% energy): {effective_rank_95}")
    
    return eigenvalues_sorted
