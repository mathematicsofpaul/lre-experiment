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
        ax.bar(layer_numbers, faithfulness_values, color='steelblue')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Faithfulness Score (Test Set)')
        ax.set_title('LRE Faithfulness by Layer (Test Set Performance)')
        ax.grid(alpha=0.3, axis='y')
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


def plot_operator_eigenvalue_spectrum(operator, title="Operator Eigenvalue Spectrum", xlim=None, ylim=None):
    """
    Plot the eigenvalue spectrum of an operator's coefficient matrix.
    
    Parameters:
    -----------
    operator : LinearRegression
        The trained linear regression operator
    title : str
        Title for the plot
    xlim : int or tuple, optional
        If int: sets x-axis limit for both plots (0 to xlim)
        If tuple: sets x-axis range as (xmin, xmax)
        If None: auto-scales based on data
    ylim : tuple, optional
        Sets y-axis limits as (ymin, ymax) for both plots
        If None: auto-scales based on data
        
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All eigenvalues (log scale)
    ax1 = axes[0]
    ax1.plot(eigenvalues_sorted, 'o-', color='steelblue', markersize=3, linewidth=1)
    ax1.set_yscale('log')
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue Magnitude (log scale)', fontsize=11)
    ax1.set_title('All Eigenvalues', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Apply x-axis limit to first plot if specified
    if xlim is not None:
        if isinstance(xlim, int):
            ax1.set_xlim(0, xlim)
        else:
            ax1.set_xlim(xlim)
    
    # Apply y-axis limit to first plot if specified
    if ylim is not None:
        ax1.set_ylim(ylim)
    
    # Plot 2: Top 50 eigenvalues
    ax2 = axes[1]
    top_50 = eigenvalues_sorted[:min(50, len(eigenvalues_sorted))]
    ax2.bar(range(len(top_50)), top_50, color='coral', alpha=0.8)
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=11)
    ax2.set_title(f'Top {len(top_50)} Eigenvalues', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Apply x-axis limit to second plot if specified
    if xlim is not None:
        if isinstance(xlim, int):
            ax2.set_xlim(-0.5, min(xlim, len(top_50)) - 0.5)
        else:
            ax2.set_xlim(xlim)
    
    # Apply y-axis limit to second plot if specified
    if ylim is not None:
        ax2.set_ylim(ylim)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
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
    
    return fig, eigenvalues_sorted


def plot_pca_predictions(test_eval_results, class1_label='men', class2_label='women', title_prefix='PCA', figsize=(16, 14)):
    """
    Perform PCA on LRE predictions and visualize the results.
    
    Args:
        test_eval_results: Dictionary containing evaluation results with 'eval_results' key
        class1_label: Label for the first class (default: 'men')
        class2_label: Label for the second class (default: 'women')
        title_prefix: Prefix for plot titles (default: 'PCA')
        figsize: Figure size tuple (default: (16, 14))
    
    Returns:
        Dictionary containing:
            - pca: Fitted PCA object
            - predictions_pca: All predictions transformed to PC space
            - class1_pca: Class 1 predictions in PC space
            - class2_pca: Class 2 predictions in PC space
            - class1_predictions: Raw class 1 predictions
            - class2_predictions: Raw class 2 predictions
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract z_pred (LRE output predictions) from test_eval_results
    class1_predictions = []
    class2_predictions = []
    
    for result in test_eval_results['eval_results']:
        z_pred = result['z_pred']
        expected_object = result['expected']
        
        if expected_object == class1_label:
            class1_predictions.append(z_pred)
        elif expected_object == class2_label:
            class2_predictions.append(z_pred)
    
    # Combine all predictions
    all_predictions = class1_predictions + class2_predictions
    labels = [class1_label] * len(class1_predictions) + [class2_label] * len(class2_predictions)
    
    # Perform PCA
    pca = PCA(n_components=3)
    predictions_pca = pca.fit_transform(all_predictions)
    
    # Split back into class1 and class2
    class1_pca = predictions_pca[:len(class1_predictions)]
    class2_pca = predictions_pca[len(class1_predictions):]
    
    # Create multiple plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: PC1 vs PC2
    ax1 = axes[0, 0]
    ax1.scatter(class1_pca[:, 0], class1_pca[:, 1], c='steelblue', label=class1_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax1.scatter(class2_pca[:, 0], class2_pca[:, 1], c='coral', label=class2_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
    ax1.set_title(f'{title_prefix}: PC1 vs PC2', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: PC1 vs PC3
    ax2 = axes[0, 1]
    ax2.scatter(class1_pca[:, 0], class1_pca[:, 2], c='steelblue', label=class1_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax2.scatter(class2_pca[:, 0], class2_pca[:, 2], c='coral', label=class2_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)', fontsize=11)
    ax2.set_title(f'{title_prefix}: PC1 vs PC3', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Plot 3: PC2 vs PC3
    ax3 = axes[1, 0]
    ax3.scatter(class1_pca[:, 1], class1_pca[:, 2], c='steelblue', label=class1_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax3.scatter(class2_pca[:, 1], class2_pca[:, 2], c='coral', label=class2_label.capitalize(), alpha=0.7, s=100, edgecolors='black')
    ax3.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
    ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)', fontsize=11)
    ax3.set_title(f'{title_prefix}: PC2 vs PC3', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Explained Variance
    ax4 = axes[1, 1]
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax4.bar(range(1, 4), pca.explained_variance_ratio_[:3], alpha=0.7, color='steelblue', label='Individual')
    ax4.plot(range(1, 4), cumulative_variance[:3], 'o-', color='coral', linewidth=2, markersize=8, label='Cumulative')
    ax4.set_xlabel('Principal Component', fontsize=11)
    ax4.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax4.set_title('Explained Variance by Component', fontsize=12, fontweight='bold')
    ax4.set_xticks([1, 2, 3])
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nPCA Analysis:")
    print(f"Number of {class1_label} samples: {len(class1_predictions)}")
    print(f"Number of {class2_label} samples: {len(class2_predictions)}")
    print(f"PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"PC2 explained variance: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"PC3 explained variance: {pca.explained_variance_ratio_[2]*100:.2f}%")
    print(f"Total explained variance (3 PCs): {sum(pca.explained_variance_ratio_[:3])*100:.2f}%")
    
    return {
        'pca': pca,
        'predictions_pca': predictions_pca,
        'class1_pca': class1_pca,
        'class2_pca': class2_pca,
        'class1_predictions': class1_predictions,
        'class2_predictions': class2_predictions
    }

def plot_operator_svd_analysis(operator, title="SVD Analysis", n_values=50):
    """
    Perform Singular Value Decomposition (SVD) on an operator and visualize the singular values.
    
    Parameters:
    -----------
    operator : LinearRegression or similar
        The operator with a coef_ attribute containing the weight matrix
    title : str, optional
        Title for the plot (default: "SVD Analysis")
    n_values : int, optional
        Number of top singular values to display (default: 50)
    
    Returns:
    --------
    singular_values : ndarray
        Array of singular values from the SVD
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Perform SVD
    _, singular_values, _ = np.linalg.svd(operator.coef_, full_matrices=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 5))
    
    n_to_plot = min(n_values, len(singular_values))
    ax.bar(range(n_to_plot), singular_values[:n_to_plot], alpha=0.8, color='coral')
    ax.set_xlabel('Index', fontsize=10)
    ax.set_ylabel('Singular Value', fontsize=10)
    ax.set_title(f'{title}\nTop {n_to_plot} Singular Values', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nSVD Analysis:")
    print(f"Total singular values: {len(singular_values)}")
    print(f"Largest singular value: {singular_values[0]:.6e}")
    print(f"Smallest singular value: {singular_values[-1]:.6e}")
    print(f"Condition number: {singular_values[0] / singular_values[-1]:.6e}")
    
    return singular_values

def plot_pc1_projection_lines(test_eval_results, pca_results, class1_label='class1', class2_label='class2', title='PC1 Projection'):
    """
    Create a 2-line visualization showing PC1 projections for two classes.
    
    This function creates a horizontal line plot where:
    - class1 samples are plotted on y=1
    - class2 samples are plotted on y=-1
    - Correct predictions are shown as circles (o)
    - Incorrect predictions are shown as X markers
    
    Parameters:
    -----------
    test_eval_results : dict
        Dictionary containing evaluation results with 'eval_results' key
    pca_results : dict
        Dictionary returned from plot_pca_predictions() containing:
        - 'pca': PCA object
        - 'predictions_pca': PCA-transformed predictions
        - 'class1_pca': PCA-transformed class1 predictions
        - 'class2_pca': PCA-transformed class2 predictions
    class1_label : str, optional
        Label for first class (default: 'class1')
    class2_label : str, optional
        Label for second class (default: 'class2')
    title : str, optional
        Plot title (default: 'PC1 Projection')
    
    Returns:
    --------
    dict with statistics about the projections
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract from pca_results
    pca = pca_results['pca']
    predictions_pca = pca_results['predictions_pca']
    class1_pca = pca_results['class1_pca']
    class2_pca = pca_results['class2_pca']
    
    # Extract PC1 values
    class1_pc1 = class1_pca[:, 0]
    class2_pc1 = class2_pca[:, 0]
    
    # Separate correct and incorrect predictions
    class1_pc1_correct = []
    class1_pc1_incorrect = []
    class2_pc1_correct = []
    class2_pc1_incorrect = []
    
    # Store indices for labeling
    class1_correct_indices = []
    class1_incorrect_indices = []
    class2_correct_indices = []
    class2_incorrect_indices = []
    
    for i, result in enumerate(test_eval_results['eval_results']):
        is_correct = result['status'] == '✓ Correct'
        pc1_value = predictions_pca[i, 0]
        
        if result['expected'] == class1_label:
            if is_correct:
                class1_pc1_correct.append(pc1_value)
                class1_correct_indices.append(i)
            else:
                class1_pc1_incorrect.append(pc1_value)
                class1_incorrect_indices.append(i)
        else:  # class2
            if is_correct:
                class2_pc1_correct.append(pc1_value)
                class2_correct_indices.append(i)
            else:
                class2_pc1_incorrect.append(pc1_value)
                class2_incorrect_indices.append(i)
    
    # Create 2-line plot
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot class1 predictions on line y=1
    ax.scatter(class1_pc1_correct, [1] * len(class1_pc1_correct), 
              c='steelblue', s=200, alpha=0.8, marker='o', 
              edgecolors='black', linewidth=2, label=f'{class1_label.capitalize()} (Correct)', zorder=3)
    ax.scatter(class1_pc1_incorrect, [1] * len(class1_pc1_incorrect), 
              c='steelblue', s=200, alpha=0.8, marker='x', 
              linewidth=3, label=f'{class1_label.capitalize()} (Incorrect)', zorder=3)
    
    # Plot class2 predictions on line y=-1
    ax.scatter(class2_pc1_correct, [-1] * len(class2_pc1_correct), 
              c='coral', s=200, alpha=0.8, marker='o', 
              edgecolors='black', linewidth=2, label=f'{class2_label.capitalize()} (Correct)', zorder=3)
    ax.scatter(class2_pc1_incorrect, [-1] * len(class2_pc1_incorrect), 
              c='coral', s=200, alpha=0.8, marker='x', 
              linewidth=3, label=f'{class2_label.capitalize()} (Incorrect)', zorder=3)
    
    # Draw horizontal lines
    ax.axhline(y=1, color='steelblue', linewidth=1, alpha=0.3, zorder=1)
    ax.axhline(y=-1, color='coral', linewidth=1, alpha=0.3, zorder=1)
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', alpha=0.5, zorder=2, label='PC1 = 0')
    
    # Configure plot
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=13, fontweight='bold')
    ax.set_yticks([1, -1])
    ax.set_yticklabels([class1_label.capitalize(), class2_label.capitalize()], fontsize=12, fontweight='bold')
    ax.set_title(f'{title}: {class1_label.capitalize()} vs {class2_label.capitalize()} Predictions', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    ax.grid(axis='x', alpha=0.3)
    ax.set_ylim(-1.8, 1.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics with subject names
    print(f"\nPC1 Statistics:")
    print(f"\n{class1_label.capitalize()} predictions (correct):")
    for idx in class1_correct_indices:
        subject = test_eval_results['eval_results'][idx]['subject']
        print(f"  [{idx}] {subject}: {predictions_pca[idx, 0]:.4f}")
    
    print(f"\n{class1_label.capitalize()} predictions (incorrect):")
    for idx in class1_incorrect_indices:
        subject = test_eval_results['eval_results'][idx]['subject']
        print(f"  [{idx}] {subject}: {predictions_pca[idx, 0]:.4f}")
    
    print(f"\n{class2_label.capitalize()} predictions (correct):")
    for idx in class2_correct_indices:
        subject = test_eval_results['eval_results'][idx]['subject']
        print(f"  [{idx}] {subject}: {predictions_pca[idx, 0]:.4f}")
    
    print(f"\n{class2_label.capitalize()} predictions (incorrect):")
    for idx in class2_incorrect_indices:
        subject = test_eval_results['eval_results'][idx]['subject']
        print(f"  [{idx}] {subject}: {predictions_pca[idx, 0]:.4f}")
    
    print(f"\nMean PC1 ({class1_label.capitalize()}): {np.mean(class1_pc1):.4f}")
    print(f"Mean PC1 ({class2_label.capitalize()}): {np.mean(class2_pc1):.4f}")
    print(f"Separation: {abs(np.mean(class1_pc1) - np.mean(class2_pc1)):.4f}")
    
    return {
        'class1_correct_indices': class1_correct_indices,
        'class1_incorrect_indices': class1_incorrect_indices,
        'class2_correct_indices': class2_correct_indices,
        'class2_incorrect_indices': class2_incorrect_indices,
        'class1_mean_pc1': np.mean(class1_pc1),
        'class2_mean_pc1': np.mean(class2_pc1),
        'separation': abs(np.mean(class1_pc1) - np.mean(class2_pc1))
    }
