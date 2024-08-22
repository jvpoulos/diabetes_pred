import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import json
import pyarrow.parquet as pq
import os
import sys
from pathlib import Path

def load_attention_weights_from_checkpoint(file_path: str) -> Dict[str, torch.Tensor]:
    """Load attention weights from a PyTorch Lightning checkpoint file."""
    try:
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        print(f"Loaded checkpoint. Keys: {checkpoint.keys()}")
        
        if 'state_dict' not in checkpoint:
            raise ValueError("The checkpoint does not contain a 'state_dict' key.")
        
        state_dict = checkpoint['state_dict']
        
        # Find attention weight keys
        attention_weight_keys = [k for k in state_dict.keys() if 'attention' in k and 'weight' in k]
        
        if not attention_weight_keys:
            raise ValueError("No attention weights found in the checkpoint.")
        
        print(f"Found attention weight keys: {attention_weight_keys}")
        
        # Extract attention weights
        attention_weights = {}
        for key in attention_weight_keys:
            attention_weights[key] = state_dict[key]
            print(f"Shape of {key}: {state_dict[key].shape}")
        
        print(f"Total number of attention weight tensors: {len(attention_weights)}")
        return attention_weights
        
    except Exception as e:
        print(f"Error loading the checkpoint: {str(e)}")
        sys.exit(1)

def process_attention_weights(attention_weights: Dict[str, torch.Tensor], labels: torch.Tensor, dynamic_indices: np.ndarray, dynamic_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Process attention weights and return weights for each attention head."""
    print("Processing attention weights...")
    
    layer_numbers = set(int(key.split('.')[3]) for key in attention_weights.keys() if 'attn.attention' in key)
    num_layers = len(layer_numbers)
    
    if num_layers == 0:
        print("Warning: No layers found in attention weights.")
        return np.array([]), labels
    
    hidden_size = next(iter(attention_weights.values())).shape[0]
    num_heads = hidden_size // 64  # Assuming head_dim is 64
    head_dim = 64
    
    print(f"Number of layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    
    # Initialize attention scores for each head
    head_scores = torch.zeros(num_layers, num_heads, head_dim, head_dim)
    
    # Compute attention scores for each layer and head
    for layer in layer_numbers:
        q_key = f'model.encoder.h.{layer}.attn.attention.q_proj.weight'
        k_key = f'model.encoder.h.{layer}.attn.attention.k_proj.weight'
        
        if q_key not in attention_weights or k_key not in attention_weights:
            print(f"Warning: Skipping layer {layer} due to missing keys")
            continue
        
        q = attention_weights[q_key].view(num_heads, head_dim, -1)
        k = attention_weights[k_key].view(num_heads, head_dim, -1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        normalized_scores = F.softmax(scores, dim=-1)
        head_scores[layer] = normalized_scores
    
    if torch.isnan(head_scores).any() or torch.isinf(head_scores).any():
        print("Warning: NaN or Inf values detected in attention scores.")
        return np.array([]), labels
    
    # Convert to numpy array
    head_scores_np = head_scores.numpy()
    
    print(f"Head scores shape: {head_scores_np.shape}")
    print(f"Head scores min: {head_scores_np.min()}, max: {head_scores_np.max()}, mean: {head_scores_np.mean()}")
    
    # Convert labels to numpy array if it's not already
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    return head_scores_np, labels

# Make sure to update the create_feature_names function to handle dynamic_values
def create_feature_names(static_measurement_indices, static_indices, dynamic_indices, dynamic_values):
    feature_names = []
    # Add static measurements
    for measurement in static_measurement_indices:
        feature_names.append(f"{measurement}_Static")
    
    # Add static indices
    for measurement, values in static_indices.items():
        for value in values:
            feature_names.append(f"{measurement}_{value}")
    
    # Add dynamic indices
    for feature in dynamic_indices:
        feature_names.append(f"{feature}_Dynamic")
    
    # Note: We're not adding dynamic values here as they might be too numerous
    # If you want to include them, you can add a similar loop as for static indices
    
    return feature_names

# Update the create_heatmap function to use a subset of feature names if necessary
def create_heatmap(head_scores: np.ndarray, labels: np.ndarray, feature_names: List[str], output_dir: str):
    """Create and save heatmaps of attention weights for each layer and head."""
    num_layers, num_heads, head_dim, _ = head_scores.shape
    
    # Use a subset of feature names if there are too many
    max_features = min(len(feature_names), head_dim)
    feature_names_subset = feature_names[:max_features]
    
    for layer in range(num_layers):
        for head in range(num_heads):
            fig, ax = plt.subplots(figsize=(20, 20))
            
            sns.heatmap(head_scores[layer, head, :max_features, :max_features], 
                        ax=ax, cmap="YlOrRd", 
                        xticklabels=feature_names_subset, 
                        yticklabels=feature_names_subset)
            ax.set_title(f"Attention Weights - Layer {layer}, Head {head}")
            ax.set_xlabel("Features")
            ax.set_ylabel("Features")
            
            plt.tight_layout()
            plt.savefig(output_dir / f"attention_weights_layer{layer}_head{head}.png", dpi=300, bbox_inches='tight')
            plt.close()

# Update the create_top_features_table function to use feature names
def create_top_features_table(head_scores: np.ndarray, feature_names: List[str], n_top: int = 20) -> pd.DataFrame:
    """Create a table of top n features with highest average attention weights for each layer and head."""
    num_layers, num_heads, head_dim, _ = head_scores.shape
    results = []
    
    # Use a subset of feature names if there are too many
    max_features = min(len(feature_names), head_dim)
    feature_names_subset = feature_names[:max_features]
    
    for layer in range(num_layers):
        for head in range(num_heads):
            head_weights = head_scores[layer, head, :max_features, :max_features]
            top_indices = head_weights.mean(axis=0).argsort()[-n_top:][::-1]
            
            for idx in top_indices:
                results.append({
                    "Layer": layer,
                    "Head": head,
                    "Feature": feature_names_subset[idx],
                    "Average Attention Weight": head_weights[:, idx].mean()
                })
    
    return pd.DataFrame(results)

def load_vocabularies(directory):
    """Load vocabulary files."""
    vocab_files = {
        'static_measurement_indices': directory / 'static_measurement_indices_vocab.json',
        'static_indices': directory / 'static_indices_vocab.json',
        'dynamic_indices': directory / 'dynamic_indices_vocab.json'
    }
    
    vocabularies = {}
    for key, filename in vocab_files.items():
        try:
            with open(filename, 'r') as f:
                vocabularies[key] = json.load(f)
        except FileNotFoundError:
            print(f"Error: The vocabulary file '{filename}' was not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The vocabulary file '{filename}' is not a valid JSON file.")
            sys.exit(1)
    
    return vocabularies['static_measurement_indices'], vocabularies['static_indices'], vocabularies['dynamic_indices']

def get_feature_name_and_value(index: int, static_measurement_indices, static_indices, dynamic_indices):
    """Get feature name and value from vocabularies."""
    for measurement, measurement_index in static_measurement_indices.items():
        if index == measurement_index:
            return measurement, "Static"
    
    for measurement, values in static_indices.items():
        for value, value_index in values.items():
            if index == value_index:
                return measurement, value
    
    for feature, feature_index in dynamic_indices.items():
        if index == feature_index:
            return feature, "Dynamic"
    
    return f"Unknown_{index}", "Unknown"

def get_feature_description(feature: str, value: str) -> str:
    """Get feature description. This is a placeholder function."""
    # In a real scenario, you would implement this function to provide meaningful descriptions
    return f"Description for {feature} with value {value}"

def print_nested_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent+1)
        else:
            print('  ' * (indent+1) + f"Shape: {value.shape}")

def main():
    parser = argparse.ArgumentParser(description="Process attention weights from a PyTorch Lightning checkpoint file.")
    parser.add_argument("checkpoint_path", help="Path to the PyTorch Lightning checkpoint file")
    parser.add_argument("use_labs", help="Whether to use labs data")
    args = parser.parse_args()

    if args.use_labs:
        # Create a data_summaries folder if it doesn't exist
        os.makedirs("data_summaries/labs", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries/labs")
        DATA_DIR = Path("data/labs")
    else:
        # Create a data_summaries folder if it doesn't exist
        os.makedirs("data_summaries", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries")
        DATA_DIR = Path("data/labs")

    # Check if the file exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: The checkpoint file '{args.checkpoint_path}' does not exist.")
        sys.exit(1)

    try:
        # Load attention weights from checkpoint
        attention_weights = load_attention_weights_from_checkpoint(args.checkpoint_path)
        
        # Print attention_weights structure for debugging
        print("Attention weights structure:")
        print_nested_dict(attention_weights)
        
        # Load vocabularies
        static_measurement_indices, static_indices, dynamic_indices = load_vocabularies(DATA_DIR)
        
        # Load labels
        labels = load_labels(DATA_DIR)
        
        # Load dynamic data
        dynamic_indices_data, dynamic_values = load_dynamic_data(directory=DATA_DIR)
        
        # Process attention weights
        head_scores, processed_labels = process_attention_weights(attention_weights, labels, dynamic_indices_data, dynamic_values)
        
        if len(head_scores) == 0:
            print("Error: Empty attention weights. Unable to create visualizations.")
            sys.exit(1)
        
        # Create feature names list
        feature_names = create_feature_names(static_measurement_indices, static_indices, dynamic_indices, dynamic_values)
        
        # Create and save heatmaps
        create_heatmap(head_scores, processed_labels, feature_names, DATA_SUMMARIES_DIR)
        print("Heatmaps saved in", DATA_SUMMARIES_DIR)
        
        # Create top features table
        top_features_df = create_top_features_table(head_scores, feature_names)
        
        # Save top features table to CSV
        top_features_df.to_csv(DATA_SUMMARIES_DIR / "top_features_attention_weights.csv", index=False)
        print("Top features table saved as 'top_features_attention_weights.csv' in", DATA_SUMMARIES_DIR)
        
        # Print top features table
        print("\nTop Features Table (first 20 rows):")
        print(top_features_df.head(20).to_string(index=False))

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def load_dynamic_data(directory, max_seq_length=1000):  # You can adjust this value as needed
    """Load dynamic indices and values from the parquet file."""
    try:
        # Load the parquet file
        df = pq.read_table(directory / "DL_reps/train_0.parquet").to_pandas()
        
        # Sort by subject_id to ensure consistency with labels
        df = df.sort_values('subject_id')
        
        # Function to truncate or pad numpy arrays
        def truncate_or_pad_array(arr, max_len, pad_value):
            if len(arr) > max_len:
                return arr[:max_len]
            return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_value)
        
        # Truncate or pad and convert to numpy arrays
        dynamic_indices = np.array([truncate_or_pad_array(seq, max_seq_length, -1) for seq in df['dynamic_indices']])
        dynamic_values = np.array([truncate_or_pad_array(seq, max_seq_length, np.nan) for seq in df['dynamic_values']])
        
        # Replace NaN values with a special token (e.g., -1)
        dynamic_values = np.where(np.isnan(dynamic_values), -1, dynamic_values)
        
        print(f"Loaded dynamic data. Shape of dynamic indices: {dynamic_indices.shape}")
        print(f"Shape of dynamic values: {dynamic_values.shape}")
        
        return dynamic_indices, dynamic_values
    
    except Exception as e:
        print(f"Error loading dynamic data: {str(e)}")
        raise

def load_labels(directory):
    """Load labels from parquet files."""
    label_files = [
        directory / "task_dfs/a1c_greater_than_7_train.parquet",
        directory / "task_dfs/a1c_greater_than_7_val.parquet",
        directory / "task_dfs/a1c_greater_than_7_test.parquet"
    ]
    
    all_df = pd.DataFrame()
    for file in label_files:
        try:
            df = pq.read_table(file).to_pandas()
            all_df = pd.concat([all_df, df], ignore_index=True)
        except FileNotFoundError:
            print(f"Error: The label file '{file}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading the label file '{file}': {str(e)}")
            sys.exit(1)
    
    # Sort by subject_id to ensure consistency with dynamic data
    all_df = all_df.sort_values('subject_id')
    
    # Convert boolean labels to integer (0 for False, 1 for True)
    labels = all_df['label'].astype(int).values
    
    print(f"Loaded labels. Shape: {labels.shape}")
    
    return labels

if __name__ == "__main__":
    main()