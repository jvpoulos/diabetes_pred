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
import yaml

def load_icd10_descriptions(file_path: str) -> Dict[str, str]:
    """Load ICD-10 descriptions from the provided file."""
    try:
        df = pd.read_csv(file_path, sep=',', header=0, 
                         names=['GUID', 'ICD10code', 'Description', 'Type', 'Source', 'Language', 'LastUpdateDTS', 'Dt'], 
                         encoding='ISO-8859-1', on_bad_lines='skip')
        df['ICD10code'] = df['ICD10code'].astype(str)
        descriptions = dict(zip(df['ICD10code'], df['Description']))
        print(f"Loaded {len(descriptions)} ICD-10 descriptions")
        print("Sample ICD-10 codes:", list(descriptions.keys())[:5])
        return descriptions
    except Exception as e:
        print(f"Error loading ICD-10 descriptions: {str(e)}")
        return {}

def load_cpt_descriptions(file_path: str, cpt_txt_file_path: str) -> Dict[str, str]:
    """Load CPT descriptions from the provided files."""
    try:
        # Load from the main CPT file
        df1 = pd.read_csv(file_path, sep="\t", header=0, usecols=['Code', 'Description'])
        df1['Code'] = df1['Code'].astype(str)
        descriptions = dict(zip(df1['Code'], df1['Description']))
        
        # Load from the CPT.txt file
        df2 = pd.read_csv(cpt_txt_file_path, sep='\t', header=None, 
                          usecols=[0, 3], names=['CPTcode', 'long_description'], 
                          encoding='latin1')
        df2['CPTcode'] = df2['CPTcode'].astype(str)
        descriptions.update(dict(zip(df2['CPTcode'], df2['long_description'])))
        
        print(f"Loaded {len(descriptions)} CPT descriptions")
        print("Sample CPT codes:", list(descriptions.keys())[:5])
        return descriptions
    except Exception as e:
        print(f"Error loading CPT descriptions: {str(e)}")
        return {}

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

def process_attention_weights(attention_weights: Dict[str, torch.Tensor], head_dim: int, input_data: torch.Tensor, static_indices: Dict, continuous_features: List[str]) -> Dict[str, np.ndarray]:
    print("Processing attention weights...")

    layer_numbers = sorted(set(int(key.split('.')[3]) for key in attention_weights.keys() if 'attn.attention' in key))
    num_layers = len(layer_numbers)
    hidden_size = next(iter(attention_weights.values())).shape[0]
    num_heads = hidden_size // head_dim

    print(f"Number of layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Input data shape: {input_data.shape}")

    # Initialize attention weights for each layer and head
    attention_weights_dict = {
        'dynamic_indices': [[] for _ in range(num_layers)],
        'dynamic_values': [[] for _ in range(num_layers)],
        'static_indices': [[] for _ in range(num_layers)],
        'SDI_score_normalized': [[] for _ in range(num_layers)],
        'AgeYears_normalized': [[] for _ in range(num_layers)],
        'InitialA1c_normalized': [[] for _ in range(num_layers)]
    }

    # Create a projection layer to map input_data to the correct dimension
    projection = torch.nn.Linear(input_data.shape[1], hidden_size)
    input_data_projected = projection(input_data)

    # Compute attention weights for each layer and head
    for layer in layer_numbers:
        for head in range(num_heads):
            q_key = f'model.encoder.h.{layer}.attn.attention.q_proj.weight'
            k_key = f'model.encoder.h.{layer}.attn.attention.k_proj.weight'

            if q_key not in attention_weights or k_key not in attention_weights:
                print(f"Warning: Skipping layer {layer}, head {head} due to missing keys")
                continue

            q = attention_weights[q_key][head*head_dim:(head+1)*head_dim, :]
            k = attention_weights[k_key][head*head_dim:(head+1)*head_dim, :]

            # Compute Q and K for this head
            Q = torch.matmul(input_data_projected, q.t())
            K = torch.matmul(input_data_projected, k.t())

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)

            # Add a small epsilon to prevent underflow
            epsilon = 1e-10
            scores = scores + epsilon

            # Apply softmax to get attention weights
            attn_weights = F.softmax(scores, dim=-1)

            # Split attention weights for different feature types
            feature_types = ['dynamic_indices', 'dynamic_values', 'static_indices'] + continuous_features
            feature_sizes = {
                'dynamic_indices': input_data.shape[1] // 2,
                'dynamic_values': input_data.shape[1] // 2,
                'static_indices': len(static_indices),
                'SDI_score_normalized': 1,
                'AgeYears_normalized': 1,
                'InitialA1c_normalized': 1
            }

            start_idx = 0
            for feature_type in feature_types:
                end_idx = start_idx + feature_sizes[feature_type]
                print(f"Processing {feature_type}: {start_idx} to {end_idx}")
                attention_weights_dict[feature_type][layer].append(attn_weights[:, start_idx:end_idx].detach().cpu().numpy())
                start_idx = end_idx

    return attention_weights_dict

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

    # Add continuous variables
    continuous_vars = ["InitialA1c_normalized", "AgeYears_normalized", "SDI_score_normalized"]
    feature_names.extend(continuous_vars)

    return feature_names
    
def create_heatmap(head_scores: np.ndarray, labels: np.ndarray, feature_names: List[str], output_dir: str):
    """Create and save heatmaps of attention weights for dynamic_indices."""
    num_layers, num_heads, num_features, head_dim = head_scores.shape

    # Clean feature names
    clean_feature_names = [name.replace('_Static', '').replace('_Dynamic', '') for name in feature_names]

    # Get features with attention weights of 0.01 or more
    avg_weights = head_scores.mean(axis=(0, 1, 3))  # Average across layers, heads, and head dimensions
    significant_indices = np.where(avg_weights >= 0.01)[0]
    
    if len(significant_indices) == 0:
        print("No features with attention weights >= 0.01 found. Skipping heatmap creation.")
        return

    significant_features = [clean_feature_names[i] for i in significant_indices]
    significant_weights = avg_weights[significant_indices]

    # Sort features by average attention weight
    sorted_indices = np.argsort(significant_weights)[::-1]
    significant_features = [significant_features[i] for i in sorted_indices]
    significant_indices = significant_indices[sorted_indices]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for layer in range(num_layers):
        for head in range(num_heads):
            fig, ax = plt.subplots(figsize=(20, max(10, len(significant_features) * 0.3)))

            heatmap_data = head_scores[layer, head][significant_indices, :]

            sns.heatmap(heatmap_data, 
                        ax=ax, cmap="YlOrRd", 
                        xticklabels=range(1, head_dim + 1), 
                        yticklabels=significant_features)
            ax.set_title(f"Dynamic Indices Attention Weights - Layer {layer + 1}, Head {head + 1}")
            ax.set_xlabel("Head Embedding Dimension")
            ax.set_ylabel("Dynamic Indices Features with Attention Weight >= 0.01")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dynamic_indices_attention_weights_layer{layer + 1}_head{head + 1}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Print overall feature importance for significant features
    feature_importance = pd.DataFrame({
        'Feature': [clean_feature_names[i] for i in significant_indices],
        'Average Attention Weight': avg_weights[significant_indices]
    }).sort_values('Average Attention Weight', ascending=False)

    print("\nSignificant Dynamic Indices Features (Attention Weight >= 0.01):")
    print(feature_importance.to_string(index=False))

    # Save feature importance to CSV
    feature_importance.to_csv(os.path.join(output_dir, "significant_dynamic_indices_feature_importance.csv"), index=False)
    print(f"\nSignificant dynamic indices feature importance saved to {os.path.join(output_dir, 'significant_dynamic_indices_feature_importance.csv')}")

def create_top_features_table(head_scores: np.ndarray, feature_names: List[str], icd10_descriptions: Dict[str, str], cpt_descriptions: Dict[str, str], index_to_code: Dict[str, str]) -> pd.DataFrame:
    """Create a table of all features with positive attention weights across all layers and heads."""
    num_layers, num_heads, num_features, num_time_steps = head_scores.shape
    results = []

    for layer in range(num_layers):
        for head in range(num_heads):
            # Calculate average attention weights across time steps for this layer and head
            avg_weights = head_scores[layer, head].mean(axis=-1)

            for idx in range(len(avg_weights)):
                if avg_weights[idx] > 0:  # Only include features with positive attention weights
                    feature = feature_names[idx]
                    clean_feature = feature.replace('_Static', '').replace('_Dynamic', '')
                    feature_type = "Static" if "_Static" in feature else "Dynamic"
                    description = get_feature_description(clean_feature, icd10_descriptions, cpt_descriptions, index_to_code)

                    results.append({
                        "Layer": layer + 1,
                        "Head": head + 1,
                        "Feature": clean_feature,
                        "Type": feature_type,
                        "Description": description,
                        "Average Attention Weight": avg_weights[idx]
                    })

    # Convert results to DataFrame without sorting or limiting
    df = pd.DataFrame(results)
    return df

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

def print_nested_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent+1)
        else:
            print('  ' * (indent+1) + f"Shape: {value.shape}")

def visualize_feature_attention(attention_weights_dict, output_dir):
    feature_types = ['static_indices', 'SDI_score_normalized', 'AgeYears_normalized', 'InitialA1c_normalized']
    
    for feature_type in feature_types:
        weights = np.array(attention_weights_dict[feature_type])
        avg_weights = weights.mean(axis=(0, 1, 2))  # Average across layers, heads, and samples
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(avg_weights)), avg_weights)
        plt.title(f'Average Attention Weights for {feature_type}')
        plt.xlabel('Feature Index')
        plt.ylabel('Average Attention Weight')
        plt.savefig(os.path.join(output_dir, f'{feature_type}_attention_weights.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Process attention weights from a PyTorch Lightning checkpoint file.")
    parser.add_argument("checkpoint_path", help="Path to the PyTorch Lightning checkpoint file")
    parser.add_argument("--use_labs", action="store_true", help="Whether to use labs data")
    parser.add_argument("--config_path", help="Path to the finetune_config.yaml file")
    parser.add_argument("--create_heatmaps", action="store_true", help="Whether to creat attention heatmaps.")
    args = parser.parse_args()

    # Load the config file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    head_dim = config['config']['head_dim']

    if args.use_labs:
        DATA_SUMMARIES_DIR = Path("data_summaries/labs")
        DATA_DIR = Path("data/labs")
    else:
        DATA_SUMMARIES_DIR = Path("data_summaries")
        DATA_DIR = Path("data")

    os.makedirs(DATA_SUMMARIES_DIR, exist_ok=True)

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: The checkpoint file '{args.checkpoint_path}' does not exist.")
        sys.exit(1)

    attention_weights = load_attention_weights_from_checkpoint(args.checkpoint_path)
    print("Attention weights structure:")
    print_nested_dict(attention_weights)

    static_measurement_indices, static_indices, dynamic_indices = load_vocabularies(DATA_DIR)

    # Load labels
    labels = load_labels(DATA_DIR)

    # Load input data
    dynamic_indices, dynamic_values = load_dynamic_data(DATA_DIR)
    continuous_features = load_continuous_features(DATA_DIR)

    # Create input tensor
    input_data = np.concatenate([dynamic_indices, dynamic_values, continuous_features], axis=-1)
    input_data = torch.tensor(input_data, dtype=torch.float32)

    attention_weights_dict = process_attention_weights(attention_weights, head_dim, input_data, static_indices, ['SDI_score_normalized', 'AgeYears_normalized', 'InitialA1c_normalized'])

    if args.create_heatmaps:
        feature_names = create_feature_names(static_measurement_indices, static_indices, dynamic_indices, dynamic_values)
        # Only create heatmap for dynamic_indices
        if 'dynamic_indices' in attention_weights_dict:
            create_heatmap(np.array(attention_weights_dict['dynamic_indices']), None, feature_names, os.path.join(DATA_SUMMARIES_DIR, 'dynamic_indices'))

    # Process and print results for each feature type
    for feature_type, weights in attention_weights_dict.items():
        print(f"\nProcessing {feature_type}:")
        for layer, layer_weights in enumerate(weights):
            for head, head_weights in enumerate(layer_weights):
                print(f"  Layer {layer + 1}, Head {head + 1}:")
                if feature_type in ['dynamic_indices', 'dynamic_values', 'static_indices']:
                    vocab = dynamic_indices if feature_type == 'dynamic_indices' else (
                        dynamic_values if feature_type == 'dynamic_values' else static_indices
                    )
                    top_indices = head_weights.mean(axis=0).argsort()[-10:][::-1]
                    for idx in top_indices:
                        feature_name = map_index_to_vocab(idx, vocab)
                        print(f"    {feature_name}: {head_weights.mean(axis=0)[idx]:.6f}")
                else:
                    print(f"    Average attention weight: {head_weights.mean():.6f}")

    # Load ICD-10 and CPT descriptions
    icd10_descriptions = load_icd10_descriptions(DATA_DIR / 'ICD10.txt')
    cpt_descriptions = load_cpt_descriptions(DATA_DIR / '2024_DHS_Code_List_Addendum_03_01_2024.txt', DATA_DIR / 'CPT.txt')

    # Load index_to_code mapping
    with open(DATA_DIR / 'index_to_code.json', 'r') as f:
        index_to_code = json.load(f)

    continuous_features = ['SDI_score_normalized', 'AgeYears_normalized', 'InitialA1c_normalized']

    # Create and save summary tables
    for feature_type in list(attention_weights_dict.keys()) + ['continuous_static']:
        try:
            if feature_type == 'continuous_static':
                summary_df = create_summary_table(attention_weights_dict, feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features)
                labeled_summary_df = create_labeled_summary_table(attention_weights_dict, feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features, labels)
            else:
                summary_df = create_summary_table(attention_weights_dict[feature_type], feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features)
                labeled_summary_df = create_labeled_summary_table(attention_weights_dict[feature_type], feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features, labels)

            summary_df.to_csv(DATA_SUMMARIES_DIR / f"{feature_type}_attention_weights_summary.csv", index=False)
            labeled_summary_df.to_csv(DATA_SUMMARIES_DIR / f"{feature_type}_attention_weights_summary_labeled.csv", index=False)
            
            print(f"\nSummary table for {feature_type} saved as '{feature_type}_attention_weights_summary.csv' in {DATA_SUMMARIES_DIR}")
            print(f"\nLabeled summary table for {feature_type} saved as '{feature_type}_attention_weights_summary_labeled.csv' in {DATA_SUMMARIES_DIR}")
            
            print(f"\nTop 20 features for {feature_type}:")
            print(summary_df.head(20).to_string(index=False))
            
            print(f"\nTop 10 features for positive labels in {feature_type}:")
            print(labeled_summary_df[labeled_summary_df['Label'] == 'Positive'].head(10).to_string(index=False))
            
            print(f"\nTop 10 features for negative labels in {feature_type}:")
            print(labeled_summary_df[labeled_summary_df['Label'] == 'Negative'].head(10).to_string(index=False))
        except Exception as e:
            print(f"Error processing {feature_type}: {str(e)}")
            print("DataFrame columns:", summary_df.columns)
            print("DataFrame head:")
            print(summary_df.head().to_string())

    # Print information about continuous variables
    for var in continuous_features:
        if var in attention_weights_dict:
            weights = attention_weights_dict[var]
            avg_weight = np.mean([np.mean(layer_weights) for layer_weights in weights])
            print(f"\nAverage attention weight for {var}: {avg_weight:.6f}")

    # Add this debug print
    print("\nAttention weights dictionary keys:", attention_weights_dict.keys())
    print("\nShape of attention weights for each feature type:")
    for k, v in attention_weights_dict.items():
        print(f"{k}: {np.array(v).shape}")

    print(f"Input data shape: {input_data.shape}")
    print(f"Dynamic indices shape: {dynamic_indices.shape}")
    print(f"Dynamic values shape: {dynamic_values.shape}")
    print(f"Static indices: {static_indices}")

    # Call this function in the main() function
    visualize_feature_attention(attention_weights_dict, DATA_SUMMARIES_DIR)

# Create a mapping function for indices to vocabulary items
def map_index_to_vocab(index, vocab_dict):
    if isinstance(vocab_dict, np.ndarray):
        # For dynamic_indices, which is a numpy array
        return f"Dynamic_{index}"
    elif isinstance(vocab_dict, dict):
        # For static_indices, which is a dictionary
        for key, value in vocab_dict.items():
            if isinstance(value, dict):
                if index in value.values():
                    return f"{key}_{list(value.keys())[list(value.values()).index(index)]}"
            elif index == value:
                return key
    return f"Unknown_{index}"

def create_labeled_summary_table(weights, feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features, labels):
    # Ensure we're using the correct number of samples
    num_samples = min(len(dynamic_indices), len(labels))
    
    positive_indices = np.where(labels[:num_samples] == 1)[0]
    negative_indices = np.where(labels[:num_samples] == 0)[0]

    positive_summary = create_summary_table(weights, feature_type, dynamic_indices, dynamic_values, static_indices, 
                                            icd10_descriptions, cpt_descriptions, index_to_code, 
                                            continuous_features, positive_indices)
    negative_summary = create_summary_table(weights, feature_type, dynamic_indices, dynamic_values, static_indices, 
                                            icd10_descriptions, cpt_descriptions, index_to_code, 
                                            continuous_features, negative_indices)

    positive_summary['Label'] = 'Positive'
    negative_summary['Label'] = 'Negative'

    combined_summary = pd.concat([positive_summary, negative_summary])
    return combined_summary

def create_summary_table(weights, feature_type, dynamic_indices, dynamic_values, static_indices, icd10_descriptions, cpt_descriptions, index_to_code, continuous_features, indices=None):
    results = []

    if feature_type == 'static_indices':
        for layer in range(len(weights)):
            for head in range(len(weights[layer])):
                for idx, (feature, values) in enumerate(static_indices.items()):
                    weight = np.mean(weights[layer][head][:, idx])
                    results.append({
                        'Feature': f"{feature}_Static",
                        'Code': None,
                        'Description': f"Static feature: {feature}",
                        'Average_Attention_Weight': float(weight),
                        'Head': head + 1,
                        'Layer': layer + 1
                    })
    elif feature_type == 'continuous_static':
        for layer in range(len(weights['SDI_score_normalized'])):
            for head in range(len(weights['SDI_score_normalized'][layer])):
                for cont_feature in continuous_features:
                    weight = np.mean(weights[cont_feature][layer][head])
                    results.append({
                        'Feature': cont_feature,
                        'Code': None,
                        'Description': f'Continuous variable: {cont_feature}',
                        'Average_Attention_Weight': float(weight),
                        'Head': head + 1,
                        'Layer': layer + 1
                    })
    else:
        for layer in range(len(weights)):
            for head in range(len(weights[layer])):
                head_weights = weights[layer][head]
                if indices is not None:
                    head_weights = head_weights[indices]
                head_weights = head_weights.mean(axis=0)  # Mean across time steps

                if feature_type in ['dynamic_indices', 'dynamic_values', 'static_indices']:
                    vocab = dynamic_indices if feature_type == 'dynamic_indices' else (
                        dynamic_values if feature_type == 'dynamic_values' else static_indices
                    )
                    feature_names = [map_index_to_vocab(i, vocab) for i in range(head_weights.shape[0])]

                    for i, weight in enumerate(head_weights):
                        if weight > 0:
                            feature = feature_names[i]
                            clean_feature = feature.replace('_Static', '').replace('_Dynamic', '')
                            code = index_to_code.get(str(i), None)  # Convert i to string
                            description = get_feature_description(clean_feature, icd10_descriptions, cpt_descriptions, index_to_code)

                            result = {
                                'Feature': feature,
                                'Code': code,
                                'Description': description,
                                'Average_Attention_Weight': float(weight),
                                'Head': head + 1,
                                'Layer': layer + 1
                            }

                            if feature_type == 'dynamic_values':
                                result['Value'] = dynamic_values[0, i]  # Assuming the first sample is representative

                            results.append(result)
                elif feature_type in continuous_features:
                    results.append({
                        'Feature': feature_type,
                        'Code': None,
                        'Description': f'Continuous variable: {feature_type}',
                        'Average_Attention_Weight': float(head_weights.mean()),
                        'Head': head + 1,
                        'Layer': layer + 1
                    })

    df = pd.DataFrame(results)

    if df.empty:
        print(f"Warning: No positive attention weights found for {feature_type}")
        return df

    # Reorder columns
    if feature_type == 'dynamic_values':
        column_order = ['Feature', 'Code', 'Value', 'Description', 'Average_Attention_Weight', 'Head', 'Layer']
    else:
        column_order = ['Feature', 'Code', 'Description', 'Average_Attention_Weight', 'Head', 'Layer']

    df = df[column_order]

    if feature_type == 'dynamic_values':
        # Filter out ICD and CPT codes
        df = df[~df['Code'].str.contains('_ICD10|_CPT', na=False)]

    return df.sort_values('Average_Attention_Weight', ascending=False)

def get_feature_description(feature: str, icd10_descriptions: Dict[str, str], cpt_descriptions: Dict[str, str], index_to_code: Dict[str, str]) -> str:
    if feature.startswith("Dynamic_"):
        index = feature.split('_')[1]
        code = index_to_code.get(index)

        if code is None:
            return f"Unknown code for index {index}"

        if "_ICD10" in code:
            icd10_code = code.replace("_ICD10", "")
            return icd10_descriptions.get(icd10_code, f"Unknown ICD-10 code {icd10_code}")
        elif "_CPT" in code:
            cpt_code = code.replace("_CPT", "")
            return cpt_descriptions.get(cpt_code, f"Unknown CPT code {cpt_code}")
        else:
            return code  # Return the lab code itself for lab tests

    elif feature.startswith("Static_"):
        return "Static feature"

    elif feature in ['SDI_score_normalized', 'AgeYears_normalized', 'InitialA1c_normalized']:
        return f"Continuous variable: {feature}"

    return "Unknown feature"

def load_dynamic_data(directory, max_seq_length=1000):
    """Load dynamic indices and values from the parquet files."""
    try:
        dfs = []
        for i in range(3):
            df = pq.read_table(directory / f"DL_reps/train_{i}.parquet").to_pandas()
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('subject_id')

        def truncate_or_pad_array(arr, max_len, pad_value):
            if len(arr) > max_len:
                return arr[:max_len]
            return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_value)

        dynamic_indices = np.array([truncate_or_pad_array(seq, max_seq_length, -1) for seq in df['dynamic_indices']])
        dynamic_values = np.array([truncate_or_pad_array(seq, max_seq_length, np.nan) for seq in df['dynamic_values']])

        dynamic_values = np.where(np.isnan(dynamic_values), -1, dynamic_values)

        print(f"Loaded dynamic data. Shape of dynamic indices: {dynamic_indices.shape}")
        print(f"Shape of dynamic values: {dynamic_values.shape}")

        print(f"Dynamic indices range: [{dynamic_indices.min()}, {dynamic_indices.max()}]")
        print(f"Dynamic values range: [{dynamic_values.min()}, {dynamic_values.max()}]")
        return dynamic_indices, dynamic_values

    except Exception as e:
        print(f"Error loading dynamic data: {str(e)}")
        raise
        
def load_continuous_features(directory):
    """Load continuous features from the parquet files."""
    try:
        dfs = []
        for i in range(3):
            df = pq.read_table(directory / f"DL_reps/train_{i}.parquet").to_pandas()
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        continuous_features = df[['InitialA1c_normalized', 'AgeYears_normalized', 'SDI_score_normalized']].values
        print(f"Continuous features shape: {continuous_features.shape}")
        print(f"Continuous features range: [{continuous_features.min()}, {continuous_features.max()}]")
        return continuous_features
    except Exception as e:
        print(f"Error loading continuous features: {str(e)}")
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