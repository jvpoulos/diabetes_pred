import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import os
import sys
import json
import pyarrow.parquet as pq
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from EventStream.transformer.fine_tuning_model import ESTForStreamClassification
from EventStream.transformer.lightning_modules.fine_tuning import ESTForStreamClassificationLM
from EventStream.transformer.config import StructuredTransformerConfig, OptimizationConfig
from EventStream.data.vocabulary import VocabularyConfig

def load_model_and_data(checkpoint_path, config_path, use_labs):
    # Load the finetune config file
    with open(config_path, 'r') as f:
        finetune_config = yaml.safe_load(f)

    if use_labs:
        DATA_SUMMARIES_DIR = Path("data_summaries/labs")
        DATA_DIR = Path("data/labs")
    else:
        DATA_SUMMARIES_DIR = Path("data_summaries")
        DATA_DIR = Path("data")

    os.makedirs(DATA_SUMMARIES_DIR, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"Error: The checkpoint file '{checkpoint_path}' does not exist.")
        sys.exit(1)

    # Load model configuration
    config = StructuredTransformerConfig.from_json_file(DATA_DIR / "config.json")
    optimization_config = OptimizationConfig.from_json_file(DATA_DIR / "optimization_config.json")
    vocabulary_config = VocabularyConfig.from_json_file(DATA_DIR / "vocabulary_config.json")

    # Update config with values from finetune_config.yaml
    config.hidden_size = finetune_config['config']['hidden_size']
    config.intermediate_size = finetune_config['config'].get('intermediate_size', config.hidden_size)
    config.num_hidden_layers = finetune_config['config']['num_hidden_layers']
    config.num_attention_heads = finetune_config['config']['num_attention_heads']
    config.head_dim = finetune_config['config']['head_dim']

    # Update seq_attention_types
    seq_attention_types = finetune_config['config'].get('seq_attention_types', ['global', 'local'])
    config.seq_attention_types = seq_attention_types * config.num_hidden_layers
    config.seq_attention_types = config.seq_attention_types[:config.num_hidden_layers]

    # Explicitly set the attention_layers attribute
    config.seq_attention_layers = config.seq_attention_types
    config.attention_layers = config.seq_attention_layers

    # Initialize the model with the updated config
    model = ESTForStreamClassification(
        config=config,
        vocabulary_config=vocabulary_config,
        optimization_config=optimization_config,
        oov_index=config.vocab_size
    )

    # Load the state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Remove the 'model.' prefix from the state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    # Load the modified state dict
    model.load_state_dict(new_state_dict, strict=False)

    # Load input data
    dynamic_indices, dynamic_values = load_dynamic_data(DATA_DIR)
    continuous_features = load_continuous_features(DATA_DIR)

    # Create input tensor
    input_data = np.concatenate([dynamic_indices, dynamic_values, continuous_features], axis=-1)
    input_data = torch.tensor(input_data, dtype=torch.float32)

    # Load labels
    labels_df = load_labels(DATA_DIR)
    labels = labels_df['label'].values

    return model, input_data, labels

def load_dynamic_data(directory, max_seq_length=1000):
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

        return dynamic_indices, dynamic_values
    except Exception as e:
        print(f"Error loading dynamic data: {str(e)}")
        raise

def load_continuous_features(directory):
    try:
        dfs = []
        for i in range(3):
            df = pq.read_table(directory / f"DL_reps/train_{i}.parquet").to_pandas()
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        continuous_features = df[['InitialA1c_normalized', 'AgeYears_normalized', 'SDI_score_normalized']].values
        return continuous_features
    except Exception as e:
        print(f"Error loading continuous features: {str(e)}")
        raise

def load_labels(data_dir: Path) -> pd.DataFrame:
    task_dfs_dir = data_dir / "task_dfs"
    label_files = [
        "a1c_greater_than_7_train.parquet",
        "a1c_greater_than_7_val.parquet",
        "a1c_greater_than_7_test.parquet"
    ]
    
    all_labels = []
    for file in label_files:
        file_path = task_dfs_dir / file
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_labels.append(df)
        else:
            print(f"Warning: {file} not found in {task_dfs_dir}")
    
    if not all_labels:
        raise FileNotFoundError(f"No label files found in {task_dfs_dir}")
    
    labels_df = pd.concat(all_labels, ignore_index=True)
    
    if 'subject_id' not in labels_df.columns or 'label' not in labels_df.columns:
        raise ValueError("Labels DataFrame must contain 'subject_id' and 'label' columns")
    
    labels_df['label'] = labels_df['label'].astype(int)
    
    return labels_df[['subject_id', 'label']]

def integrated_gradients_analysis(model, inputs, target):
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(inputs, target=target, return_convergence_delta=True)
    return attributions, delta

def layer_conductance_analysis(model, inputs, target, layer):
    lc = LayerConductance(model, layer)
    attributions = lc.attribute(inputs, target=target)
    return attributions

def neuron_conductance_analysis(model, inputs, target, layer, neuron_selector):
    nc = NeuronConductance(model, layer)
    attributions = nc.attribute(inputs, neuron_selector=neuron_selector, target=target)
    return attributions

def visualize_attributions(attributions, feature_names, title):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(attributions)), attributions)
    plt.xlabel('Features')
    plt.ylabel('Attribution')
    plt.title(title)
    plt.xticks(range(len(attributions)), feature_names, rotation='vertical')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Apply Captum attribution techniques to analyze the transformer model.")
    parser.add_argument("checkpoint_path", help="Path to the PyTorch Lightning checkpoint file")
    parser.add_argument("--use_labs", action="store_true", help="Whether to use labs data")
    parser.add_argument("--config_path", help="Path to the finetune_config.yaml file")
    args = parser.parse_args()

    model, input_data, labels = load_model_and_data(args.checkpoint_path, args.config_path, args.use_labs)
    
    # Assuming single-label binary classification
    target = 1  # For positive class
    
    # Integrated Gradients
    ig_attributions, ig_delta = integrated_gradients_analysis(model, input_data, target)
    visualize_attributions(ig_attributions.sum(0), model.config.feature_names, "Integrated Gradients Feature Attribution")
    
    # Layer Conductance
    # Assuming the first transformer layer
    layer = model.encoder.h[0]
    lc_attributions = layer_conductance_analysis(model, input_data, target, layer)
    visualize_attributions(lc_attributions.mean(0), range(lc_attributions.shape[1]), "Layer Conductance Attribution")
    
    # Neuron Conductance
    # Assuming we're interested in the first neuron of the first layer
    nc_attributions = neuron_conductance_analysis(model, input_data, target, layer, neuron_selector=0)
    visualize_attributions(nc_attributions.mean(0), model.config.feature_names, "Neuron Conductance Attribution")

if __name__ == "__main__":
    main()