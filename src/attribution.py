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
from EventStream.transformer.transformer import ConditionallyIndependentPointProcessInputLayer
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

    # Add default value for intermediate_dropout if not present
    if not hasattr(config, 'intermediate_dropout'):
        config.intermediate_dropout = finetune_config['config'].get('intermediate_dropout', 0.1)

    # Update seq_attention_types
    seq_attention_types = finetune_config['config'].get('seq_attention_types', ['global', 'local'])
    config.seq_attention_types = seq_attention_types * config.num_hidden_layers
    config.seq_attention_types = config.seq_attention_types[:config.num_hidden_layers]

    # Explicitly set the attention_layers attribute
    config.seq_attention_layers = config.seq_attention_types
    config.attention_layers = config.seq_attention_layers

    # Load the state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Update vocab size in config
    static_embedding_weight = state_dict.get('model.static_indices_embedding.weight')
    if static_embedding_weight is not None:
        config.vocab_size = static_embedding_weight.shape[0]

    # Set oov_index to be the last index of the vocabulary
    oov_index = config.vocab_size - 1

    # Update vocabulary_config with the new vocab_size
    vocabulary_config.vocab_sizes_by_measurement = {k: min(v, config.vocab_size) for k, v in vocabulary_config.vocab_sizes_by_measurement.items()}

    # Initialize the model with the updated config
    model = ESTForStreamClassification(
        config=config,
        vocabulary_config=vocabulary_config,
        optimization_config=optimization_config,
        oov_index=oov_index
    )

    # Manually update the encoder's input layer
    model.encoder.input_layer = ConditionallyIndependentPointProcessInputLayer(
        config=config,
        vocab_sizes_by_measurement=vocabulary_config.vocab_sizes_by_measurement,
        oov_index=oov_index
    )

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
    dynamic_data = load_dynamic_data(DATA_DIR)
    static_data = load_static_data(DATA_DIR)

    # Create input tensor
    input_data = {
        'dynamic_indices': torch.tensor(dynamic_data['dynamic_indices'], dtype=torch.long),
        'dynamic_values': torch.tensor(dynamic_data['dynamic_values'], dtype=torch.float),
        'dynamic_measurement_indices': torch.tensor(dynamic_data['dynamic_measurement_indices'], dtype=torch.long),
        'static_indices': torch.tensor(static_data['static_indices'], dtype=torch.long),
        'static_measurement_indices': torch.tensor(static_data['static_measurement_indices'], dtype=torch.long),
        'time': torch.tensor(dynamic_data['time'], dtype=torch.float),
    }

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

        def pad_sequence(seq, max_len, pad_value):
            seq = np.array(seq)
            if len(seq) > max_len:
                return seq[:max_len]
            return np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=pad_value)

        dynamic_indices = np.array([pad_sequence(seq, max_seq_length, -1) for seq in df['dynamic_indices']])
        dynamic_values = np.array([pad_sequence(seq, max_seq_length, np.nan) for seq in df['dynamic_values']])
        dynamic_measurement_indices = np.array([pad_sequence(seq, max_seq_length, 0) for seq in df['dynamic_measurement_indices']])
        time = np.array([pad_sequence(seq, max_seq_length, 0) for seq in df['time']])

        dynamic_values = np.where(np.isnan(dynamic_values), -1, dynamic_values)

        return {
            'dynamic_indices': dynamic_indices,
            'dynamic_values': dynamic_values,
            'dynamic_measurement_indices': dynamic_measurement_indices,
            'time': time
        }
    except Exception as e:
        print(f"Error loading dynamic data: {str(e)}")
        raise

def load_static_data(directory):
    try:
        dfs = []
        for i in range(3):
            df = pq.read_table(directory / f"DL_reps/train_{i}.parquet").to_pandas()
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('subject_id')

        max_static_length = max(len(seq) for seq in df['static_indices'])

        def pad_or_truncate_sequence(seq, max_len, pad_value):
            seq = np.array(seq)
            if len(seq) > max_len:
                return seq[:max_len]
            return np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=pad_value)

        static_indices = np.array([pad_or_truncate_sequence(seq, max_static_length, 0) for seq in df['static_indices']])
        static_measurement_indices = np.array([pad_or_truncate_sequence(seq, max_static_length, 0) for seq in df['static_measurement_indices']])

        return {
            'static_indices': static_indices,
            'static_measurement_indices': static_measurement_indices
        }
    except Exception as e:
        print(f"Error loading static data: {str(e)}")
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

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def integrated_gradients_analysis(model, inputs, target):
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)
    
    # Create baseline inputs
    baselines = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            baselines[key] = torch.zeros_like(value)
    
    attributions = ig.attribute(inputs, baselines=baselines, target=target)
    
    # Combine attributions from different input components
    combined_attributions = torch.cat([attr.view(attr.size(0), -1) for attr in attributions.values()], dim=1)
    
    return combined_attributions, None  # We're not using delta in this case

def layer_conductance_analysis(model, inputs, target, layer):
    wrapped_model = ModelWrapper(model)
    lc = LayerConductance(wrapped_model, layer)
    
    attributions = lc.attribute(inputs, target=target)
    
    # Combine attributions from different input components
    combined_attributions = torch.cat([attr.view(attr.size(0), -1) for attr in attributions.values()], dim=1)
    
    return combined_attributions

def neuron_conductance_analysis(model, inputs, target, layer, neuron_selector):
    wrapped_model = ModelWrapper(model)
    nc = NeuronConductance(wrapped_model, layer)
    
    attributions = nc.attribute(inputs, neuron_selector=neuron_selector, target=target)
    
    # Combine attributions from different input components
    combined_attributions = torch.cat([attr.view(attr.size(0), -1) for attr in attributions.values()], dim=1)
    
    return combined_attributions

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
    ig_attributions, _ = integrated_gradients_analysis(model, input_data, target)
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