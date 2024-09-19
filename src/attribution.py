import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import os
import gc
import sys
import json
import types
import pyarrow.parquet as pq
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from EventStream.transformer.fine_tuning_model import ESTForStreamClassification
from EventStream.transformer.lightning_modules.fine_tuning import ESTForStreamClassificationLM
from EventStream.transformer.config import StructuredTransformerConfig, OptimizationConfig
from EventStream.transformer.transformer import ConditionallyIndependentPointProcessInputLayer
from EventStream.data.vocabulary import VocabularyConfig
from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.data.data_embedding_layer import DataEmbeddingLayer, EmbeddingMode

def safe_tensor_operation(tensor, operation):
    try:
        if operation == torch.mean and tensor.dtype == torch.long:
            return tensor.float().mean().item()
        return operation(tensor).item()
    except RuntimeError as e:
        print(f"Error occurred: {str(e)}")
        print(f"Tensor dtype: {tensor.dtype}, shape: {tensor.shape}")
        if tensor.numel() > 0:
            return operation(tensor.float()).item()
        return None

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

class SimpleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def forward(self, dynamic_indices):
        print(f"Input shape: {dynamic_indices.shape}")
        
        dynamic_indices = dynamic_indices.long().to(self.device)
        
        # Create a dummy input dictionary with all required fields
        input_dict = {
            'dynamic_indices': dynamic_indices,
            'dynamic_values': torch.ones_like(dynamic_indices, dtype=torch.float).to(self.device),
            'dynamic_measurement_indices': torch.zeros_like(dynamic_indices).to(self.device),
            'static_indices': torch.zeros((dynamic_indices.size(0), 1), dtype=torch.long).to(self.device),
            'static_measurement_indices': torch.zeros((dynamic_indices.size(0), 1), dtype=torch.long).to(self.device),
            'time': torch.zeros_like(dynamic_indices, dtype=torch.float).to(self.device)
        }
        
        # Process through the encoder
        output = self.model.encoder(input_dict)
        
        if isinstance(output, dict):
            output = output.get('last_hidden_state', output)
        
        # Apply final layer
        output = self.model.logit_layer(output[:, -1, :])  # Use the last token's embedding
        
        return output.squeeze(-1)  # Ensure output is 1D

    def preprocess_input(self, input_dict):
        # Create placeholder values for missing arguments
        batch_size, seq_length = input_dict['dynamic_indices'].shape
        values = input_dict.get('dynamic_values', torch.ones_like(input_dict['dynamic_indices'], dtype=torch.float))
        values_mask = input_dict.get('dynamic_values_mask', torch.ones_like(input_dict['dynamic_indices'], dtype=torch.bool))
        
        return PytorchBatch(
            dynamic_indices=input_dict['dynamic_indices'],
            dynamic_values=values,
            dynamic_measurement_indices=input_dict['dynamic_measurement_indices'],
            static_indices=input_dict['static_indices'],
            static_measurement_indices=input_dict['static_measurement_indices'],
            time=input_dict['time'],
            dynamic_values_mask=values_mask,
            event_mask=None,
            time_delta=None,
            dynamic_indices_event_type=None,
            start_time=None,
            start_idx=None,
            end_idx=None,
            subject_id=None,
            InitialA1c_normalized=None,
            AgeYears_normalized=None,
            SDI_score_normalized=None,
            time_to_index=None,
            stream_labels=None,
            event_type=None
        )

def integrated_gradients_analysis(model, inputs, target):
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)
    
    # Process data in smaller batches
    batch_size = 64  # Reduced batch size
    num_batches = (next(iter(inputs.values())).shape[0] + batch_size - 1) // batch_size
    
    all_attributions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, next(iter(inputs.values())).shape[0])
        
        batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs.items()}
        
        # Convert inputs dictionary to a tuple of tensors
        input_tensors = tuple(batch_inputs.values())
        
        # Create baseline inputs
        baselines = tuple(torch.zeros_like(tensor) for tensor in input_tensors)
        
        # Ensure all input tensors are on the same device
        device = next(model.parameters()).device
        input_tensors = tuple(tensor.to(device) for tensor in input_tensors)
        baselines = tuple(tensor.to(device) for tensor in baselines)
        
        # Attribute
        try:
            attributions = ig.attribute(input_tensors, baselines=baselines, target=target)
        except Exception as e:
            print(f"Error during attribution: {str(e)}")
            print("Input tensor shapes and statistics:")
            for i, tensor in enumerate(input_tensors):
                print(f"Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
                print(f"         min={safe_tensor_operation(tensor, torch.min)}, "
                      f"max={safe_tensor_operation(tensor, torch.max)}, "
                      f"mean={safe_tensor_operation(tensor, torch.mean)}")
            raise
        
        # Combine attributions from different input components
        combined_attributions = torch.cat([attr.view(attr.size(0), -1) for attr in attributions], dim=1)
        all_attributions.append(combined_attributions)
        
        # Clear unnecessary variables from memory
        del batch_inputs, input_tensors, baselines, attributions
        gc.collect()
        torch.cuda.empty_cache()
    
    # Combine attributions from all batches
    final_attributions = torch.cat(all_attributions, dim=0)
    
    return final_attributions, None  # We're not using delta in this case

def config_to_dict(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [config_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: config_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {key: config_to_dict(value) for key, value in obj.__dict__.items()
                if not key.startswith('_')}
    else:
        return str(obj)

def main():
    parser = argparse.ArgumentParser(description="Apply Captum attribution techniques to analyze the transformer model.")
    parser.add_argument("checkpoint_path", help="Path to the PyTorch Lightning checkpoint file")
    parser.add_argument("--use_labs", action="store_true", help="Whether to use labs data")
    parser.add_argument("--config_path", help="Path to the finetune_config.yaml file")
    args = parser.parse_args()

    model, input_data, labels = load_model_and_data(args.checkpoint_path, args.config_path, args.use_labs)
    
    wrapped_model = ModelWrapper(model)
    
    # Assuming single-label binary classification
    target = 1  # For positive class
    
    # Integrated Gradients
    ig = IntegratedGradients(wrapped_model)
    
    batch_size = 32  # Adjust this based on your GPU memory
    num_batches = (input_data['dynamic_indices'].shape[0] + batch_size - 1) // batch_size
    
    all_attributions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, input_data['dynamic_indices'].shape[0])
        
        batch_input = input_data['dynamic_indices'][start_idx:end_idx].to(wrapped_model.device)
        
        attributions = ig.attribute(batch_input, target=target)
        all_attributions.append(attributions.cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    attributions = torch.cat(all_attributions, dim=0)
    
    # Visualize attributions
    feature_names = [f"Feature_{i}" for i in range(attributions.shape[1])]
    visualize_attributions(attributions.sum(0), feature_names, "Integrated Gradients Feature Attribution")
    
    # Layer Conductance and Neuron Conductance can be implemented similarly if needed

if __name__ == "__main__":
    main()