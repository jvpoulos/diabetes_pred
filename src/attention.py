import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tab_transformer_pytorch import TabTransformer, FTTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import logging


def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else 'module.' + k  # Adjust keys
        new_state_dict[name] = v
    return new_state_dict

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Since features and labels are already tensors, no conversion is needed
        return self.features[idx], self.labels[idx]

def load_model(model_type, model_path, dim, attn_dropout, categories, num_continuous):

    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            dim_out=1,
            depth=6,
            heads=8,
            attn_dropout=attn_dropout,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU()
        )
    elif model_type == 'FTTransformer':
        model = FTTransformer(
            categories = categories,
            num_continuous = num_continuous,
            dim = dim,
            dim_out = 1,
            depth = 3,
            heads = 8,
            attn_dropout = attn_dropout,
            ff_dropout = 0.1
        )
    else:
        raise ValueError("Invalid model type. Choose 'TabTransformer' or 'FTTransformer'.")

    # Load the state dict with fix_state_dict applied
    state_dict = torch.load(model_path)
    model.load_state_dict(fix_state_dict(state_dict))
    model.eval()
    return model

def get_attention_maps(model, loader, binary_feature_indices, numerical_feature_indices):
    attention_maps = []
    for batch in loader:
        features = batch[0].to(model.device)
        labels = batch[1].to(model.device)  # If labels are needed

        # Segregate categorical and continuous features using their indices
        x_categ = features[:, binary_feature_indices]
        x_cont = features[:, numerical_feature_indices]

        print(f"Actual x_categ shape: {x_categ.shape}, x_cont shape: {x_cont.shape}")
        _, attns = model(x_categ, x_cont, return_attn=True)
        attention_maps.append(attns)
    return attention_maps

def plot_attention_maps(attention_maps, model_type, model_path):
    # Assuming you are only interested in the attention from the last layer
    attention_map = attention_maps[-1].mean(dim=1)  # Taking the mean attention across heads
    avg_attention_map = attention_map.mean(dim=0)  # Further averaging across all batches
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention_map.cpu().detach().numpy(), cmap='viridis')
    plt.title(f'Attention Map - {model_type}')
    
    # Construct filename based on model_type and model_path
    filename = f"attention_map_{model_type}_{os.path.basename(model_path).replace('.pth', '')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Attention map saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['train','validate'], required=True, help='Specify dataset type for evaluation')
    parser.add_argument('--model_type', type=str, choices=['TabTransformer', 'FTTransformer'], help='Type of the model to load')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--outcome', type=str, required=True, choices=['A1cGreaterThan7', 'A1cLessThan7'], help='Outcome variable to predict')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    # Conditional defaults based on model_type
    if args.model_type == 'TabTransformer':
        if args.dim is None:
            args.dim = 32  # Default for TabTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.1  # Default for TabTransformer
    elif args.model_type == 'FTTransformer':
        if args.dim is None:
            args.dim = 192  # Default for FTTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.2  # Default for FTTransformer

    if args.dataset_type == 'train':
        features = torch.load('train_features.pt')
        labels = torch.load('train_labels.pt')
    elif args.dataset_type == 'validate':
        features = torch.load('validate_features.pt')
        labels = torch.load('validate_labels.pt')

    # Create a TensorDataset
    dataset = CustomDataset(features, labels)

    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    with open('columns_to_normalize.json', 'r') as file:
        columns_to_normalize = json.load(file)

    # Define excluded columns and additional binary variables
    excluded_columns = ["A1cGreaterThan7", "A1cLessThan7",  "studyID"]
    additional_binary_vars = ["Female", "Married", "GovIns", "English", "Veteran"]

    # Filter out excluded columns
    column_names_filtered = [col for col in column_names if col not in excluded_columns]
    encoded_feature_names_filtered = [name for name in encoded_feature_names if name in column_names_filtered]

    # Combine and deduplicate encoded and additional binary variables
    binary_features_combined = list(set(encoded_feature_names_filtered + additional_binary_vars))

    # Calculate binary feature indices, ensuring they're within the valid range
    binary_feature_indices = [column_names_filtered.index(col) for col in binary_features_combined if col in column_names_filtered]

    # Find indices of the continuous features
    numerical_feature_indices = [column_names.index(col) for col in columns_to_normalize if col not in excluded_columns]

    categories = [2] * len(binary_feature_indices)
    print("Categories:", len(categories))

    num_continuous = len(numerical_feature_indices)
    print("Continuous:", num_continuous)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_type, args.model_path, args.dim, args.attn_dropout, categories, num_continuous)
    model.to(device)  # Move the model to the appropriate device
    model.device = device 

    attention_maps = get_attention_maps(model, data_loader, binary_feature_indices, numerical_feature_indices)
    plot_attention_maps(attention_maps, args.model_type, args.model_path)

if __name__ == '__main__':
    main()