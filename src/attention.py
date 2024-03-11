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
from tqdm import tqdm
import pandas as pd
import numpy as np

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

def load_model(model_type, model_path, dim, depth, heads, attn_dropout, ff_dropout, categories, num_continuous):

    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,  # tuple containing the number of unique values within each category
            num_continuous=num_continuous,                              # number of continuous values
            dim=dim,                                               # dimension, paper set at 32
            dim_out=1,                                                  # binary prediction, but could be anything
            depth=depth,                                           # depth, paper recommended 6
            heads=heads,                                           # heads, paper recommends 8
            attn_dropout=attn_dropout,                             # post-attention dropout
            ff_dropout=ff_dropout ,                                # feed forward dropout
            mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        )
    elif model_type == 'FTTransformer':
            model = FTTransformer(
            categories = categories,      # tuple containing the number of unique values within each category
            num_continuous = num_continuous,  # number of continuous values
            dim = dim,                     # dimension, paper set at 192
            dim_out = 1,                        # binary prediction, but could be anything
            depth = depth,                          # depth, paper recommended 3
            heads = heads,                          # heads, paper recommends 8
            attn_dropout = attn_dropout,   # post-attention dropout, paper recommends 0.2
            ff_dropout = ff_dropout                   # feed forward dropout, paper recommends 0.1
        )
    else:
        raise ValueError("Invalid model type. Choose 'TabTransformer' or 'FTTransformer'.")

    # Load the state dict with fix_state_dict applied
    state_dict = torch.load(model_path)
    model.load_state_dict(fix_state_dict(state_dict))
    model.eval()
    return model

def get_attention_maps(model, loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns, device):
    model.eval()
    model.to(device)
    
    top_attention_weights = []
    feature_names = []
    
    print("Starting attention map computation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            features, _ = batch
            x_categ = features[:, binary_feature_indices].to(device, dtype=torch.long)  # dtype is specified here
            x_cont = features[:, numerical_feature_indices].to(device)
            
            _, attns = model(x_categ, x_cont, return_attn=True)
            
            attns = attns if isinstance(attns, list) else [attns]
            
            for attn in attns:
                # Reduce the size of the tensor by averaging across heads and layers if they exist
                while attn.dim() > 3:
                    attn = attn.mean(dim=1)
                
                # Move the tensor to CPU to save GPU memory
                attn = attn.to('cpu')
                
                # Now attn should be of shape [batch_size, num_features, num_features]
                batch_size = attn.size(0)
                num_features = attn.size(1)
                
                # Compute the average attention weights for each feature across the batch
                avg_attn = attn.mean(dim=(0, 1))
                
                # Append to the lists
                top_attention_weights.append(avg_attn.cpu().numpy())
                feature_names.append(column_names[:num_features])
            
            # Optionally clear some cache if necessary
            torch.cuda.empty_cache()
    
    print("Attention map computation completed.")
    
    # Check if top_attention_weights and feature_names are empty
    if len(top_attention_weights) == 0 or len(feature_names) == 0:
        print("Warning: No attention weights or feature names computed.")
        return None, None
    
    print(f"Number of attention weight matrices: {len(top_attention_weights)}")
    print(f"Number of feature name lists: {len(feature_names)}")
    
    # Aggregate the attention weights and feature names across all batches
    aggregated_attention_weights = np.mean(top_attention_weights, axis=0)
    
    # Take the feature names from the first batch (assuming they are the same for all batches)
    aggregated_feature_names = feature_names[0]
    
    # Select the top 100 features based on the aggregated attention weights
    top_indices = np.argsort(aggregated_attention_weights)[::-1][:100]
    top_attention_weights = aggregated_attention_weights[top_indices]
    top_feature_names = [aggregated_feature_names[idx] for idx in top_indices]
    
    return top_attention_weights, top_feature_names

def identify_top_feature_values(model, loader, binary_feature_indices, numerical_feature_indices, column_names, top_feature_names, device):
    model.eval()
    model.to(device)
    
    feature_value_attention_weights = {feature: {} for feature in top_feature_names}
    
    print("Starting feature value attention weight computation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            features, labels = batch
            x_categ = features[:, binary_feature_indices].to(device, dtype=torch.long)  # dtype is specified here
            x_cont = features[:, numerical_feature_indices].to(device)
            
            _, attns = model(x_categ, x_cont, return_attn=True)
            
            attns = attns if isinstance(attns, list) else [attns]
            
            for attn in attns:
                # Reduce the size of the tensor by averaging across heads and layers if they exist
                while attn.dim() > 3:
                    attn = attn.mean(dim=1)
                
                # Move the tensor to CPU to save GPU memory
                attn = attn.to('cpu')
                
                # Now attn should be of shape [batch_size, num_features, num_features]
                batch_size = attn.size(0)
                num_features = attn.size(1)
                
                for feature_name in top_feature_names[:10]:  # Consider only the top 10 features
                    if feature_name in column_names:
                        feature_index = column_names.index(feature_name)
                        if feature_index < features.size(1):
                            feature_values = features[:, feature_index].cpu().numpy()
                            
                            for value, label, attention in zip(feature_values, labels.cpu().numpy(), attn[:, feature_index, feature_index].cpu().numpy()):
                                if label == 1:
                                    feature_value_attention_weights[feature_name].setdefault('HbA1c ≥ 7.0%', {}).setdefault(value, []).append(attention)
                                else:
                                    feature_value_attention_weights[feature_name].setdefault('HbA1c < 7.0%', {}).setdefault(value, []).append(attention)
            
            # Optionally clear some cache if necessary
            torch.cuda.empty_cache()
    
    print("Feature value attention weight computation completed.")
    
    # Compute average attention weights for each feature value and outcome
    for feature_name in top_feature_names[:10]:
        if feature_name in feature_value_attention_weights:
            for outcome in ['HbA1c ≥ 7.0%', 'HbA1c < 7.0%']:
                if outcome in feature_value_attention_weights[feature_name]:
                    for value in feature_value_attention_weights[feature_name][outcome]:
                        feature_value_attention_weights[feature_name][outcome][value] = np.mean(feature_value_attention_weights[feature_name][outcome][value])
    
    return feature_value_attention_weights

def save_attention_maps_to_html(attention_maps, feature_names, filename):
    attention_data = list(zip(feature_names, attention_maps))

    # Create a DataFrame to represent the attention data
    df = pd.DataFrame(attention_data, columns=['Feature', 'Attention Weight'])

    # Load the df_train_summary.csv file
    df_train_summary = pd.read_csv('df_train_summary.csv')

    # Merge the attention data with the descriptions from df_train_summary
    df = pd.merge(df, df_train_summary[['Feature', 'Description']], on='Feature', how='left')

    # Sort the DataFrame by the attention weights in descending order
    df = df.sort_values(by='Attention Weight', ascending=False)

    # Convert the DataFrame to an HTML table representation
    html_table = df.to_html(index=False)

    # Save the HTML table to a file
    with open(filename, 'w') as file:
        file.write('<h1>Average attention weights per feature across all patients, regardless of their outcome: Top 100 features</h1>')
        file.write(html_table)

    print(f"Attention maps saved to {filename}")

def save_top_feature_values_to_html(top_feature_value_attention_weights, filename):
    # Create a DataFrame for each outcome
    df_high = pd.DataFrame(columns=['Feature', 'Feature Value', 'Attention Weight'])
    df_low = pd.DataFrame(columns=['Feature', 'Feature Value', 'Attention Weight'])

    for feature_name, outcome_values in top_feature_value_attention_weights.items():
        if 'HbA1c ≥ 7.0%' in outcome_values:
            for (value, weight) in outcome_values['HbA1c ≥ 7.0%'].items():
                df_high = df_high.append({'Feature': feature_name, 'Feature Value': value, 'Attention Weight': weight}, ignore_index=True)
        if 'HbA1c < 7.0%' in outcome_values:
            for (value, weight) in outcome_values['HbA1c < 7.0%'].items():
                df_low = df_low.append({'Feature': feature_name, 'Feature Value': value, 'Attention Weight': weight}, ignore_index=True)

    # Load the df_train_summary.csv file
    df_train_summary = pd.read_csv('df_train_summary.csv')

    # Merge the attention data with the descriptions from df_train_summary
    df_high = pd.merge(df_high, df_train_summary[['Feature', 'Description']], on='Feature', how='left')
    df_low = pd.merge(df_low, df_train_summary[['Feature', 'Description']], on='Feature', how='left')

    # Sort the DataFrames by attention weights in descending order
    df_high = df_high.sort_values(by='Attention Weight', ascending=False)
    df_low = df_low.sort_values(by='Attention Weight', ascending=False)

    # Convert the DataFrames to HTML tables
    html_table_high = df_high.to_html(index=False)
    html_table_low = df_low.to_html(index=False)

    # Save the HTML tables to a file
    with open(filename, 'w') as file:
        file.write('<h1>Average attention weights for each feature value within each outcome group (HbA1c >= 7.0% and HbA1c < 7.0%): Top 10 features</h1>')
        file.write('<h2>Patients with HbA1c >= 7.0%</h2>')
        file.write(html_table_high)
        file.write('<h2>Patients with HbA1c < 7.0%</h2>')
        file.write(html_table_low)

    print(f"Top feature values saved to {filename}")
    
def main():
    parser = argparse.ArgumentParser(description='Extract attention maps.')
    parser.add_argument('--dataset_type', type=str, choices=['train','validation'], required=True, help='Specify dataset type for evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--outcome', type=str, required=True, choices=['A1cGreaterThan7', 'A1cLessThan7'], help='Outcome variable to predict')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--model_path', type=str, default=None,
                    help='Optional path to the saved model file to load before training')
    args = parser.parse_args()

    # Conditional defaults based on model_type
    if args.model_type == 'TabTransformer':
        if args.dim is None:
            args.dim = 32  # Default for TabTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.1  # Default for TabTransformer
        args.depth = args.depth if args.depth is not None else 6
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1
    elif args.model_type == 'FTTransformer':
        if args.dim is None:
            args.dim = 192  # Default for FTTransformer
        if args.attn_dropout is None:
            args.attn_dropout = 0.2  # Default for FTTransformer
        args.depth = args.depth if args.depth is not None else 3
        args.heads = args.heads if args.heads is not None else 8
        args.ff_dropout = args.ff_dropout if args.ff_dropout is not None else 0.1

    print("Loading dataset...")
    if args.dataset_type == 'train':
        features = torch.load('train_features.pt')
        labels = torch.load('train_labels.pt')
    elif args.dataset_type == 'validation':
        features = torch.load('validation_features.pt')
        labels = torch.load('validation_labels.pt')

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

    print("Loading model...")
    model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous)
    print("Model loaded.")

    # # Using multiple GPUs if available
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    model = model.to(device)  # Move the model to the appropriate device
    print("Computing attention maps...")
    attention_maps, feature_names = get_attention_maps(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns, device)
    print("Attention maps computed.")
    
    if attention_maps is not None and feature_names is not None:
        save_attention_maps_to_html(attention_maps, feature_names, "attention_maps.html")

        print("Identifying top feature values...")
        top_feature_value_attention_weights = identify_top_feature_values(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, feature_names[:10], device)
        print("Feature value attention weights:", top_feature_value_attention_weights)
        save_top_feature_values_to_html(top_feature_value_attention_weights, "top_feature_values.html")
        print("Top feature values identified.")
    else:
        print("No attention maps to process.")


if __name__ == '__main__':
    main()