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
from model_utils import load_model, CustomDataset
import pickle

import os
import pickle

def get_attention_maps(model, loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns):
    model.eval()

    attention_weights_dir = "attention_weights"
    os.makedirs(attention_weights_dir, exist_ok=True)

    feature_names = []

    print("Starting attention map computation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
            features, _ = batch
            x_categ = features[:, binary_feature_indices].to(dtype=torch.long)  # dtype is specified here
            x_cont = features[:, numerical_feature_indices]

            # Move the input tensors to the same device as the model
            x_categ = x_categ.to(next(model.parameters()).device)
            x_cont = x_cont.to(next(model.parameters()).device)

            # Pass the input tensors to the model
            _, attns = model(x_categ=x_categ, x_numer=x_cont, return_attn=True)

            if isinstance(attns, tuple):
                attns = list(attns)
            elif isinstance(attns, torch.Tensor):
                attns = [attns]

            for attn in attns:
                if attn.dim() == 5:
                    attn = attn.mean(dim=0)  # Average across the batch dimension

                # Reduce the size of the tensor by averaging across heads and layers if they exist
                while attn.dim() > 3:
                    attn = attn.mean(dim=0)  # Average across heads or layers

                # Now attn should be of shape [num_features, num_features]
                num_features = attn.size(0)

                # Normalize attention weights across features
                attn = attn / attn.sum(dim=-1, keepdim=True)

                # Save the attention weights to disk
                attention_weights_file = os.path.join(attention_weights_dir, f"attention_weights_{batch_idx}.pkl")
                with open(attention_weights_file, "wb") as f:
                    pickle.dump(attn.cpu().numpy(), f)

                feature_names.append([col for col in column_names[:num_features] if col not in excluded_columns])

            # Clear the GPU cache and release memory after each batch
            del x_categ, x_cont, attns
            torch.cuda.empty_cache()

    print("Attention map computation completed.")

    # Check if feature_names is empty
    if len(feature_names) == 0:
        print("Warning: No feature names computed.")
        return None, None

    print(f"Number of attention weight files: {len(os.listdir(attention_weights_dir))}")
    print(f"Number of feature name lists: {len(feature_names)}")

    # Load the attention weights from disk and aggregate them
    attention_weights_files = sorted(os.listdir(attention_weights_dir))
    aggregated_attention_weights = []
    valid_feature_names = []

    for file in attention_weights_files:
        with open(os.path.join(attention_weights_dir, file), "rb") as f:
            attention_weights = pickle.load(f)
            if len(aggregated_attention_weights) > 0 and attention_weights.shape != aggregated_attention_weights[0].shape:
                print(f"Skipping attention weights from file {file} due to shape mismatch.")
                continue
            aggregated_attention_weights.append(attention_weights)
            valid_feature_names.append(feature_names[int(file.split("_")[-1].split(".")[0])])

    if len(aggregated_attention_weights) == 0:
        print("Warning: No valid attention weights found.")
        return None, None

    aggregated_attention_weights = np.mean(aggregated_attention_weights, axis=0)
    aggregated_feature_names = valid_feature_names[0]  # Assuming feature names are the same across valid batches

    # Select the top 100 features based on the aggregated attention weights
    top_indices = np.argsort(aggregated_attention_weights)[::-1][:100]
    top_attention_weights = aggregated_attention_weights[top_indices]
    top_feature_names = [aggregated_feature_names[idx] for idx in top_indices]

    return top_attention_weights, top_feature_names

def identify_top_feature_values(model, loader, binary_feature_indices, numerical_feature_indices, column_names, top_feature_names):
    model.eval()
    
    feature_value_attention_weights = {feature: {} for feature in top_feature_names}
    
    print("Starting feature value attention weight computation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            features, labels = batch
            x_categ = features[:, binary_feature_indices].to(dtype=torch.long)  # dtype is specified here
            x_cont = features[:, numerical_feature_indices]
            
            _, attns = model(x_categ=x_categ, x_numer=x_cont, return_attn=True)
            
            attns = attns if isinstance(attns, list) else [attns]
            
            for attn in attns:
                # Reduce the size of the tensor by averaging across heads and layers if they exist
                while attn.dim() > 3:
                    attn = attn.mean(dim=1)
                
                # Now attn should be of shape [batch_size, num_features, num_features]
                batch_size = attn.size(0)
                num_features = attn.size(1)
                
                # Normalize attention weights across features for each sample
                attn = attn / attn.sum(dim=-1, keepdim=True)
                
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
            
            # Clear the GPU cache and release memory after each batch
            del x_categ, x_cont, attns
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
        file.write('<h1>Normalized attention weights per feature across all patients, regardless of their outcome: Top 100 features</h1>')
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
        file.write('<h1>Normalized attention weights for each feature value within each outcome group (HbA1c >= 7.0% and HbA1c < 7.0%): Top 10 features</h1>')
        file.write('<h2>Patients with HbA1c >= 7.0%</h2>')
        file.write(html_table_high)
        file.write('<h2>Patients with HbA1c < 7.0%</h2>')
        file.write(html_table_low)

    print(f"Top feature values saved to {filename}")
    
def main():
    parser = argparse.ArgumentParser(description='Extract attention maps.')
    parser.add_argument('--dataset_type', type=str, choices=['train','validation'], required=True, help='Specify dataset type for evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Transformer','TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--num_encoder_layers', type=float, default=6, help='Number of sub-encoder-layers in the encoder')
    parser.add_argument('--num_decoder_layers', type=float, default=6, help=' Number of sub-decoder-layers in the decoder')
    parser.add_argument('--dim_feedforward', type=float, default=2048, help='Dimension of the feedforward network model ')
    parser.add_argument('--dropout', type=float, default=0.1, help='Attention dropout rate')
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
    elif args.model_type == 'Transformer':
        if args.heads is None:
            raise ValueError("The 'heads' argument must be provided when using the 'Transformer' model type.")
        if args.dtype is None:
            args.dtype = torch.float32  # Set a default value for dtype
        if args.dim is None:
            args.dim = 512
        args.heads = args.heads if args.heads is not None else 8  # Set a default value of 8 if args.heads is None

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
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

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

    print("Loading model...")
    model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous)
    print("Model loaded.")

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the appropriate device

    # Using multiple GPUs if available, wrapping the model just once here
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    print("Computing attention maps...")
    attention_maps, feature_names = get_attention_maps(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns)
    print("Attention maps computed.")
    
    if attention_maps is not None and feature_names is not None:
        save_attention_maps_to_html(attention_maps, feature_names, "attention_maps.html")

        print("Identifying top feature values...")
        top_feature_value_attention_weights = identify_top_feature_values(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, feature_names[:10])
        print("Feature value attention weights:", top_feature_value_attention_weights)
        save_top_feature_values_to_html(top_feature_value_attention_weights, "top_feature_values.html")
        print("Top feature values identified.")
    else:
        print("No attention maps to process.")

if __name__ == '__main__':
    main()