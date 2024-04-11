import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
from tab_transformer_pytorch import TabTransformer, FTTransformer
import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from model_utils import load_model, CustomDataset
from torch.utils.checkpoint import checkpoint
import h5py

def get_attention_maps(model, loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns, model_type, batch_size, chunk_size=64):
    if os.path.exists('attention_data.h5'):
        print("Attention data already exists. Skipping attention map computation.")
        return None, None, None

    model.eval()

    aggregated_attention_weights = {}
    valid_feature_names = {}
    aggregated_cls_attention_weights = {}

    with h5py.File('attention_data.h5', 'w') as hdf_file:
        with torch.no_grad():
            with autocast():
                for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
                    features, _ = batch
                    x_categ = features[:, binary_feature_indices].to(dtype=torch.long)
                    x_cont = features[:, numerical_feature_indices]

                    for i in range(0, x_categ.size(0), chunk_size):
                        x_categ_chunk = x_categ[i:i+chunk_size]
                        x_cont_chunk = x_cont[i:i+chunk_size]

                        x_categ_chunk = x_categ_chunk.to(next(model.parameters()).device)
                        x_cont_chunk = x_cont_chunk.to(next(model.parameters()).device)

                        x_cont_chunk.requires_grad_(True)

                        def forward_pass(x_categ_chunk, x_cont_chunk):
                            _, attns_chunk = model(x_categ=x_categ_chunk, x_numer=x_cont_chunk, return_attn=True)
                            return attns_chunk

                        attns_chunk = checkpoint(forward_pass, x_categ_chunk, x_cont_chunk)

                        attns_chunk = attns_chunk if isinstance(attns_chunk, list) else [attns_chunk]

                        for head_idx, attn_chunk in enumerate(attns_chunk):
                            if attn_chunk.dim() == 5:
                                attn_chunk = attn_chunk.mean(dim=0)  # Average across the batch dimension

                            # Now attn_chunk should be of shape [num_heads, num_features, num_features]
                            num_heads = attn_chunk.size(0)
                            num_features = attn_chunk.size(1)

                            # Extract the attention map corresponding to the [CLS] token for each head
                            cls_attns_chunk = attn_chunk[:, 0, :]  # Shape: (num_heads, num_features)

                            # Normalize attention weights across features for each head
                            attn_chunk = attn_chunk / attn_chunk.sum(dim=-1, keepdim=True)

                            # Write attention weights and feature names to HDF5 file
                            hdf_file.create_dataset(f'batch_{batch_idx}_chunk_{i}_head_{head_idx}_attn', data=attn_chunk.cpu().numpy(), compression="gzip")
                            hdf_file.create_dataset(f'batch_{batch_idx}_chunk_{i}_head_{head_idx}_cls_attn', data=cls_attns_chunk.cpu().numpy(), compression="gzip")
                            hdf_file.create_dataset(f'batch_{batch_idx}_chunk_{i}_head_{head_idx}_feature_names', data=[col for col in column_names[:num_features] if col not in excluded_columns], dtype=h5py.special_dtype(vlen=str))

                            # Aggregate attention weights and feature names
                            if head_idx not in aggregated_attention_weights:
                                aggregated_attention_weights[head_idx] = []
                                valid_feature_names[head_idx] = []
                                aggregated_cls_attention_weights[head_idx] = []

                            aggregated_attention_weights[head_idx].append(attn_chunk.cpu().numpy())
                            valid_feature_names[head_idx].append([col for col in column_names[:num_features] if col not in excluded_columns])
                            aggregated_cls_attention_weights[head_idx].append(cls_attns_chunk.cpu().numpy())

                        # Clear the GPU cache and release memory after each chunk
                        del x_categ_chunk, x_cont_chunk, attns_chunk
                        torch.cuda.empty_cache()

    print("Attention map computation completed.")

    # Aggregate attention weights and feature names across chunks
    for head_idx in aggregated_attention_weights:
        aggregated_attention_weights[head_idx] = np.mean(aggregated_attention_weights[head_idx], axis=0)
        valid_feature_names[head_idx] = valid_feature_names[head_idx][0]  # Assuming feature names are the same across valid batches

    # Calculate the average attention map for each head
    for head_idx in aggregated_cls_attention_weights:
        aggregated_cls_attention_weights[head_idx] = np.mean(aggregated_cls_attention_weights[head_idx], axis=(0, 1))

    # Calculate the final feature importance distribution by averaging the attention maps across all heads
    feature_importance_distribution = np.mean(list(aggregated_cls_attention_weights.values()), axis=0)
    feature_importance_distribution /= feature_importance_distribution.sum()

    # Select the top 100 features based on the aggregated attention weights for each head
    top_attention_weights = {}
    top_feature_names = {}

    for head_idx in aggregated_attention_weights:
        top_indices = np.argsort(aggregated_attention_weights[head_idx])[::-1][:100]
        top_attention_weights[head_idx] = aggregated_attention_weights[head_idx][top_indices]
        top_feature_names[head_idx] = [valid_feature_names[head_idx][idx] for idx in top_indices]

    return top_attention_weights, top_feature_names, feature_importance_distribution

def save_attention_maps_to_html(attention_maps, feature_names, feature_importance_distribution, filename):
    if not feature_names:
        print("No feature names found. Skipping saving attention maps to HTML.")
        return

    for head_idx in attention_maps:
        if head_idx not in feature_names:
            print(f"Feature names not found for head {head_idx}. Skipping saving attention map for this head.")
            continue

        attention_data = list(zip(feature_names[head_idx], attention_maps[head_idx]))

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
        with open(f"{filename.split('.')[0]}_head_{head_idx}.html", 'w') as file:
            file.write(f'<h1>Normalized attention weights per feature across all patients, regardless of their outcome: Top 100 features (Attention Head {head_idx})</h1>')
            file.write(html_table)

        print(f"Attention maps for head {head_idx} saved to {filename.split('.')[0]}_head_{head_idx}.html")

    if 0 not in feature_names:
        print("Feature names not found for feature importance distribution. Skipping saving feature importance distribution to HTML.")
        return

    importance_data = list(zip(feature_names[0], feature_importance_distribution))
    df_importance = pd.DataFrame(importance_data, columns=['Feature', 'Importance'])
    df_importance = pd.merge(df_importance, df_train_summary[['Feature', 'Description']], on='Feature', how='left')
    df_importance = df_importance.sort_values(by='Importance', ascending=False)
    html_table_importance = df_importance.to_html(index=False)

    with open(f"{filename.split('.')[0]}_importance.html", 'w') as file:
        file.write(f'<h1>Feature Importance Distribution</h1>')
        file.write(html_table_importance)

def identify_top_feature_values(model, loader, binary_feature_indices, numerical_feature_indices, column_names, top_feature_names, chunk_size=64):
    model.eval()

    with h5py.File('feature_value_attention_data.h5', 'w') as hdf_file:
        print("Starting feature value attention weight computation...")
        with torch.no_grad():
            with autocast():
                for batch in tqdm(loader, desc="Processing batches"):
                    features, labels = batch
                    x_categ = features[:, binary_feature_indices].to(dtype=torch.long)
                    x_cont = features[:, numerical_feature_indices]

                    # Split the batch into smaller chunks
                    for i in range(0, x_categ.size(0), chunk_size):
                        x_categ_chunk = x_categ_chunk.to(next(model.parameters()).device)
                        x_cont_chunk = x_cont_chunk.to(next(model.parameters()).device)
                        labels_chunk = labels_chunk.to(next(model.parameters()).device)

                        # Set requires_grad=True for the continuous features
                        x_cont_chunk.requires_grad_(True)

                        if args.model_type == 'FTTransformer':
                            def forward_pass(x_categ_chunk, x_cont_chunk):
                                _, attns_chunk = model(x_categ=x_categ_chunk, x_numer=x_cont_chunk, return_attn=True)
                                return attns_chunk
                            attns_chunk = checkpoint(forward_pass, x_categ_chunk, x_cont_chunk)
                        elif args.model_type == 'TabTransformer':
                            def forward_pass(x_categ_chunk, x_cont_chunk):
                                _, attns_chunk = model(x_categ=x_categ_chunk, x_cont=x_cont_chunk, return_attn=True)
                                return attns_chunk
                            attns_chunk = checkpoint(forward_pass, x_categ_chunk, x_cont_chunk)
                        else:
                            raise ValueError(f"Unsupported model type: {args.model_type}")

                        attns_chunk = attns_chunk if isinstance(attns_chunk, list) else [attns_chunk]

                        for head_idx, attn_chunk in enumerate(attns_chunk):
                            # Reduce the size of the tensor by averaging across layers if they exist
                            while attn_chunk.dim() > 3:
                                attn_chunk = attn_chunk.mean(dim=1)

                            # Now attn_chunk should be of shape [chunk_size, num_heads, num_features, num_features]
                            chunk_size = attn_chunk.size(0)
                            num_heads = attn_chunk.size(1)
                            num_features = attn_chunk.size(2)

                            # Normalize attention weights across features for each sample and head
                            attn_chunk = attn_chunk / attn_chunk.sum(dim=-1, keepdim=True)

                            for feature_name in top_feature_names[head_idx][:10]:  # Consider only the top 10 features
                                if feature_name in column_names:
                                    feature_index = column_names.index(feature_name)
                                    if feature_index < features.size(1):
                                        feature_values = features[i:i+chunk_size, feature_index].cpu().numpy()

                                        for value, label, attention in zip(feature_values, labels_chunk.cpu().numpy(), attn_chunk[:, head_idx, feature_index, feature_index].cpu().numpy()):
                                            if label == 1:
                                                hdf_file.create_dataset(f'head_{head_idx}_chunk_{i}_feature_{feature_name}_high_value_{value}', data=attention, compression="gzip")
                                            else:
                                                hdf_file.create_dataset(f'head_{head_idx}_chunk_{i}_feature_{feature_name}_low_value_{value}', data=attention, compression="gzip")

                        # Clear the GPU cache and release memory after each chunk
                        del x_categ_chunk, x_cont_chunk, attns_chunk, labels_chunk
                        torch.cuda.empty_cache()

        print("Feature value attention weight computation completed.")
    
    # Compute average attention weights for each feature value and outcome
    for head_idx in feature_value_attention_weights:
        for feature_name in top_feature_names[head_idx][:10]:
            if feature_name in feature_value_attention_weights[head_idx]:
                for outcome in ['HbA1c ≥ 7.0%', 'HbA1c < 7.0%']:
                    if outcome in feature_value_attention_weights[head_idx][feature_name]:
                        for value in feature_value_attention_weights[head_idx][feature_name][outcome]:
                            feature_value_attention_weights[head_idx][feature_name][outcome][value] = np.mean(feature_value_attention_weights[head_idx][feature_name][outcome][value])
    
    return feature_value_attention_weights

def save_top_feature_values_to_html(top_feature_value_attention_weights, filename):
    for head_idx in top_feature_value_attention_weights:
        # Create a DataFrame for each outcome
        df_high = pd.DataFrame(columns=['Feature', 'Feature Value', 'Attention Weight'])
        df_low = pd.DataFrame(columns=['Feature', 'Feature Value', 'Attention Weight'])

        for feature_name, outcome_values in top_feature_value_attention_weights[head_idx].items():
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
        with open(f"{filename.split('.')[0]}_head_{head_idx}.html", 'w') as file:
            file.write(f'<h1>Normalized attention weights for each feature value within each outcome group (HbA1c >= 7.0% and HbA1c < 7.0%): Top 10 features (Attention Head {head_idx})</h1>')
            file.write('<h2>Patients with HbA1c >= 7.0%</h2>')
            file.write(html_table_high)
            file.write('<h2>Patients with HbA1c < 7.0%</h2>')
            file.write(html_table_low)

        print(f"Top feature values for head {head_idx} saved to {filename.split('.')[0]}_head_{head_idx}.html")
    
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    parser.add_argument('--pruning', action='store_true', help='Enable model pruning')
    parser.add_argument('--quantization', type=int, default=None, help='Quantization bit width (8)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Optional path to the saved model file to load before training')

    # Parse known arguments and ignore unknown arguments (like --local_rank)
    args, _ = parser.parse_known_args()

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
        
    with open('numerical_feature_indices.json', 'r') as file:
        numerical_feature_indices = json.load(file)

    with open('binary_feature_indices.json', 'r') as file:
        binary_feature_indices = json.load(file)

    # Define excluded columns and additional binary variables
    excluded_columns = ["A1cGreaterThan7", "A1cLessThan7",  "studyID"]

    categories = [2] * len(binary_feature_indices)
    print("Categories:", len(categories))

    num_continuous = len(numerical_feature_indices)
    print("Continuous:", num_continuous)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pruning and os.path.exists(args.model_path + '_pruned'):
        print("Loading pruned model...")
        model = load_model(args.model_type, args.model_path + '_pruned', args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)
    else:
        print("Loading model...")
        model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)

        if args.pruning:
            print("Model pruning.")
            # Specify the pruning percentage
            pruning_percentage = 0.4

            # Perform global magnitude-based pruning
            parameters_to_prune = [(module, 'weight') for module in model.modules() if isinstance(module, (nn.Linear, nn.Embedding))]
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=pruning_percentage)

            # Remove the pruning reparameterization
            for module, _ in parameters_to_prune:
                prune.remove(module, 'weight')
            
            # Save the pruned model
            if args.model_path is not None:
                pruned_model_path =  args.model_path + '_pruned'
                torch.save(model.state_dict(), pruned_model_path)

        if args.quantization is not None and device.type == 'cpu':
            print(f"Quantizing model to {args.quantization} bits.")
            model = quantize_dynamic(model, {nn.Linear, nn.Embedding}, dtype=torch.qint8)

        if args.pruning or (args.quantization is not None and device.type == 'cpu'):
            # Save the pruned and/or quantized model
            if args.model_path is not None:
                model_path_ext = '_pruned' if args.pruning else ''
                model_path_ext += f'_quantized_{args.quantization}' if args.quantization is not None else ''
                torch.save(model.state_dict(), args.model_path + model_path_ext)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print("Model loaded.")

    with autocast():
        if not os.path.exists('attention_data.h5'):
            print("Computing attention maps...")
            attention_maps, feature_names, feature_importance_distribution = get_attention_maps(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, excluded_columns, args.model_type, args.batch_size, chunk_size=1)
            print("Attention maps computed.")
        else:
            print("Attention data already exists. Loading from file.")
            with h5py.File('attention_data.h5', 'r') as hdf_file:
                attention_maps = {}
                feature_names = {}
                feature_importance_distribution = None

                for key in hdf_file.keys():
                    if key.endswith('_attn'):
                        batch_idx, chunk_idx, head_idx, _ = key.split('_')
                        head_idx = int(head_idx)

                        if head_idx not in attention_maps:
                            attention_maps[head_idx] = []

                        attention_maps[head_idx].append(hdf_file[key][:])

                    elif key.endswith('_feature_names'):
                        batch_idx, chunk_idx, head_idx, _ = key.split('_')
                        head_idx = int(head_idx)

                        if head_idx not in feature_names:
                            feature_names[head_idx] = []

                        feature_names[head_idx].append(hdf_file[key][:])

                for head_idx in attention_maps:
                    attention_maps[head_idx] = np.mean(attention_maps[head_idx], axis=0)
                    feature_names[head_idx] = feature_names[head_idx][0]  # Assuming feature names are the same across chunks

        if attention_maps is not None and feature_names is not None:
            save_attention_maps_to_html(attention_maps, feature_names, feature_importance_distribution, "attention_maps.html")

            print("Identifying top feature values...")
            top_feature_value_attention_weights = identify_top_feature_values(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, feature_names, chunk_size=1)
            print("Feature value attention weights:", top_feature_value_attention_weights)
            save_top_feature_values_to_html(top_feature_value_attention_weights, "top_feature_values.html")
            print("Top feature values identified.")
        else:
            if os.path.exists('attention_data.h5'):
                print("Attention data already exists. Loading from file.")
                try:
                    with h5py.File('attention_data.h5', 'r') as hdf_file:
                        attention_maps = {}
                        feature_names = {}
                        feature_importance_distribution = None

                        for key in hdf_file.keys():
                            if key.endswith('_attn'):
                                try:
                                    batch_idx, chunk_idx, head_idx, _ = key.split('_')
                                    head_idx = int(head_idx)
                                except ValueError:
                                    # Skip the key if it doesn't match the expected format
                                    continue

                                if head_idx not in attention_maps:
                                    attention_maps[head_idx] = []

                                attention_maps[head_idx].append(hdf_file[key][:])

                            elif key.endswith('_feature_names'):
                                try:
                                    batch_idx, chunk_idx, head_idx, _ = key.split('_')
                                    head_idx = int(head_idx)
                                except ValueError:
                                    # Skip the key if it doesn't match the expected format
                                    continue

                                if head_idx not in feature_names:
                                    feature_names[head_idx] = []

                                feature_names[head_idx].append(hdf_file[key][:])

                        for head_idx in attention_maps:
                            attention_maps[head_idx] = np.mean(attention_maps[head_idx], axis=0)
                            feature_names[head_idx] = feature_names[head_idx][0]  # Assuming feature names are the same across chunks

                except RuntimeError as e:
                    print(f"Error loading attention data from file: {str(e)}")
                    print("Please check the integrity of the attention_data.h5 file or delete it and recompute the attention maps.")
                    return

                if attention_maps is not None and feature_names is not None:
                    if feature_names:
                        save_attention_maps_to_html(attention_maps, feature_names, feature_importance_distribution, "attention_maps.html")
                    else:
                        print("No valid feature names found. Skipping saving attention maps and feature importance distribution to HTML.")

                    print("Identifying top feature values...")
                    top_feature_value_attention_weights = identify_top_feature_values(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, feature_names, chunk_size=1)
                    print("Feature value attention weights:", top_feature_value_attention_weights)
                    save_top_feature_values_to_html(top_feature_value_attention_weights, "top_feature_values.html")
                    print("Top feature values identified.")
                else:
                    print("No attention maps to process.")
            else:
                print("Attention data file does not exist. Skipping loading from file.")

if __name__ == '__main__':
    main()