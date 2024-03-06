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

def save_attention_maps_to_html(attention_maps, feature_names, filename):
    # Assume attention_maps is a list of attention weights
    # And feature_names is a list of lists, where each inner list contains the top 100 feature names for each instance

    attention_data = []

    # Iterate over each instance's attention weights and feature names
    for instance_attention, instance_features in zip(attention_maps, feature_names):
        # Convert the attention weights to a list for easier processing
        attention_values = instance_attention.mean(dim=0).cpu().tolist()  # Take the mean across attention heads

        # Pair feature names with their corresponding attention weights
        instance_data = list(zip(instance_features, attention_values))

        attention_data.extend(instance_data)

    # Create a DataFrame to represent the attention data
    df = pd.DataFrame(attention_data, columns=['Feature', 'Attention Weight'])

    # Keep only the first occurrence of each feature
    df = df.drop_duplicates(subset='Feature', keep='first')

    # Sort the DataFrame by the attention weights in descending order
    df = df.sort_values(by='Attention Weight', ascending=False)

    # Select the top 100 features
    num_top_features = 100
    df = df.head(num_top_features)

    # Convert the DataFrame to an HTML table representation
    html_table = df.to_html(index=False)

    filename = f"attention_maps_{num_top_features}_attention_{model_type}_{os.path.basename(model_path).replace('.pth', '')}.html"

    # Save the HTML table to a file
    with open(filename, 'w') as file:
        file.write(html_table)

    print(f"Attention maps saved to {filename}")

def get_attention_maps(model, loader, binary_feature_indices, numerical_feature_indices, column_names, device):
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
                # Flatten it to [batch_size, num_features^2] to get individual attention weights
                attn = attn.view(attn.size(0), -1)
                
                # Get the top 100 attention weights
                top_vals, top_indices = torch.topk(attn, 100, dim=1, largest=True, sorted=True)
                
                # Move top indices to CPU if they are on GPU, then convert to numpy
                top_indices = top_indices.cpu().numpy()
                
                # Retrieve the feature names using numpy indexing
                for idx in top_indices:
                    # Make sure the indexing is within the length of column names
                    idx = [i if i < len(column_names) else len(column_names) - 1 for i in idx]
                    batch_feature_names.append([column_names[i] for i in idx])
                
                top_attention_weights.append(top_vals.cpu().numpy())
                feature_names.extend(batch_feature_names)
            
            # Optionally clear some cache if necessary
            torch.cuda.empty_cache()
    
    print("Attention map computation completed.")
    return top_attention_weights, feature_names

def plot_attention_maps(attention_maps, feature_names, model_type, model_path, instance_idx=0):
    # Get the attention weights and feature names for the selected instance
    attention_weights = attention_maps[instance_idx]
    top_feature_names = feature_names[instance_idx]

    # Convert attention_weights to a PyTorch tensor if it's a list
    if isinstance(attention_weights, list):
        attention_weights = torch.stack(attention_weights)

    # Average the attention weights across attention heads
    avg_attention_map = attention_weights.mean(dim=0)

    # Select the top 100 features or the maximum available features
    num_top_features = min(100, len(top_feature_names))
    avg_attention_map = avg_attention_map[:num_top_features]
    top_feature_names = top_feature_names[:num_top_features]

    plt.figure(figsize=(20, 8))  # Adjust the figure size as needed

    # Use the top feature names as y-ticks
    sns.heatmap(avg_attention_map.cpu().detach().numpy().reshape(-1, 1), cmap='viridis', cbar_kws={'orientation': 'vertical'}, yticklabels=top_feature_names)

    plt.title(f'Top {num_top_features} Attention Weights - {model_type}')
    plt.xlabel('Attention Weights')
    plt.ylabel('Top Features')
    plt.yticks(rotation=0)  # Keep the feature names horizontal for readability

    filename = f"labeled_top_{num_top_features}_attention_{model_type}_{os.path.basename(model_path).replace('.pth', '')}_instance_{instance_idx}.png"
    plt.tight_layout()  # Adjust layout to fit feature names
    plt.savefig(filename)
    plt.close()

    print(f"Labeled top {num_top_features} attention map for instance {instance_idx} saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['train','validation'], required=True, help='Specify dataset type for evaluation')
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
    model = load_model(args.model_type, args.model_path, args.dim, args.attn_dropout, categories, num_continuous)
    print("Model loaded.")

    # # Using multiple GPUs if available
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    model = model.to(device)  # Move the model to the appropriate device

    print("Computing attention maps...")
    attention_maps, feature_names = get_attention_maps(model, data_loader, binary_feature_indices, numerical_feature_indices, column_names, device)
    print("Attention maps computed.")

    save_attention_maps_to_html(attention_maps[-1], feature_names[-1])

    print("Plotting attention maps...")
    plot_attention_maps(attention_maps, feature_names, args.model_type, args.model_path)
    print("Attention maps plotted.")

if __name__ == '__main__':
    main()