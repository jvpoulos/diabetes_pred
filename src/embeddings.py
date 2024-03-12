import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import pandas as pd
from tab_transformer_pytorch import TabTransformer, FTTransformer
print(FTTransformer)
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
from model_utils import load_model, CustomDataset

def extract_embeddings(model, loader, device):
    model = model.to(device)  # Ensure the model is on the correct device
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x_categ, x_cont = [t.to(device) for t in batch]  # Use the passed device directly
            emb = model.get_embeddings(x_categ, x_cont)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings).numpy()

def plot_embeddings(tsne_df, model_type, model_path):

    # Create the Seaborn plot
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        palette=sns.color_palette("hsv", 10),
        data=tsne_df,
        legend="full",
        alpha=0.3
    )
    plt.title(f'Learned embeddings - {model_type}')

    # Construct filename based on model_type and model_path
    filename = f"tSNE_embeddings_plot_{model_type}_{os.path.basename(model_path).replace('.pth', '')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Embeddings plot saved to {filename}")

def main():
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
    model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)
    print(f"Model created: {model}")

    model = model.to(device)  # Move the model to the appropriate device

    if hasattr(model, 'get_embeddings'):
        print("get_embeddings method exists in model.")
    else:
        print("get_embeddings method does not exist in model.")

    # Get embeddings
    embeddings = extract_embeddings(model, data_loader, device)

    # Apply t-SNE to the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Convert to DataFrame for Seaborn plotting
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

    plot_embeddings(tsne_df, args.model_type, args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings.')
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
    elif args.model_type == 'Transformer':
        if args.dim is None:
            args.dim = 512  # Default for FTTransformer
        args.heads = args.heads if args.heads is not None else 8
        # args.num_encoder_layers = 6
        # args.num_decoder_layers = 6
        # args.dim_feedforward = 2048
        # args.dropout = 0.1
        # args.activation = torch.nn.ReLU()
        # args.custom_encoder = None
        # args.custom_decoder = None
        # args.layer_norm_eps = 1e-05
        # args.batch_first = False
        # args.norm_first = False
        # args.bias = True
    main()