import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging
from model_utils import load_model, CustomDataset

def evaluate_model(model, test_loader):
    model.eval()

    true_labels = []
    predictions = []

    # Wrap the test loader with tqdm for a progress bar
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            categorical_features = features[:, binary_feature_indices].to(device)
            categorical_features = categorical_features.long()
            numerical_features = features[:, numerical_feature_indices].to(device)
            labels = labels.squeeze()  # Adjust labels shape if necessary
            labels = labels.to(device)  # Move labels to the device
            if model_type == 'Transformer':
                src = torch.cat((categorical_features, numerical_features), dim=-1)
                outputs = model(src)
            else:
                outputs = model(categorical_features, numerical_features)
                outputs = outputs.squeeze()  # Squeeze the output tensor to remove the singleton dimension

            # Accumulate true labels and predictions for AUROC calculation
            true_labels.extend(labels.cpu().squeeze().numpy())
            predictions.extend(outputs.detach().cpu().squeeze().numpy())

    auroc = roc_auc_score(true_labels, predictions)
    print(f'Test AUROC: {auroc:.4f}')

    return auroc

def main(args):
    # Load the test dataset
    test_dataset = torch.load('test_dataset.pt')

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    with open('columns_to_normalize.json', 'r') as file:
        columns_to_normalize = json.load(file)

    # Define excluded columns and additional binary variables
    excluded_columns = ["A1cGreaterThan7",  "studyID"]
    additional_binary_vars = ["Female", "Married", "GovIns", "English", "Veteran"]

    # Find indices of excluded columns
    excluded_columns_indices = [column_names.index(col) for col in excluded_columns]

    # Filter out excluded columns
    column_names_filtered = [col for col in column_names if col not in excluded_columns]
    encoded_feature_names_filtered = [name for name in encoded_feature_names if name in column_names_filtered]

    # Combine and deduplicate encoded and additional binary variables
    binary_features_combined = list(set(encoded_feature_names_filtered + additional_binary_vars))

    # Calculate binary feature indices, ensuring they're within the valid range
    binary_feature_indices = [column_names_filtered.index(col) for col in binary_features_combined if col in column_names_filtered]

    # Find indices of the continuous features
    numerical_feature_indices = [column_names.index(col) for col in columns_to_normalize if col not in excluded_columns]

    # Assuming dataset is a TensorDataset containing a single tensor with both features and labels
    dataset_tensor = test_dataset.tensors[0]  # This gets the tensor from the dataset

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns_indices]

    label_index = column_names.index('A1cGreaterThan7')

    # Assuming test_dataset is a tensor, use torch.index_select
    test_features = dataset_tensor[:, feature_indices]
    test_labels = dataset_tensor[:, label_index]

    # Create custom datasets
    test_data = CustomDataset(test_features, test_labels)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    categories = [2] * len(binary_feature_indices)
    print("Categories length:", len(categories))
    print("Features length:", len(feature_indices))

    num_continuous =len(numerical_feature_indices)
    print("Continuous length:", len(num_continuous))

    # Load the model
    model = load_model(args.model_type, args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Using multiple GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Evaluate the model
    print("Loading model...")
    model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous, device)
    print("Model loaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an attention network on the test set.')
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

    main(args)