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

logging.basicConfig(filename='test.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def evaluate_model(model, test_loader):
    y_true = []  # List to store all true labels
    y_scores = []  # List to store all model output scores for AUC computation

    with torch.no_grad():  # No gradients to track
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            # Get the output scores (probabilities) directly since sigmoid has already been applied
            scores = outputs.squeeze().cpu().numpy()
            y_scores.extend(scores)
            y_true.extend(labels.cpu().numpy())

    # Convert scores to binary predictions for accuracy, precision, recall, and f1 calculation
    predicted = np.round(y_scores)
    accuracy = np.mean(y_true == predicted)
    precision = precision_score(y_true, predicted)
    recall = recall_score(y_true, predicted)
    f1 = f1_score(y_true, predicted)

    # Compute AUC score
    auc_score = roc_auc_score(y_true, y_scores)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc_score:.4f}')  # Print AUC score

    return accuracy, precision, recall, f1, auc_score

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
    excluded_columns = ["A1cGreaterThan7", "A1cLessThan7",  "studyID"]
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

    label_index = column_names.index(args.outcome)

    # Assuming test_dataset is a tensor, use torch.index_select
    test_features = dataset_tensor[:, feature_indices]
    test_labels = dataset_tensor[:, label_index]

    # Create custom datasets
    test_data = CustomDataset(test_features, test_labels)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = load_model(args.model_type, args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    print("Loading model...")
    model = load_model(args.model_type, args.model_path, args.dim, args.depth, args.heads, args.attn_dropout, args.ff_dropout, categories, num_continuous)
    print("Model loaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an attention network on the test set.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Transformer','TabTransformer', 'FTTransformer'],
                        help='Type of the model to evaluate: TabTransformer or FTTransformer')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file (trained_model.pth)')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--outcome', type=str, required=True, choices=['A1cGreaterThan7', 'A1cLessThan7'], help='Outcome variable to predict')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')

    args = parser.parse_args()
    main(args)