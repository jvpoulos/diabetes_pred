import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging

logging.basicConfig(filename='test.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class CustomDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column):
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.label_column = label_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.loc[idx, self.feature_columns].to_numpy(dtype=float)
        label = self.dataframe.loc[idx, self.label_column]
        return torch.tensor(features, dtype=torch.float), torch.tensor(label, dtype=torch.float)

def load_model(model_type, model_path):
    # Load encoded feature names
    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    categories=[2 for _ in range(len(encoded_feature_names))]

    # Initialize the correct model based on model_type
    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=len(numerical_feature_indices),
            dim=32,
            dim_out=1,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU()
        )
    elif model_type == 'FTTransformer':
        model = FTTransformer(
            categories = categories,
            num_continuous = len(numerical_feature_indices),
            dim = 192,
            dim_out = 1,
            depth = 3,
            heads = 8,
            attn_dropout = 0.2,
            ff_dropout = 0.1
        )
    else:
        raise ValueError("Invalid model type. Choose 'TabTransformer' or 'FTTransformer'.")

    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_true = []  # List to store all true labels
    y_scores = []  # List to store all model output scores for AUC computation

    true_labels = []
    predictions = []

    with torch.no_grad():  # No gradients to track
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            # Get the output scores (probabilities) instead of binary predictions
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
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

    # Assuming test_dataset is a tensor, use torch.index_select
    test_features = torch.index_select(test_dataset, 1, torch.tensor(feature_indices))
    test_labels = test_dataset[:, column_names.index("A1cGreaterThan7")]

    # Create custom datasets
    test_data = CustomDataset(test_features, test_labels)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = load_model(args.model_type, args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an attention network on the test set.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to evaluate: TabTransformer or FTTransformer')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file (trained_model.pth)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')

    args = parser.parse_args()
    main(args)