import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def load_model(model_type, model_path):
    # Load encoded feature names
    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    categories = [2 for _ in encoded_feature_names]

    # Initialize the correct model based on model_type
    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=12,
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
            categories=categories,
            num_continuous=12,
            dim=32,
            dim_out=1,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1
        )
    else:
        raise ValueError("Invalid model type. Choose 'TabTransformer' or 'FTTransformer'.")

    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_pred, y_true = [], []

    with torch.no_grad():  # No gradients to track
        for data, labels in test_loader:
            outputs = model(data)
            predicted = torch.round(torch.sigmoid(outputs))  # Assuming model outputs raw scores
            y_pred.extend(predicted.view(-1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    return accuracy, precision, recall, f1

def main(args):
    # Load the test dataset
    test_dataset = torch.load('filtered_test_tensor.pt')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    model = load_model(args.model_type, args.model_path)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an attention network on the test set.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to evaluate: TabTransformer or FTTransformer')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file (best_model.pth)')

    args = parser.parse_args()
    main(args)