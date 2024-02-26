import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import logging

logging.basicConfig(filename='attention.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_model(model_type, model_path, categories, num_continuous):

    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=11,
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
            num_continuous = 11,
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
    model.eval()
    return model

def get_attention_maps(model, loader):
    attention_maps = []
    for batch in loader:
        x_categ, x_cont = batch
        x_categ, x_cont = x_categ.to(model.device), x_cont.to(model.device)
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
    parser.add_argument('model_type', type=str, choices=['TabTransformer', 'FTTransformer'], help='Type of the model to load')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    # Load dataset and create DataLoader here as `data_loader`
    validation_features = torch.load('validation_features.pt')
    validation_labels = torch.load('validation_labels.pt')

    # Create a TensorDataset for the validation set
    validation_dataset = TensorDataset(validation_features, validation_labels)

    # Create the DataLoader for the validation dataset
    data_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)  # Adjust batch_size as needed

    # Load encoded feature names
    with open('encoded_feature_names_filtered.json', 'r') as file:
        encoded_feature_names = json.load(file)

    num_continuous = 12  # replace with actual number of continuous features
    categories=[2 for _ in range(len(encoded_feature_names))]
    
    model = load_model(args.model_type, args.model_path, categories, num_continuous)
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    attention_maps = get_attention_maps(model, data_loader)
    plot_attention_maps(attention_maps, args.model_type, args.model_path)

if __name__ == '__main__':
    main()