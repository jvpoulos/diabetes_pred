import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
import pandas as pd
from tab_transformer_pytorch import TabTransformer, FTTransformer
import logging
import os
import argparse
import json

logging.basicConfig(filename='embeddings.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_model(model_type, model_path, categories, num_continuous):

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
            categories = categories,
            num_continuous = 12,
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

def extract_embeddings(model, loader):
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x_categ, x_cont = [t.to(model.device) for t in batch]
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
    data_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # Load encoded feature names
    with open('encoded_feature_names_filtered.json', 'r') as file:
        encoded_feature_names = json.load(file)

    num_continuous = 12  # replace with actual number of continuous features
    categories=[2 for _ in range(len(encoded_feature_names))]
    
    model = load_model(args.model_type, args.model_path, categories, num_continuous)
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Get embeddings from the validation set
    validation_embeddings = extract_embeddings(model, data_loader)

    # Apply t-SNE to the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(validation_embeddings)

    # Convert to DataFrame for Seaborn plotting
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

    plot_embeddings(tsne_df, args.model_type, args.model_path)

if __name__ == '__main__':
    main()