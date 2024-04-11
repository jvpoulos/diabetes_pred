import argparse
import os
import torch
from transformers import utils
from bertviz import head_view, model_view
from model_utils import load_model
from tab_transformer_pytorch import TabTransformer, FTTransformer
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize attention from a loaded model using BertViz.')
parser.add_argument('--dataset_type', type=str, choices=['train', 'validation'], required=True, help='Specify dataset type for visualization')
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
parser.add_argument('--model_type', type=str, required=True, choices=['Transformer', 'TabTransformer', 'FTTransformer'], help='Type of the model')
parser.add_argument('--dim', type=int, required=True, help='Dimension of the model')
parser.add_argument('--depth', type=int, required=True, help='Depth of the model')
parser.add_argument('--heads', type=int, required=True, help='Number of attention heads')
parser.add_argument('--attn_dropout', type=float, required=True, help='Attention dropout rate')
parser.add_argument('--ff_dropout', type=float, required=True, help='Feed forward dropout rate')
args = parser.parse_args()

# Load features and labels based on the dataset type
if args.dataset_type == 'train':
    features = torch.load('train_features.pt')
    labels = torch.load('train_labels.pt')
elif args.dataset_type == 'validation':
    features = torch.load('validation_features.pt')
    labels = torch.load('validation_labels.pt')

# Feature indices

with open('numerical_feature_indices.json', 'r') as file:
    numerical_feature_indices = json.load(file)

with open('binary_feature_indices.json', 'r') as file:
    binary_feature_indices = json.load(file)

categories = [2] * len(binary_feature_indices)
print("Categories:", len(categories))

num_continuous = len(numerical_feature_indices)
print("Continuous:", num_continuous)

# Load the model
model_path = args.model_path
model_type = args.model_type
dim = args.dim
depth = args.depth
heads = args.heads
attn_dropout = args.attn_dropout
ff_dropout = args.ff_dropout
binary_feature_indices = binary_feature_indices
numerical_feature_indices = numerical_feature_indices
categories = [2] * len(binary_feature_indices)
num_continuous = len(numerical_feature_indices)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
model = load_model(model_type, model_path, dim, depth, heads, attn_dropout, ff_dropout, categories, num_continuous, device)
print("Model loaded.")

# Suppress standard warnings
utils.logging.set_verbosity_error()

# Prepare input
input_features = torch.tensor(features[0].tolist()).unsqueeze(0)  # Select the first sample from the dataset
input_binary = input_features[:, binary_feature_indices].to(dtype=torch.long)
input_continuous = input_features[:, numerical_feature_indices]

# Generate attention maps
with torch.no_grad():
    if model_type == 'FTTransformer':
        outputs = model(x_categ=input_binary, x_numer=input_continuous, return_attn=True)
    elif model_type == 'TabTransformer':
        outputs = model(x_categ=input_binary, x_cont=input_continuous, return_attn=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    attention = outputs[-1]  # Output includes attention weights when return_attn=True
    
    # Check the dimensions of the attention tensor
    if isinstance(attention, tuple):
        attention = list(attention)
    elif isinstance(attention, torch.Tensor):
        if attention.dim() == 5:
            # Reshape the attention tensor from [num_layers, batch_size, num_heads, seq_len, seq_len]
            # to a list of [batch_size, num_heads, seq_len, seq_len] tensors
            attention = [attn.squeeze(0) for attn in torch.unbind(attention, dim=0)]
        else:
            raise ValueError(f"Unexpected attention tensor shape: {attention.shape}")
    else:
        raise ValueError(f"Unexpected attention type: {type(attention)}")

# Create output directory if it doesn't exist
output_dir = 'attn_html'
os.makedirs(output_dir, exist_ok=True)

# Generate head view HTML
tokens = [f"Feature {i}" for i in range(input_features.size(1))]  # Placeholder tokens
html_head_view = head_view(attention, tokens, html_action='return')
with open(os.path.join(output_dir, "head_view.html"), 'w') as file:
    file.write(html_head_view.data)

# Generate model view HTML
html_model_view = model_view(attention, tokens, html_action='return')
with open(os.path.join(output_dir, "model_view.html"), 'w') as file:
    file.write(html_model_view.data)