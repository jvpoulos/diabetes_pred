import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import copy
import logging
import json
import torchvision
import torchvision.transforms
import wandb
import random
import numpy as np
from tqdm import tqdm
import re
import pickle
import math
from einops import rearrange, repeat
from torch import Tensor
import typing as ty

def train_model(model, train_loader, criterion, optimizer, device, model_type, use_cutmix, cutmix_prob, cutmix_alpha, use_mixup, mixup_alpha, binary_feature_indices, numerical_feature_indices, clipping, max_norm, use_batch_accumulation, accum_iter=4):
    model.train()
    total_loss = 0

    true_labels = []
    predictions = []

    for batch_idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
       # Extracting categorical and numerical features based on their indices
        categorical_features = features[:, binary_feature_indices].to(device).long()  # Convert to long data type
        numerical_features = features[:, numerical_feature_indices].to(device)
        labels = labels.to(device)

        # Ensure variables are initialized
        lam = 1.0  # Default value for lam
        labels_a = labels.clone()  # Default values for labels_a and labels_b
        labels_b = labels.clone()

        # Initialize augmented_cat and augmented_num before the if-else blocks
        augmented_cat, augmented_num = None, None

        # Apply mixup or cutmix augmentation if enabled
        if use_mixup and np.random.rand() < mixup_alpha:
            combined_features = torch.cat((categorical_features, numerical_features), dim=1)
            augmented_data, labels_a, labels_b, lam = apply_mixup_numerical(combined_features, labels, mixup_alpha)
            augmented_cat = augmented_data[:, :len(binary_feature_indices)].long()
            augmented_num = augmented_data[:, len(binary_feature_indices):]
        elif use_cutmix and np.random.rand() < cutmix_prob:
            augmented_data, labels_a, labels_b, lam = apply_cutmix_numerical(features, labels, cutmix_alpha)
            augmented_cat = augmented_data[:, :len(binary_feature_indices)].long()
            augmented_num = augmented_data[:, len(binary_feature_indices):]
        else:
            augmented_cat = categorical_features
            augmented_num = numerical_features

        # Forward pass through the model and calculate the loss
        if model_type == 'Transformer':
            outputs = model(augmented_cat, augmented_num)
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, labels_a.squeeze()) + (1 - lam) * criterion(outputs, labels_b.squeeze())
            else:
                loss = criterion(outputs, labels.squeeze())
        else:
            outputs = model(augmented_cat, augmented_num).squeeze()
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

        # Normalize loss to account for batch accumulation if enabled
        if use_batch_accumulation:
            loss = loss / accum_iter

        # Backward pass and gradient accumulation
        loss.backward()

        # Perform optimizer step and zero gradients after accumulating specified number of batches
        if not use_batch_accumulation or ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate loss
        if use_batch_accumulation:
            total_loss += loss.item() * features.size(0) * accum_iter  # Multiply by accum_iter to account for normalized loss
        else:
            total_loss += loss.item() * features.size(0)

        torch.cuda.empty_cache()

        # Accumulate true labels and predictions for AUROC calculation
        true_labels.extend(labels.cpu().squeeze().numpy())
        predictions.extend(outputs.detach().cpu().squeeze().numpy())

    # Calculate average loss
    average_loss = total_loss / len(train_loader.dataset)
    print(f'Average Training Loss: {average_loss:.4f}')

    # Calculate AUROC on the training set
    train_auroc = roc_auc_score(true_labels, predictions)
    print(f'Training AUROC: {train_auroc:.4f}')

    return average_loss, train_auroc

def validate_model(model, validation_loader, criterion, device, model_type, binary_feature_indices, numerical_feature_indices):
    model.eval()
    total_loss = 0

    true_labels = []
    predictions = []

    # Wrap the validation loader with tqdm for a progress bar
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(validation_loader), total=len(validation_loader), desc="Validating"):
            numerical_features = features[:, numerical_feature_indices].to(device)  # Numerical features
            categorical_features = features[:, binary_feature_indices].to(device).long()  # Categorical features (convert to long data type)
            labels = labels.to(device)  # Move labels to the device

            if model_type == 'Transformer':
                outputs = model(categorical_features, numerical_features)
                loss = criterion(outputs, labels.squeeze())
            else:
                outputs = model(categorical_features, numerical_features).squeeze()
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Accumulate true labels and predictions for AUROC calculation
            true_labels.extend(labels.cpu().squeeze().numpy())
            predictions.extend(outputs.detach().cpu().squeeze().numpy())

    average_loss = total_loss / len(validation_loader)
    print(f'Average Validation Loss: {average_loss:.4f}')

    auroc = roc_auc_score(true_labels, predictions)
    print(f'Validation AUROC: {auroc:.4f}')
    return average_loss, auroc

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerWithInputProjection(nn.Module):
    def __init__(self, categories, num_continuous, d_model, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, activation="gelu", device=None):
        super().__init__()
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.d_model = d_model
        self.device = device

        self.categorical_embedder = nn.Embedding(self.num_unique_categories, d_model) if self.num_unique_categories > 0 else None
        self.numerical_embedder = nn.Linear(num_continuous, d_model) if num_continuous > 0 else None
        self.categorical_layer_norm = nn.LayerNorm(d_model)
        self.numerical_layer_norm = nn.LayerNorm(d_model)
        self.position_embedding = nn.Embedding(512, d_model)  # Adjust the maximum sequence length as needed
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, categorical_features, numerical_features):
        batch_size = categorical_features.size(0)
        
        if self.categorical_embedder:
            categorical_embeddings = self.categorical_embedder(categorical_features)
            categorical_embeddings = self.categorical_layer_norm(categorical_embeddings)
        else:
            categorical_embeddings = torch.zeros(batch_size, categorical_features.size(1), self.d_model, device=self.device)

        if self.numerical_embedder:
            numerical_embeddings = self.numerical_embedder(numerical_features)
            numerical_embeddings = numerical_embeddings.unsqueeze(1)  # Add an extra dimension for sequence length
            numerical_embeddings = self.numerical_layer_norm(numerical_embeddings)
        else:
            numerical_embeddings = torch.zeros(batch_size, numerical_features.size(1), self.d_model, device=self.device)

        src = torch.cat([categorical_embeddings, numerical_embeddings], dim=1)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat([cls_tokens, src], dim=1)

        seq_length = src.size(1)

        # Add position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        src = src + position_embeddings

        src = self.dropout(src)
        src = src.transpose(0, 1)  # Shape: (seq_length, batch_size, d_model)

        output = self.transformer_encoder(src)

        cls_output = output[0]  # Extract the CLS token representation

        output = self.output_projection(cls_output)
        return torch.sigmoid(output.squeeze(-1))

def get_activation_fn(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

class ResNetBlock(nn.Module):
    def __init__(self, d: int, dropout: float, normalization: str, activation: str):
        super().__init__()
        self.norm = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[normalization](d)
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        z = self.dropout(self.activation(self.linear(z)))
        return x + z

class ResNetPrediction(nn.Module):
    def __init__(self, input_dim: int, d: int, d_hidden_factor: float, n_layers: int, dropout: float, normalization: str, activation: str):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d)  # Add an input layer to project the input to the desired dimension
        self.blocks = nn.Sequential(*[ResNetBlock(d, dropout, normalization, activation) for _ in range(n_layers)])
        self.linear = nn.Linear(d, int(d * d_hidden_factor))
        self.norm = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[normalization](int(d * d_hidden_factor))
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(d * d_hidden_factor), 1)

    def forward(self, categorical_features: Tensor, numerical_features: Tensor) -> Tensor:
        x = torch.cat([categorical_features, numerical_features], dim=1)
        x = self.input_layer(x)  # Project the input to the desired dimension
        x = self.blocks(x)
        x = self.dropout(self.activation(self.norm(self.linear(x))))
        x = self.head(x)
        return x.squeeze(-1)

class MLPBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))

class MLPPrediction(nn.Module):
    def __init__(self, d_in_num: int, d_in_cat: int, d_layers: ty.List[int], dropout: float, d_out: int, categories: ty.Optional[ty.List[int]], d_embedding: int, numerical_feature_indices: ty.List[int], binary_feature_indices: ty.List[int]):
        super().__init__()
        self.d_embedding = d_embedding
        self.category_embeddings = None

        if categories is not None:
            self.category_embeddings = nn.ModuleList([nn.Embedding(cat_size, d_embedding) for cat_size in categories])
            for embedding in self.category_embeddings:
                nn.init.kaiming_uniform_(embedding.weight, a=math.sqrt(5))
            d_in_cat = len(categories) * d_embedding  # Calculate the total embedding size

        d_in = d_in_num + d_in_cat
        layers = []

        # Add an input layer to match the dimension of d_in with the first hidden layer
        if d_layers:
            layers.append(nn.Linear(d_in, d_layers[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = d_layers[0]
        else:
            layers.append(nn.Flatten())  # Add a flatten layer if no input layer is needed
        
        for i in range(1, len(d_layers)):  # Iterate over the hidden layers
            layers.append(MLPBlock(d_layers[i-1], d_layers[i], dropout))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)  # Adjust the dimension of the head layer

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x_num = x_num.view(x_num.size(0), -1)  # Flatten numerical features
            x.append(x_num)
        if x_cat is not None and self.category_embeddings is not None:
            x_cat = x_cat.long()  # Convert x_cat to long data type
            x_cat_split = torch.split(x_cat, 1, dim=1)  # Split x_cat along the second dimension
            x_cat_emb = [embedding(split.squeeze(1)) for split, embedding in zip(x_cat_split, self.category_embeddings)]
            x_cat_emb = torch.cat(x_cat_emb, dim=1)  # Concatenate the embeddings
            x.append(x_cat_emb)

        x = torch.cat(x, dim=-1)

        for idx, layer in enumerate(self.layers):
            x = layer(x)

        x = self.head(x)
        x = x.squeeze(-1)
        return x
        
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def load_model(model_type, model_path, dim, depth, heads, attn_dropout, ff_dropout, categories, num_continuous, device, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, normalization='layernorm', activation='relu', d_hidden_factor=2.0, d_layers=None, d_embedding=None, checkpoint_grads=False):
    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            dim_out=1,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            mlp_hidden_mults=(2, 1),
            mlp_act=nn.ReLU(),
            dim_head = 16,                                              
            shared_categ_dim_divisor = 16,                               
            use_shared_categ_embed = True,                              
            checkpoint_grads=checkpoint_grads                                       
        )
    elif model_type == 'FTTransformer':
        model = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            dim_out=1,
            dim_head = 16,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            checkpoint_grads=checkpoint_grads
        )
    elif model_type == 'FTTransformerOG':
        model = FTTransformerOG(
            d_numerical=num_continuous,
            categories=categories,
            token_bias=True,
            n_layers=depth,
            d_token=dim,
            n_heads=heads,
            d_ffn_factor=1.0,
            attention_dropout=attn_dropout,
            ffn_dropout=ff_dropout,
            residual_dropout=0.0,
            activation='reglu',
            prenormalization=True,
            initialization='kaiming',
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=1
        )
    elif model_type == 'Transformer':
        model = TransformerWithInputProjection(
            categories=categories,
            num_continuous=len(numerical_feature_indices),
            d_model=dim,
            nhead=heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='geglu',
            device=device
        )
    elif model_type == 'ResNet':
        input_dim = len(binary_feature_indices) + len(numerical_feature_indices)  # Calculate the input dimension
        model = ResNetPrediction(
            input_dim=input_dim,  # Pass the input dimension
            d=dim,
            d_hidden_factor=d_hidden_factor,
            n_layers=depth,
            dropout=dropout,
            normalization=normalization,
            activation=activation
        )
    elif model_type == 'MLP':
        model = MLPPrediction(
            d_in_num=len(numerical_feature_indices),
            d_in_cat=len(binary_feature_indices) * args.d_embedding,
            d_layers=args.d_layers,
            dropout=args.dropout,
            d_out=1,
            categories=[2] * len(binary_feature_indices),
            d_embedding=args.d_embedding,
            numerical_feature_indices=numerical_feature_indices,
            binary_feature_indices=binary_feature_indices
        )
    else:
        raise ValueError("Invalid model type. Choose 'Transformer', 'TabTransformer', 'FTTransformer', 'ResNet', or 'MLP'.")

    if model_path is not None:
        # Load the model weights on the CPU first
        state_dict = torch.load(model_path, map_location='cpu')

        # Remove unexpected keys from the state dictionary
        unexpected_keys = ["module.epoch", "module.model_state_dict", "module.optimizer_state_dict", "module.scheduler_state_dict",
                           "module.train_losses", "module.train_aurocs", "module.val_losses", "module.val_aurocs"]
        for key in unexpected_keys:
            if key in state_dict:
                del state_dict[key]

        # Load the state dictionary while ignoring missing keys
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def plot_auroc(train_aurocs, val_aurocs, hyperparameters, plot_dir='auroc_plots'):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_aurocs, label='Training AUROC')
    plt.plot(val_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC Over Epochs')
    plt.legend()
    
    # Construct filename based on hyperparameters
    filename = '_'.join([f"{key}-{value}" for key, value in hyperparameters.items()]) + '_auroc.png'
    filepath = os.path.join(plot_dir, filename)
    
    plt.savefig(filepath)
    plt.close()

def plot_losses(train_losses, val_losses, hyperparameters, plot_dir='loss_plots'):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    
    # Construct filename based on hyperparameters
    filename = '_'.join([f"{key}-{value}" for key, value in hyperparameters.items()]) + '.png'
    filepath = os.path.join(plot_dir, filename)
    
    plt.savefig(filepath)
    plt.close()

def load_model_weights(model, saved_model_path):
    model_state_dict = model.state_dict()
    saved_state_dict = torch.load(saved_model_path)
    for name, param in saved_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping {name} due to size mismatch: model({model_state_dict[name].shape}) saved({param.shape})")
        else:
            print(f"Skipping {name} as it is not in the current model.")
    model.load_state_dict(model_state_dict)

def load_performance_history(performance_file_name):
    if os.path.exists(performance_file_name):
        with open(performance_file_name, 'rb') as f:
            history = pickle.load(f)
        return history.get('train_losses', []), history.get('train_aurocs', []), history.get('val_losses', []), history.get('val_aurocs', [])
    else:
        return [], [], [], []

def extract_epoch(filename):
    # Extracts epoch number from filename using a regular expression
    match = re.search(r'ep(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if not found

def apply_cutmix_numerical(data, labels, beta=1.0):
    lam = np.random.beta(beta, beta)
    batch_size = data.size(0)
    feature_count = data.size(1)
    index = torch.randperm(batch_size).to(data.device)

    # Determine the number of features to mix
    mix_feature_count = int(feature_count * lam)

    # Randomly choose the features to mix
    mix_features_indices = torch.randperm(feature_count)[:mix_feature_count]

    # Swap the chosen features for all examples in the batch
    for i in range(batch_size):
        data[i, mix_features_indices] = data[index[i], mix_features_indices]

    labels_a, labels_b = labels, labels[index]
    return data, labels_a, labels_b, lam

def apply_mixup_numerical(data, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(data.device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_data, labels_a, labels_b, lam