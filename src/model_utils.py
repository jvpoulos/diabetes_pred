import argparse
import torch
import torch.nn as nn
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

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Since features and labels are already tensors, no conversion is needed
        return self.features[idx], self.labels[idx]

def load_model(model_type, model_path, dim, depth, heads, attn_dropout, ff_dropout, categories, num_continuous):

    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,  # tuple containing the number of unique values within each category
            num_continuous=num_continuous,                              # number of continuous values
            dim=dim,                                               # dimension, paper set at 32
            dim_out=1,                                                  # binary prediction, but could be anything
            depth=depth,                                           # depth, paper recommended 6
            heads=heads,                                           # heads, paper recommends 8
            attn_dropout=attn_dropout,                             # post-attention dropout
            ff_dropout=ff_dropout ,                                # feed forward dropout
            mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        )
    elif model_type == 'FTTransformer':
        model = FTTransformer(
            categories = categories,      # tuple containing the number of unique values within each category
            num_continuous = num_continuous,  # number of continuous values
            dim = dim,                     # dimension, paper set at 192
            dim_out = 1,                        # binary prediction, but could be anything
            depth = depth,                          # depth, paper recommended 3
            heads = heads,                          # heads, paper recommends 8
            attn_dropout = attn_dropout,   # post-attention dropout, paper recommends 0.2
            ff_dropout = ff_dropout                   # feed forward dropout, paper recommends 0.1
        )
    elif args.model_type == 'Transformer':
        model = torch.nn.Transformer(
            d_model=dim,                                           # embedding dimension
            nhead=heads,                                           # number of attention heads
            num_encoder_layers=6,                                       # number of encoder layers
            num_decoder_layers=6,                                       # number of decoder layers
            dim_feedforward=2048,                                       # dimension of the feedforward network
            dropout=0.1,                                                # dropout rate
            activation=torch.nn.ReLU(),                                 # activation function
            custom_encoder=None,                                        # custom encoder
            custom_decoder=None,                                        # custom decoder
            layer_norm_eps=1e-05,                                       # layer normalization epsilon
            batch_first=False,                                          # if True, input and output tensors are provided as (batch, seq, feature)
            norm_first=False,                                           # if True, layer normalization is done before self-attention
            device=device,                                              # device to run the model on
            dtype=None                                                  # data type
        )
    else:
        raise ValueError("Invalid model type. Choose 'Transformer', 'TabTransformer' or 'FTTransformer'.")

    # Load the state dict with fix_state_dict applied
    state_dict = torch.load(model_path)
    model.load_state_dict(fix_state_dict(state_dict))
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