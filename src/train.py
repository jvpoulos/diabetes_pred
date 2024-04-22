import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
from model_utils import load_model_weights, extract_epoch, load_performance_history, apply_cutmix_numerical, apply_mixup_numerical, CustomDataset, worker_init_fn, plot_auroc, plot_losses, train_model, validate_model, ResNetPrediction

def main(args):
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Set deterministic and benchmark flags for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load datasets
    train_dataset = torch.load('train_dataset.pt')
    validation_dataset = torch.load('validation_dataset.pt')

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    with open('columns_to_normalize.json', 'r') as file:
        columns_to_normalize = json.load(file)

    # Define excluded columns and additional binary variables
    excluded_columns = ["A1cGreaterThan7", "studyID"]
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

    with open('binary_feature_indices.json', 'w') as file:
        json.dump(binary_feature_indices, file)

    with open('numerical_feature_indices.json', 'w') as file:
        json.dump(numerical_feature_indices, file)

    # Assuming dataset is a TensorDataset containing a single tensor with both features and labels
    dataset_tensor = train_dataset.tensors[0]  # This gets the tensor from the dataset

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns_indices]

    label_index = column_names.index('A1cGreaterThan7')
 
    # Selecting features and labels
    train_features = dataset_tensor[:, feature_indices]
    train_labels = dataset_tensor[:, label_index]

    # Repeat for validation dataset if necessary
    validation_dataset_tensor = validation_dataset.tensors[0]
    validation_features = validation_dataset_tensor[:, feature_indices]
    validation_labels = validation_dataset_tensor[:, label_index]

    # Save validation and training features to file (for attention.py)
    torch.save(train_features, 'train_features.pt')
    torch.save(train_labels, 'train_labels.pt')

    torch.save(validation_features, 'validation_features.pt')
    torch.save(validation_labels, 'validation_labels.pt')

    print(f"Total columns in dataset: {len(column_names)}")
    print(f"Excluded columns: {excluded_columns}")
    print(f"Total feature indices: {len(feature_indices)}")
    print(f"Total binary feature indices: {len(binary_feature_indices)}")
    print(f"Total numerical feature indices: {len(numerical_feature_indices)}") 

    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {validation_features.shape}")
    print(f"Validation labels shape: {validation_labels.shape}")

    # Create custom datasets
    train_data = CustomDataset(train_features, train_labels)
    validation_data = CustomDataset(validation_features, validation_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    categories = [2] * len(binary_feature_indices)
    print("Categories length:", len(categories))
    print("Features length:", len(feature_indices))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, criterion, optimizer here with the current set of hyperparameters
    if args.model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,  # tuple containing the number of unique values within each category
            num_continuous=len(numerical_feature_indices),              # number of continuous values
            dim=args.dim,                                               # dimension, paper set at 32
            dim_out=1,                                                  # binary prediction, but could be anything
            depth=args.depth,                                           # depth, paper recommended 6
            heads=args.heads,                                           # heads, paper recommends 8
            attn_dropout=args.attn_dropout,                             # post-attention dropout
            ff_dropout=args.ff_dropout,                                 # feed forward dropout
            mlp_hidden_mults=(4,2),                                     # relative multiples of each hidden dimension of the last mlp to logits; paper recommends (4, 2)
            mlp_act=nn.ReLU(),                                          # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            dim_head = 16,                                              # default is 16
            shared_categ_dim_divisor = 8,                               # in paper, they reserve dimension / 8 for category shared embedding
            use_shared_categ_embed = True,                              # default is True
            checkpoint_grads=True,                                      # enable gradient checkpointing
            use_flash_attn=False                                         # use flash attention instead of standard softmax attention
        ).to(device)
    elif args.model_type == 'FTTransformer':
        model = FTTransformer(
            categories=categories,                                      # tuple containing the number of unique values within each category
            num_continuous=len(numerical_feature_indices),              # number of continuous values
            dim=args.dim,                                               # dimension, paper set at 192
            dim_out=1,                                                  # binary prediction, but could be anything
            dim_head = 16,                                              # default is 16
            depth=args.depth,                                           # depth, paper recommended 3
            heads=args.heads,                                           # heads, paper recommends 8
            attn_dropout=args.attn_dropout,                             # post-attention dropout, paper recommends 0.2
            ff_dropout=args.ff_dropout,                                 # feed forward dropout, paper recommends 0.1
            checkpoint_grads=False,                                     # enable gradient checkpointing
            use_flash_attn=False                                        # use flash attention instead of standard softmax attention
        ).to(device)
    elif args.model_type == 'ResNet':
        input_dim = len(binary_feature_indices) + len(numerical_feature_indices)  # Calculate the input dimension
        model = ResNetPrediction(
            input_dim=input_dim,  # Pass the input dimension
            d=args.dim,
            d_hidden_factor=args.d_hidden_factor,
            n_layers=args.depth,
            dropout=args.dropout,
            normalization=args.normalization,
            activation=args.activation
        ).to(device)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    # Using multiple GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    epoch_counter = 0
    # Check if model_path is specified and exists
    if args.model_path and os.path.isfile(args.model_path):
        epoch_counter = extract_epoch(args.model_path)
        load_model_weights(model, args.model_path)
        print(f"Loaded model from {args.model_path} starting from epoch {epoch_counter}")
    else:
        print("Starting training from scratch.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 1e3)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    hyperparameters = {
    'model_type': args.model_type,
    'dim': args.dim,
    'dep': args.depth,
    'heads': args.heads,
    'ffdr': args.ff_dropout,
    'adr': args.attn_dropout,
    'bs': args.batch_size,
    'lr': args.learning_rate,
    'wd': args.weight_decay,
    'ep': epoch_counter,
    'es': args.disable_early_stopping,
    'esp': args.early_stopping_patience,
    'rs': args.random_seed,
    'cmp': args.cutmix_prob,
    'cml': args.cutmix_alpha,
    'um': 'true' if args.use_mixup else 'false',
    'ma': args.mixup_alpha,
    'uc': 'true' if args.use_cutmix else 'false',
    'cl': 'true' if args.clipping else 'false',
    'mn': args.max_norm,
    'ba': 'true' if args.use_batch_accumulation else 'false',
    'sch': args.scheduler if args.scheduler else 'none'
    }  

    # start a new wandb run to track this script
    if args.run_id:
        print("Resuming a wandb run.")
        wandb.init(id=args.run_id, project="diabetes_pred", resume="allow")

        # Restore the model checkpoint
        checkpoint = torch.load(wandb.restore(args.wandb_path).name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.use_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_counter = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        train_aurocs = checkpoint["train_aurocs"]
        val_losses = checkpoint["val_losses"]
        val_aurocs = checkpoint["val_aurocs"]
    else:
        print("Starting a new wandb run.")
        wandb.init(
            # set the wandb project where this run will be logged
            project="diabetes_pred",
            
            # track hyperparameters and run metadata
            config=hyperparameters
        )
    
    # Constructing the file name from hyperparameters
    hyperparameters_str = '_'.join([f"{key}-{str(value).replace('.', '_')}" for key, value in hyperparameters.items()])
    performance_file_name = f"training_performance_{hyperparameters_str}.pkl"

    # Initialize losses
    train_losses, train_aurocs, val_losses, val_aurocs = load_performance_history(performance_file_name)

    # Initialize early stopping parameters
    best_val_auroc = float('-inf')  # Initialize to negative infinity
    patience_counter = 0
        
    if args.disable_early_stopping:
        early_stopping_patience = float('inf')  # Effectively disable early stopping
    else:
        early_stopping_patience = args.early_stopping_patience  # Use the provided early stopping patience

    # Define the directory where model weights will be saved
    model_weights_dir = 'model_weights'
    # Ensure the directory exists
    os.makedirs(model_weights_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_auroc = train_model(model, train_loader, criterion, optimizer, device, args.model_type, args.use_cutmix, args.cutmix_prob, args.cutmix_alpha, args.use_mixup, args.mixup_alpha, binary_feature_indices, numerical_feature_indices, args.max_norm, args.use_batch_accumulation, args.clipping)
        val_loss, val_auroc = validate_model(model, validation_loader, criterion, device, args.model_type, binary_feature_indices, numerical_feature_indices)

        # Save losses for plotting
        train_losses.append(train_loss)
        train_aurocs.append(train_auroc)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)

        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss}, Training AUROC: {train_auroc}, Validation Loss: {val_loss}, Validation AUROC: {val_auroc}')

        # log metrics to wandb
        wandb.log({"train_loss": train_loss, "train_auroc": train_auroc, "val_loss": val_loss, "val_auroc": val_auroc})

        if args.scheduler == 'cosine':
            scheduler.step()
        elif args.scheduler == 'plateau':
            scheduler.step(val_auroc)

        # Save model weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            if args.model_type=='FTTransformer' or args.model_type=='TabTransformer':
                model_filename = f"{args.model_type}_dim{args.dim}_dep{args.depth}_heads{args.heads}_fdr{args.ff_dropout}_adr{args.attn_dropout}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_ep{epoch + 1}_es{args.disable_early_stopping}_esp{args.early_stopping_patience}_rs{args.random_seed}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_mn{args.max_norm}_uc{'true' if args.use_cutmix else 'false'}_cl{'true' if args.clipping else 'false'}_ba{'true' if args.use_batch_accumulation else 'false'}_sch{'true' if args.scheduler else 'false'}.pth"
            elif args.model_type=='ResNet':
                model_filename = f"{args.model_type}_dep{args.depth}_dr{args.dropout}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_ep{epoch + 1}_es{args.disable_early_stopping}_esp{args.early_stopping_patience}_rs{args.random_seed}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_mn{args.max_norm}_uc{'true' if args.use_cutmix else 'false'}_cl{'true' if args.clipping else 'false'}_ba{'true' if args.use_batch_accumulation else 'false'}_sch{'true' if args.scheduler else 'false'}.pth"

            model_filepath = os.path.join(model_weights_dir, model_filename)
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "train_aurocs": train_aurocs,
                "val_losses": val_losses,
                "val_aurocs": val_aurocs
            }

            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(checkpoint, model_filepath)
            wandb.save(model_filepath)
            
            print(f"Model checkpoint saved as {model_filepath}")

        # Check for early stopping conditions
        if not args.disable_early_stopping:
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                # Save the best model
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                print(f"No improvement in validation AUROC for {patience_counter} epochs.")
                if patience_counter >= args.early_stopping_patience and val_auroc <= val_aurocs[-2]:
                    print(f"Stopping early at epoch {epoch+1}")
                    break

    # Save the best model if early stopping was triggered
    if not args.disable_early_stopping and patience_counter >= args.early_stopping_patience:
        if args.model_type=='FTTransformer' or args.model_type=='TabTransformer':
            best_model_filename = f"{args.model_type}_dim{args.dim}_dep{args.depth}_heads{args.heads}_fdr{args.ff_dropout}_adr{args.attn_dropout}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_ep{epoch + 1}_es{args.disable_early_stopping}_esp{args.early_stopping_patience}_rs{args.random_seed}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_mn{args.max_norm}_uc{'true' if args.use_cutmix else 'false'}_cl{'true' if args.clipping else 'false'}_ba{'true' if args.use_batch_accumulation else 'false'}_sch{'true' if args.scheduler else 'false'}_best.pth"
        elif args.model_type=='ResNet':
            best_model_filename = f"{args.model_type}_dep{args.depth}_dr{args.dropout}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_ep{epoch + 1}_es{args.disable_early_stopping}_esp{args.early_stopping_patience}_rs{args.random_seed}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_mn{args.max_norm}_uc{'true' if args.use_cutmix else 'false'}_cl{'true' if args.clipping else 'false'}_ba{'true' if args.use_batch_accumulation else 'false'}_sch{'true' if args.scheduler else 'false'}_best.pth"

        best_model_filepath = os.path.join(model_weights_dir, best_model_filename)
        
        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": best_model_wts,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_aurocs": train_aurocs,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs
        }

        if scheduler is not None:
            best_checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(best_checkpoint, best_model_filepath)
        wandb.save(best_model_filepath)
        
        print(f"Best model saved as {best_model_filepath}")
    
    # After training, plot the losses and save them to file
    plot_losses(train_losses, val_losses, hyperparameters)
    plot_auroc(train_aurocs, val_aurocs, hyperparameters)

    losses_and_aurocs = {
    'train_losses': train_losses,
    'train_aurocs': train_aurocs,
    'val_losses': val_losses,
    'val_aurocs': val_aurocs
    }

    # Define the directory path
    dir_path = 'losses'
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Modify the performance_file_name to include the directory path
    performance_file_path = os.path.join(dir_path, performance_file_name)

    # Saving to the file within the 'losses' directory
    with open(performance_file_path, 'wb') as f:
        pickle.dump(losses_and_aurocs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an attention network.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Transformer','TabTransformer', 'FTTransformer', 'FTTransformerOG', 'ResNet'],
                        help='Type of the model to train')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default=None,
                    help='Optional path to the saved model file to load before training')
    parser.add_argument('--wandb_path', type=str, default=None,
                    help='Optional path to the saved model file to load before training')
    parser.add_argument('--run_id', type=str, default=None,
                    help='Optional Weights & Biases run ID.')
    parser.add_argument('--cutmix_prob', type=float, default=0.3, help='Probability to apply CutMix')
    parser.add_argument('--cutmix_alpha', type=float, default=10, help='Alpha value for the CutMix beta distribution. Higher values result in more mixing.')
    parser.add_argument('--use_mixup', action='store_true', help='Enable MixUp data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha value for the MixUp beta distribution. Higher values result in more mixing.')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix data augmentation')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for the Transformer model')
    parser.add_argument('--use_batch_accumulation', action='store_true', help='Enable batch accumulation')
    parser.add_argument('--clipping', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--scheduler', type=str, default=None, choices=['cosine', 'plateau'], help='Learning rate scheduler type')
    parser.add_argument('--max_norm', type=float, default=10, help='Clip gradient values to max_norm.')
    parser.add_argument('--d_hidden_factor', type=float, default=2.0, help='Hidden dimension factor for the ResNet model')
    parser.add_argument('--normalization', type=str, default='layernorm', choices=['batchnorm', 'layernorm'], help='Normalization type for the ResNet model')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function for the ResNet model')

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
    elif args.model_type == 'ResNet':
        if args.dim is None:
            args.dim = 256
        if args.dim is None:
            args.dropout = 0.5
        if args.depth is None:
            args.depth = 3
    main(args)
    wandb.finish()