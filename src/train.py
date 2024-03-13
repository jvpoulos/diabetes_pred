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
from model_utils import CustomDataset

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers for both file and console
file_handler = logging.FileHandler('train.log')
console_handler = logging.StreamHandler()

# Set logging level for both handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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

def train_model(model, train_loader, criterion, optimizer, device, use_cutmix, cutmix_prob, cutmix_alpha, use_mixup, mixup_alpha, binary_feature_indices, numerical_feature_indices):
    model.train()
    total_loss = 0

    true_labels = []
    predictions = []

    for batch_idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        optimizer.zero_grad()

        # Extracting categorical and numerical features based on their indices
        categorical_features = features[:, binary_feature_indices].to(device)
        categorical_features = categorical_features.long()  # Convert categorical_features to long type
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

        # Forward pass through the model
        outputs = model(augmented_cat, augmented_num).squeeze()

        # Calculate loss
        if use_mixup or use_cutmix:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * features.size(0)
        torch.cuda.empty_cache()

        # Accumulate true labels and predictions for AUROC calculation
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(outputs.detach().cpu().numpy())  # logits applied in model

    # Calculate average loss
    average_loss = total_loss / len(train_loader.dataset)
    print(f'Average Training Loss: {average_loss:.4f}')

    # Calculate AUROC on the training set
    train_auroc = roc_auc_score(true_labels, predictions)
    print(f'Training AUROC: {train_auroc:.4f}')

    return average_loss, train_auroc

def validate_model(model, validation_loader, criterion, device, binary_feature_indices, numerical_feature_indices):
    model.eval()
    total_loss = 0

    true_labels = []
    predictions = []

    # Wrap the validation loader with tqdm for a progress bar
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(validation_loader), total=len(validation_loader), desc="Validating"):
            categorical_features = features[:, binary_feature_indices].to(device)
            categorical_features = categorical_features.long()
            numerical_features = features[:, numerical_feature_indices].to(device)
            labels = labels.squeeze()  # Adjust labels shape if necessary
            labels = labels.to(device)  # Move labels to the device
            outputs = model(categorical_features, numerical_features)
            outputs = outputs.squeeze()  # Squeeze the output tensor to remove the singleton dimension
            loss = criterion(outputs, labels)  # Now both tensors have compatible shapes
            total_loss += loss.item()

            true_labels.extend(labels.cpu().numpy())  # Accumulate true labels
            predictions.extend(outputs.detach().cpu().numpy())  # logits applied in model  # Accumulate predictions (logits applied in model)

    average_loss = total_loss / len(validation_loader)
    print(f'Average Validation Loss: {average_loss:.4f}')
    auroc = roc_auc_score(true_labels, predictions)  # Calculate AUROC
    print(f'Validation AUROC: {auroc:.4f}')
    return average_loss, auroc

def main(args):
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
    dataset_tensor = train_dataset.tensors[0]  # This gets the tensor from the dataset

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns_indices]

    label_index = column_names.index(args.outcome)
 
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

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

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
            mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        ).to(device)
    elif args.model_type == 'FTTransformer':
        model = FTTransformer(
            categories=categories,                                      # tuple containing the number of unique values within each category
            num_continuous=len(numerical_feature_indices),              # number of continuous values
            dim=args.dim,                                               # dimension, paper set at 192
            dim_out=1,                                                  # binary prediction, but could be anything
            depth=args.depth,                                           # depth, paper recommended 3
            heads=args.heads,                                           # heads, paper recommends 8
            attn_dropout=args.attn_dropout,                             # post-attention dropout, paper recommends 0.2
            ff_dropout=args.ff_dropout                                  # feed forward dropout, paper recommends 0.1
        ).to(device)
    elif args.model_type == 'Transformer':
        model = torch.nn.Transformer(
            d_model=args.dim,                                           # embedding dimension
            nhead=args.heads,                                           # number of attention heads
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    hyperparameters = {
    'model_type': args.model_type,
    'dim': args.dim,
    'depth': args.depth,
    'heads': args.heads,
    'ff_dropout': args.ff_dropout,
    'attn_dropout': args.attn_dropout,
    'outcome': args.outcome,
    'batch_size': args.batch_size,
    'lr': args.learning_rate,
    'ep': epoch_counter,
    'esp': args.early_stopping_patience,
    'cutmix_prob': args.cutmix_prob,
    'cutmix_alpha': args.cutmix_alpha,
    'use_mixup': 'true' if args.use_mixup else 'false',
    'mixup_alpha': args.mixup_alpha,
    'use_cutmix': 'true' if args.use_cutmix else 'false'
    }   

    # start a new wandb run to track this script
    if args.run_id:
        print("Resuming a wandb run.")
        wandb.init(id=args.run_id, project="diabetes_pred", resume="allow")

        # Restore the model checkpoint
        checkpoint = torch.load(wandb.restore("model_checkpoint.tar").name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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

    # Add early stopping parameters
    best_val_auroc = float('-inf')  # Initialize to negative infinity
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience  # assuming this argument is added to the parser

    # CutMix and mixup parameters
    use_cutmix = args.use_cutmix
    cutmix_prob = args.cutmix_prob
    cutmix_alpha = args.cutmix_alpha

    use_mixup = args.use_mixup
    mixup_alpha = args.mixup_alpha

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_auroc = train_model(model, train_loader, criterion, optimizer, device, use_cutmix, cutmix_prob, cutmix_alpha, use_mixup, mixup_alpha, binary_feature_indices, numerical_feature_indices)
        val_loss, val_auroc = validate_model(model, validation_loader, criterion, device, binary_feature_indices, numerical_feature_indices)

        # Save losses for plotting
        train_losses.append(train_loss)
        train_aurocs.append(train_auroc)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)

        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss}, Training AUROC: {train_auroc}, Validation Loss: {val_loss}, Validation AUROC: {val_auroc}')

        # log metrics to wandb
        wandb.log({"train_loss": train_loss, "train_auroc": train_auroc, "val_loss": val_loss, "val_auroc": val_auroc})

        # Check for early stopping conditions
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            # Save the best model
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"No improvement in validation AUROC for {patience_counter} epochs.")
            if patience_counter >= early_stopping_patience:
                print(f"Stopping early at epoch {epoch+1}")
                break

    # Define the directory where model weights will be saved
    model_weights_dir = 'model_weights'
    # Ensure the directory exists
    os.makedirs(model_weights_dir, exist_ok=True)

    # Checkpoint saving logic
    if patience_counter < early_stopping_patience:
        # Save checkpoints only if early stopping didn't trigger
        for checkpoint_epoch in range(10, args.epochs + 1, 10):
            model_filename = f"{args.model_type}_dim{args.dim}_dim{args.depth}_heads{args.heads}_fdr{args.ff_dropout}_adr{args.attn_dropout}_{args.outcome}_bs{args.batch_size}_lr{args.learning_rate}_ep{epoch + 1}_esp{args.early_stopping_patience}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_uc{'true' if args.use_cutmix else 'false'}.pth"
            # Modify the file path to include the model_weights directory
            model_filepath = os.path.join(model_weights_dir, model_filename)
            
            # Save the model checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "train_aurocs": train_aurocs,
                "val_losses": val_losses,
                "val_aurocs": val_aurocs
            }
            torch.save(checkpoint, model_filepath)
            wandb.save(model_filepath)
            
            print(f"Model checkpoint saved as {model_filepath}")
    else:
        # If early stopping was triggered, save the best model weights
        best_model_filename = f"{args.model_type}_dim{args.dim}_dim{args.depth}_heads{args.heads}_fdr{args.ff_dropout}_adr{args.attn_dropout}_{args.outcome}_bs{args.batch_size}_lr{args.learning_rate}_ep{epoch + 1}_esp{args.early_stopping_patience}_cmp{args.cutmix_prob}_cml{args.cutmix_alpha}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_uc{'true' if args.use_cutmix else 'false'}_best.pth"
        # Modify the file path to include the model_weights directory
        best_model_filepath = os.path.join(model_weights_dir, best_model_filename)
        
        # Save the best model checkpoint
        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": best_model_wts,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_aurocs": train_aurocs,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs
        }
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
                        choices=['Transformer','TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--dim', type=int, default=None, help='Dimension of the model')
    parser.add_argument('--depth', type=int, help='Depth of the model.')
    parser.add_argument('--heads', type=int, help='Number of attention heads.')
    parser.add_argument('--ff_dropout', type=float, help='Feed forward dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=None, help='Attention dropout rate')
    parser.add_argument('--outcome', type=str, required=True, choices=['A1cGreaterThan7', 'A1cLessThan7'], help='Outcome variable to predict')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--model_path', type=str, default=None,
                    help='Optional path to the saved model file to load before training')
    parser.add_argument('--run_id', type=str, default=None,
                    help='Optional Weights & Biases run ID.')
    parser.add_argument('--cutmix_prob', type=float, default=0.3, help='Probability to apply CutMix')
    parser.add_argument('--cutmix_alpha', type=float, default=10, help='Alpha value for the CutMix beta distribution. Higher values result in more mixing.')
    parser.add_argument('--use_mixup', action='store_true', help='Enable MixUp data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha value for the MixUp beta distribution. Higher values result in more mixing.')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix data augmentation')
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
        if args.dim is None:
            args.dim = 512  # Default for FTTransformer
        args.heads = args.heads if args.heads is not None else 8
        # args.num_encoder_layers = 6
        # args.num_decoder_layers = 6
        # args.dim_feedforward = 2048
        # args.dropout = 0.1
        # args.activation = torch.nn.ReLU()
        # args.custom_encoder = None
        # args.custom_decoder = None
        # args.layer_norm_eps = 1e-05
        # args.batch_first = False
        # args.norm_first = False
        # args.bias = True

    main(args)
    wandb.finish()