import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy
import logging
import json
import torchvision
import torchvision.transforms
import random
import numpy as np
from tqdm import tqdm


# Set the max_split_size_mb parameter
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def apply_cutmix_numerical(data, labels, beta=1.0):
    lam = np.random.beta(beta, beta)
    batch_size = data.size(0)
    feature_count = data.size(1)
    index = torch.randperm(batch_size).to(data.device)

    # Determine the number of features to mix
    mix_feature_count = int(feature_count * lam)

    # Randomly choose the features to mix
    mix_features_indices = torch.randperm(feature_count)[:mix_feature_count]

    # Swap the chosen features
    data[:, mix_features_indices] = data[index, mix_features_indices]

    labels_a, labels_b = labels, labels[index]
    return data, labels_a, labels_b, lam


def apply_mixup_numerical(data, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(data.device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_data, labels_a, labels_b, lam

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Since features and labels are already tensors, no conversion is needed
        return self.features[idx], self.labels[idx]

def plot_losses(train_losses, val_losses, hyperparameters, plot_dir='loss_plots'):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Construct filename based on hyperparameters
    filename = '_'.join([f"{key}-{value}" for key, value in hyperparameters.items()]) + '.png'
    filepath = os.path.join(plot_dir, filename)
    
    plt.savefig(filepath)
    plt.close()

def train_model(model, train_loader, criterion, optimizer, device, use_cutmix, cutmix_prob, cutmix_lambda, use_mixup, mixup_alpha, binary_feature_indices, numerical_feature_indices):
    model.train()
    total_loss = 0

    for batch_idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        # Extracting categorical and numerical features based on their indices
        categorical_features = features[:, binary_feature_indices].to(device)
        # Convert categorical_features to long type before passing to the model
        categorical_features = categorical_features.long()  # Ensure this is done after moving to the device

        numerical_features = features[:, numerical_feature_indices].to(device)
        labels = labels.to(device)
        num_categorical_features = len(binary_feature_indices)  # Adjust this based on your dataset

    if use_mixup and np.random.rand() < mixup_alpha:
        combined_features = torch.cat((categorical_features, numerical_features), dim=1)
        augmented_data, labels_a, labels_b, lam = apply_mixup_numerical(combined_features, labels, mixup_alpha)
        # Assuming you know how to split augmented_data back
        augmented_cat = augmented_data[:, :num_categorical_features]
        augmented_num = augmented_data[:, num_categorical_features:]
        outputs = model(augmented_cat, augmented_num)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    elif use_cutmix and np.random.rand() < cutmix_prob:
        augmented_data, labels_a, labels_b, lam = apply_cutmix_numerical(features, labels, cutmix_lambda)
        # Splitting augmented_data back into augmented_cat and augmented_num
        augmented_cat = augmented_data[:, :num_categorical_features]
        augmented_num = augmented_data[:, num_categorical_features:]
        outputs = model(augmented_cat, augmented_num)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    else:
        combined_features = torch.cat((categorical_features, numerical_features), dim=1)
        # Directly using combined_features as your model might require adjustments to accept this
        outputs = model(categorical_features, numerical_features)  # Original call before error
        loss = criterion(outputs.squeeze(), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Training loss: {average_loss:.4f}')
    return average_loss

def validate_model(model, validation_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total, correct = 0, 0
    all_labels = []  # List to store all true labels
    all_predictions = []  # List to store all predictions

    with torch.no_grad():  # No gradients to track
        for data, labels in tqdm(validation_loader, desc="Validation"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())  # Compute loss
            total_loss += loss.item()

            # Instead of converting predictions to binary, keep the sigmoid outputs for AUC computation
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_predictions.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

            predicted = torch.round(torch.sigmoid(outputs))  # Convert to binary predictions for accuracy calculation
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    average_loss = total_loss / len(validation_loader)
    accuracy = correct / total
    auc_score = roc_auc_score(all_labels, all_predictions)  # Compute AUC score
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, AUC: {auc_score:.4f}')
    return average_loss, accuracy, auc_score

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

    # Excluded column names
    excluded_columns = ["A1cGreaterThan7", "A1cAfter12Months", "EMPI"]

    # Find indices of the one-hot encoded features
    binary_feature_indices = [column_names.index(col) for col in encoded_feature_names]

    # Find indices of the continuous features
    numerical_feature_indices = [column_names.index(col) for col in columns_to_normalize]

    # Assuming dataset is a TensorDataset containing a single tensor with both features and labels
    dataset_tensor = train_dataset.tensors[0]  # This gets the tensor from the dataset

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns]

    label_index = column_names.index("A1cGreaterThan7")

    # Selecting features and labels
    train_features = dataset_tensor[:, feature_indices]
    train_labels = dataset_tensor[:, label_index]

    # Repeat for validation dataset if necessary
    validation_dataset_tensor = validation_dataset.tensors[0]
    validation_features = validation_dataset_tensor[:, feature_indices]
    validation_labels = validation_dataset_tensor[:, label_index]

    # Save to file (for attention.py)
    torch.save(validation_features, 'validation_features.pt')
    torch.save(validation_labels, 'validation_labels.pt')

    print(f"Total columns in dataset: {len(column_names)}")
    print(f"Excluded columns: {excluded_columns}")
    print(f"Total feature indices: {len(feature_indices)}")

    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {validation_features.shape}")
    print(f"Validation labels shape: {validation_labels.shape}")

    # Set device to GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Initialize model, criterion, optimizer here with the current set of hyperparameters
    model = TabTransformer(
        categories=categories,  # tuple containing the number of unique values within each category
        num_continuous=len(numerical_feature_indices),              # number of continuous values
        dim=32,                                                     # dimension, paper set at 32
        dim_out=1,                                                  # binary prediction, but could be anything
        depth=6,                                                    # depth, paper recommended 6
        heads=8,                                                    # heads, paper recommends 8
        attn_dropout=0.1,                                           # post-attention dropout
        ff_dropout=0.1,                                             # feed forward dropout
        mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    ).to(device) if args.model_type == 'TabTransformer' else FTTransformer(
        categories = categories,      # tuple containing the number of unique values within each category
        num_continuous = len(numerical_feature_indices),  # number of continuous values
        dim = 192,                           # dimension, paper set at 192
        dim_out = 1,                        # binary prediction, but could be anything
        depth = 3,                          # depth, paper recommended 3
        heads = 8,                          # heads, paper recommends 8
        attn_dropout = 0.2,                 # post-attention dropout
        ff_dropout = 0.1                    # feed forward dropout
    ).to(device)
    
    if args.model_path:
        # Ensure the file exists before attempting to load
        if os.path.isfile(args.model_path):
            model.load_state_dict(torch.load(args.model_path))
            print(f"Loaded saved model from {args.model_path}")
        else:
            print(f"Saved model file {args.model_path} not found. Training from scratch.")

    # Using multiple GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Lists to store loss values
    train_losses = []
    val_losses = []

    # Add early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience  # assuming this argument is added to the parser

    # CutMix and mixup parameters
    use_cutmix = args.use_cutmix
    cutmix_prob=args.cutmix_prob
    cutmix_lambda=args.cutmix_lambda

    use_mixup=args.use_mixup
    mixup_alpha=args.mixup_alpha

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, use_cutmix, cutmix_prob, cutmix_lambda, use_mixup, mixup_alpha, binary_feature_indices, numerical_feature_indices)
        val_loss, _ = validate_model(model, validation_loader, criterion, device)

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        # Check for early stopping conditions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs.")
            if patience_counter >= early_stopping_patience:
                print(f"Stopping early at epoch {epoch+1}")
                break

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # After training, plot the losses
    hyperparameters = {
    'model_type': args.model_type,
    'batch_size': args.batch_size,
    'lr': args.learning_rate,
    'ep': args.epochs,
    'esp': args.early_stopping_patience,
    'cutmix_prob': args.cutmix_prob,
    'cutmix_lambda': args.cutmix_lambda,
    'use_mixup': 'true' if args.use_mixup else 'false',
    'mixup_alpha': args.mixup_alpha,
    'use_cutmix': 'true' if args.use_cutmix else 'false'
    }

    plot_losses(train_losses, val_losses, hyperparameters)

    # Save the best model to a file
    model_filename = f"{args.model_type}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.epochs}_esp{args.early_stopping_patience}_cmp{args.cutmix_prob}_cml{args.cutmix_lambda}_um{'true' if args.use_mixup else 'false'}_ma{args.mixup_alpha}_uc{'true' if args.use_cutmix else 'false'}.pth"

    torch.save(model.state_dict(), model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an attention network.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--model_path', type=str, default=None,
                    help='Optional path to the saved model file to load before training')
    parser.add_argument('--cutmix_prob', type=float, default=0.3, help='Probability to apply CutMix')
    parser.add_argument('--cutmix_lambda', type=float, default=10.0, help='Lambda for CutMix blending')
    parser.add_argument('--use_mixup', action='store_true', help='Enable MixUp data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha value for the MixUp beta distribution. Higher values result in more mixing.')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix data augmentation')
    args = parser.parse_args()
    main(args)