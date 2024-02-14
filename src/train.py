import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tab_transformer import TabTransformer, FTTransformer
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy
import logging

logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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

def train_model(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()  # Clear gradients for the next train
        outputs = model(data)  # Forward pass: compute the output class given a image
        loss = criterion(outputs.squeeze(), labels.float())  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)
        
        total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / len(train_loader)
    print(f'Training loss: {average_loss:.4f}')
    return average_loss

def validate_model(model, validation_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total, correct = 0, 0
    all_labels = []  # List to store all true labels
    all_predictions = []  # List to store all predictions

    with torch.no_grad():  # No gradients to track
        for data, labels in validation_loader:
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
    train_dataset = torch.load('filtered_training_tensor.pt')
    validation_dataset = torch.load('filtered_validation_tensor.pt')

    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    # Excluded column names
    excluded_columns = ["A1cGreaterThan7", "A1cAfter12Months", "studyID"]

    # Find indices of the columns to be excluded
    excluded_indices = [column_names.index(col) for col in excluded_columns]

    # Find indices of all columns
    #  conversion from column names to indices is necessary because PyTorch tensors do not support direct column selection by name
    all_indices = list(range(len(column_names)))

    # Determine indices for features by excluding the indices of excluded columns
    feature_indices = [index for index in all_indices if index not in excluded_indices]

    # Assuming train_dataset and validation_dataset are tensors, use torch.index_select
    train_features = torch.index_select(train_dataset, 1, torch.tensor(feature_indices))
    train_labels = train_dataset[:, column_names.index("A1cGreaterThan7")]

    validation_features = torch.index_select(validation_dataset, 1, torch.tensor(feature_indices))
    validation_labels = validation_dataset[:, column_names.index("A1cGreaterThan7")]

    # Save to file (for attention.py)
    torch.save(validation_features, 'validation_features.pt')
    torch.save(validation_labels, 'validation_labels.pt')

    # Create custom datasets
    train_data = CustomDataset(train_features, train_labels)
    validation_data = CustomDataset(validation_features, validation_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    categories=[2 for _ in range(len(encoded_feature_names))]

    # Initialize model, criterion, optimizer here with the current set of hyperparameters
    model = TabTransformer(
        categories=categories,  # tuple containing the number of unique values within each category
        num_continuous=12,                                          # number of continuous values
        dim=32,                                                     # dimension, paper set at 32
        dim_out=1,                                                  # binary prediction, but could be anything
        depth=6,                                                    # depth, paper recommended 6
        heads=8,                                                    # heads, paper recommends 8
        attn_dropout=0.1,                                           # post-attention dropout
        ff_dropout=0.1,                                             # feed forward dropout
        mlp_hidden_mults=(4, 2),                                    # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU()                                           # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    ) if args.model_type == 'TabTransformer' else FTTransformer(
        categories = categories,      # tuple containing the number of unique values within each category
        num_continuous = 12,                # number of continuous values
        dim = 192,                           # dimension, paper set at 192
        dim_out = 1,                        # binary prediction, but could be anything
        depth = 3,                          # depth, paper recommended 3
        heads = 8,                          # heads, paper recommends 8
        attn_dropout = 0.2,                 # post-attention dropout
        ff_dropout = 0.1                    # feed forward dropout
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Lists to store loss values
    train_losses = []
    val_losses = []

    # Add early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience  # assuming this argument is added to the parser

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, _ = validate_model(model, validation_loader, criterion)

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
    plot_losses(train_losses, val_losses, {'lr': args.learning_rate, 'batch_size': args.batch_size})

    # Save the best model to a file
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an attention network.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    args = parser.parse_args()
    main(args)