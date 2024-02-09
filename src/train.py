import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tab_transformer_pytorch import TabTransformer, FTTransformer

def train_model(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0

    for data, labels in train_loader:
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

    with torch.no_grad():  # No gradients to track
        for data, labels in validation_loader:
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())  # Compute loss
            total_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs))  # Convert to binary predictions
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    average_loss = total_loss / len(validation_loader)
    accuracy = correct / total
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
    return average_loss, accuracy

def main(args):
    # Load datasets
    train_dataset = torch.load('filtered_training_tensor.pt')
    validation_dataset = torch.load('filtered_validation_tensor.pt')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # Define hyperparameters ranges to test
    learning_rates = [0.001, 0.0001]
    batch_sizes = [32, 64]
    # Add other hyperparameters here as needed

    best_validation_score = float('inf')
    best_hyperparams = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Initialize model, criterion, optimizer here with the current set of hyperparameters
            model = TabTransformer(
                categories=[2 for _ in range(len(encoded_feature_names))],  # Assuming encoded_feature_names is defined
                num_continuous=12,
                dim=32,
                dim_out=1,
                depth=6,
                heads=8,
                attn_dropout=0.1,
                ff_dropout=0.1,
                mlp_hidden_mults=(4, 2),
                mlp_act=nn.ReLU()
            ) if args.model_type == 'TabTransformer' else FTTransformer(
                categories = categories,      # tuple containing the number of unique values within each category
                num_continuous = 12,                # number of continuous values
                dim = 32,                           # dimension, paper set at 32
                dim_out = 1,                        # binary prediction, but could be anything
                depth = 6,                          # depth, paper recommended 6
                heads = 8,                          # heads, paper recommends 8
                attn_dropout = 0.1,                 # post-attention dropout
                ff_dropout = 0.1                    # feed forward dropout
            )

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train the model with the current set of hyperparameters
            train_model(model, train_loader, criterion, optimizer)

            # Evaluate the model on the validation set
            validation_score, _ = validate_model(model, validation_loader, criterion)

            # Update best hyperparameters based on validation performance
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_hyperparams = {'lr': lr, 'batch_size': batch_size}
                best_model = model

    print(f"Best hyperparameters: {best_hyperparams}")

    # Save the best model to a file
    torch.save(best_model.state_dict(), 'best_model.pth')

    # Final validation on the separate validation set
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    validate_model(best_model, validation_loader, criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an attention network.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TabTransformer', 'FTTransformer'],
                        help='Type of the model to train: TabTransformer or FTTransformer')
    # Add argparse arguments for batch_size, learning_rate, etc., if you plan to vary them during runs
    args = parser.parse_args()
    main(args)