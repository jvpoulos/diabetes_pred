import pickle
import matplotlib.pyplot as plt
import os
import re
import argparse

# Extract hyperparameters from the filename
def extract_hyperparameters(filename):
    pattern = re.compile(r'(\w+)-([\w\.]+)')
    matches = pattern.findall(filename)
    hyperparameters = {}
    for key, value in matches:
        if value.replace('.', '', 1).isdigit() and '_' in value:  # Check for numeric values with underscores
            hyperparameters[key] = float(value.replace('_', '.'))
        elif value.isdigit():  # Check for pure numeric values
            hyperparameters[key] = int(value)
        else:
            hyperparameters[key] = value  # Keep as string
    return hyperparameters

# Function to plot AUROC
def plot_auroc(val_aurocs, hyperparameters, plot_dir='auroc_plots'):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(val_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('Validation AUROC Over Epochs')
    plt.legend()

    filename = '_'.join([f"{key}-{value}" for key, value in hyperparameters.items()]) + '_auroc.png'
    filepath = os.path.join(plot_dir, filename)

    plt.savefig(filepath)
    plt.close()

# Function to plot losses
def plot_losses(train_losses, val_losses, hyperparameters, plot_dir='loss_plots'):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    filename = '_'.join([f"{key}-{value}" for key, value in hyperparameters.items()]) + '.png'
    filepath = os.path.join(plot_dir, filename)

    plt.savefig(filepath)
    plt.close()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot training losses and AUROCs from a file.')
parser.add_argument('performance_file_name', type=str, help='The path to the performance data file.')
args = parser.parse_args()

# Now, use args.performance_file_name to access the file name provided by the user
performance_file_name = args.performance_file_name

# Load losses and aurocs from file
with open(performance_file_name, 'rb') as f:
    losses_and_aurocs = pickle.load(f)

# Extract losses and aurocs from the loaded object
train_losses = losses_and_aurocs['train_losses']
val_losses = losses_and_aurocs['val_losses']
val_aurocs = losses_and_aurocs['val_aurocs']

# Extract hyperparameters from file name
hyperparameters = extract_hyperparameters(args.performance_file_name)

# Plot losses and aurocs
plot_losses(train_losses, val_losses, hyperparameters)
plot_auroc(val_aurocs, hyperparameters)