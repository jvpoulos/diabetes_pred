import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset  # Import TensorDataset
import json

# Load column names from a separate file
with open('column_names.json', 'r') as file:
    column_names = json.load(file)

# Load the tensor from the file
loaded_tensor = torch.load('preprocessed_tensor.pt')

# Convert the tensor to a DataFrame and assign column names
df = pd.DataFrame(loaded_tensor.numpy(), columns=column_names)

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
validation_df, test_df = train_test_split(temp_df, test_size=(1/3), random_state=42)

print("Training set size:", train_df.shape[0])
print("Validation set size:", validation_df.shape[0])
print("Test set size:", test_df.shape[0])

# Convert the DataFrames back to tensors for PyTorch processing
train_dataset = TensorDataset(torch.tensor(train_df.values, dtype=torch.float32))
validation_dataset = TensorDataset(torch.tensor(validation_df.values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_df.values, dtype=torch.float32))

# Save the datasets to files
torch.save(train_dataset, 'train_dataset.pt')
torch.save(validation_dataset, 'validation_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
