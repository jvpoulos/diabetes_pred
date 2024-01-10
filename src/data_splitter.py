import torch
from torch.utils.data import TensorDataset, random_split


# Load the tensor from the file
loaded_tensor = torch.load('preprocessed_tensor.pt')

# Assuming pytorch_tensor is your tensor
dataset = TensorDataset(loaded_tensor)

# Calculate the lengths of each split
total_size = len(dataset)
train_size = int(0.7 * total_size)
print("Training set size:", train_size)
validation_size = int(0.2 * total_size)
print("Validation set size:", validation_size)
test_size = total_size - train_size - validation_size
print("Test set size:", test_size)

# Perform the split
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# Save the datasets to files
torch.save(train_dataset, 'train_dataset.pt')
torch.save(validation_dataset, 'validation_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')