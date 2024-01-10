import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the preprocessed tensor
loaded_train_dataset = torch.load('train_dataset.pt')
loaded_validation_dataset = torch.load('validation_dataset.pt')
loaded_test_dataset = torch.load('test_dataset.pt')

# Convert to DataFrames for analysis
# Assuming that the subsets contain tensor data
df_train = pd.DataFrame(loaded_train_dataset.dataset.tensors[0].numpy())
df_validation = pd.DataFrame(loaded_validation_dataset.dataset.tensors[0].numpy())
df_test = pd.DataFrame(loaded_test_dataset.dataset.tensors[0].numpy())

# # Load the lists of categorical and numeric columns
# with open('categorical_columns.json', 'r') as file:
#     categorical_columns = json.load(file)

with open('numeric_columns.json', 'r') as file:
    numeric_columns = json.load(file)

# Determine categorical columns as all other columns not in numeric_columns
categorical_columns = [col for col in df_train.columns if col not in numeric_columns]

# Print column names of df_train for verification
print(df_train.columns)

# Identify and remove missing columns from numeric_columns
missing_columns = [col for col in numeric_columns if col not in df_train.columns]
print("Missing columns in df_train:", missing_columns)

# Update numeric_columns by removing the missing columns
numeric_columns = [col for col in numeric_columns if col in df_train.columns]

# Identify missing values for numeric columns (denoted by -1)
numeric_missing_rate = df_train[numeric_columns].apply(lambda x: (x == -1).mean())

# Identify one-hot encoded columns representing missing values (with _-1 in column name)
missing_value_columns = [col for col in df_train.columns if '_-1' in str(col)]
one_hot_missing_rate = df_train[missing_value_columns].mean()

# Calculate missing rate for numeric columns
if numeric_columns:  # Ensure there are numeric columns to process
    numeric_missing_rate = df_train[numeric_columns].apply(lambda x: (x == -1).mean())

    # Plotting the missing rate for numeric features
    if not numeric_missing_rate.empty:  # Check if numeric_missing_rate is not empty
        plt.figure(figsize=(12, 6))
        sns.barplot(x=numeric_missing_rate.index, y=numeric_missing_rate.values)
        plt.title("Missing Rate per Numeric Feature in Training Set")
        plt.xlabel("Feature")
        plt.ylabel("Missing Rate")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout
        plt.savefig("numeric_missing_rate_plot.png")  # Save plot to file
    else:
        print("No numeric features to plot missing rate for.")
else:
    print("No numeric columns found in df_train.")

# Plot missing rate for one-hot encoded features and save to file
plt.figure(figsize=(12, 6))
sns.barplot(x=one_hot_missing_rate.index, y=one_hot_missing_rate.values)
plt.title("Missing Rate per One-Hot Encoded Feature in Training Set")
plt.xlabel("Feature")
plt.ylabel("Missing Rate")
plt.xticks(rotation=45)
plt.savefig("one_hot_encoded_features_missing_rate.png")

# Filter out extremely rare features (features with missing rate >= 99%)
rare_numeric_features = numeric_missing_rate[numeric_missing_rate >= 0.99].index
rare_one_hot_features = one_hot_missing_rate[one_hot_missing_rate >= 0.99].index

all_rare_features = set(rare_numeric_features).union(set(rare_one_hot_features))
non_rare_features = [col for col in df_train.columns if col not in all_rare_features]

# Apply filtering
df_train_filtered = df_train[non_rare_features]
df_validation_filtered = df_validation[non_rare_features]
df_test_filtered = df_test[non_rare_features]

# Task 5: Print number of features after filtering out rare features
print("Number of features after filtering:", df_train_filtered.shape[1])

# Convert to PyTorch tensors and save
filtered_train_tensor = torch.tensor(df_train_filtered.values, dtype=torch.float32)
filtered_validation_tensor = torch.tensor(df_validation_filtered.values, dtype=torch.float32)
filtered_test_tensor = torch.tensor(df_test_filtered.values, dtype=torch.float32)

torch.save(filtered_train_tensor, 'filtered_training_tensor.pt')
torch.save(filtered_validation_tensor, 'filtered_validation_tensor.pt')
torch.save(filtered_test_tensor, 'filtered_test_tensor.pt')