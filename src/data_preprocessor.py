import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load column names from a separate file
with open('column_names.json', 'r') as file:
    column_names = json.load(file)

with open('numeric_columns.json', 'r') as file:
    numeric_columns = json.load(file) # ['Dataset_EMPI', 'InitialA1c', 'Demp_Age']

with open('categorical_columns.json', 'r') as file:
    categorical_columns = json.load(file)  

# Load the preprocessed tensor
loaded_train_dataset = torch.load('train_dataset.pt')
loaded_validation_dataset = torch.load('validation_dataset.pt')
loaded_test_dataset = torch.load('test_dataset.pt')

# Convert to DataFrames for analysis
df_train = pd.DataFrame(loaded_train_dataset.tensors[0].numpy(), columns=column_names)
df_validation = pd.DataFrame(loaded_validation_dataset.tensors[0].numpy(), columns=column_names)
df_test = pd.DataFrame(loaded_test_dataset.tensors[0].numpy(), columns=column_names)

#print("Training set column Names (preprocessed):")
#print(df_train.columns.tolist())

# Removing 'Dataset_EMPI' from the numeric list
numeric_columns = [col for col in numeric_columns if col != 'Dataset_EMPI']

# Removing 'Dataset_EMPI' from the list
numeric_columns = [col for col in numeric_columns if col != 'Dataset_EMPI']

# Identify missing values for numeric columns (denoted by -1)
if numeric_columns:  # Check if there are any numeric columns
    numeric_missing_rate = df_train[numeric_columns].apply(lambda x: (x == -1).mean())

    # Plot and save the missing rate for numeric features
    if not numeric_missing_rate.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(x=numeric_missing_rate.index, y=numeric_missing_rate.values)
        plt.title("Missing Rate per Numeric Feature in Training Set")
        plt.xlabel("Feature")
        plt.ylabel("Missing Rate")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("numeric_missing_rate_plot.png")
    else:
        print("No numeric columns to plot missing rate for.")
else:
    print("No numeric columns found in df_train.")

# Identify one-hot encoded columns representing missing values (with _-1 in column name)
one_hot_missing_columns = [col for col in df_train.columns if '-1' in str(col)] # # convert each column name to string before checking for '-1' substring

one_hot_missing_rate = df_train[one_hot_missing_columns].mean()

# Calculate missing rate for one-hot encoded columns
if one_hot_missing_columns:
    one_hot_missing_rate = (df_train[one_hot_missing_columns] == 1).mean()

    # Filter out features with a missing rate of 0
    positive_missing_rate = one_hot_missing_rate[one_hot_missing_rate > 0]

    print(positive_missing_rate)

    # Plotting the missing rate for one-hot encoded features with positive missing rate
    if not positive_missing_rate.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(x=positive_missing_rate.index, y=positive_missing_rate.values)
        plt.title("Missing Rate per One-Hot Encoded Feature in Training Set (Positive Missing Rates Only)")
        plt.xlabel("Feature")
        plt.ylabel("Missing Rate")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout
        plt.savefig("one_hot_missing_rate_positive_plot.png")  # Save plot to file
    else:
        print("No one-hot encoded features with positive missing rate to plot.")
else:
    print("No one-hot encoded columns found in df_train.")

# Filter out extremely rare features (features with missing rate >= 99%)
rare_numeric_features = numeric_missing_rate[numeric_missing_rate >= 0.99].index
rare_one_hot_features = one_hot_missing_rate[one_hot_missing_rate >= 0.99].index

all_rare_features = set(rare_numeric_features).union(set(rare_one_hot_features))
non_rare_features = [col for col in df_train.columns if col not in all_rare_features]

# Apply filtering
df_train_filtered = df_train[non_rare_features]
df_validation_filtered = df_validation[non_rare_features]
df_test_filtered = df_test[non_rare_features]

# Update the one_hot_missing_columns list to include only those present in df_train_filtered
one_hot_missing_columns_filtered = [col for col in one_hot_missing_columns if col in df_train_filtered.columns]

# Calculate missing rate for one-hot encoded columns in df_train_filtered
if one_hot_missing_columns_filtered:
    one_hot_missing_rate_filtered = (df_train_filtered[one_hot_missing_columns_filtered] == 1).mean()

    # Filter out features with a missing rate of 0 in the filtered dataset
    positive_missing_rate_filtered = one_hot_missing_rate_filtered[one_hot_missing_rate_filtered > 0]

    print(positive_missing_rate_filtered)

    # Plotting the missing rate for one-hot encoded features with positive missing rate in the filtered dataset
    if not positive_missing_rate_filtered.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(x=positive_missing_rate_filtered.index, y=positive_missing_rate_filtered.values)
        plt.title("Missing Rate per One-Hot Encoded Feature in Filtered Training Set (Positive Missing Rates Only)")
        plt.xlabel("Feature")
        plt.ylabel("Missing Rate")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout
        plt.savefig("one_hot_missing_rate_positive_filtered_plot.png")  # Save plot to file
    else:
        print("No one-hot encoded features with positive missing rate to plot in the filtered dataset.")
else:
    print("No one-hot encoded columns found in df_train_filtered.")

# Print number of features after filtering out rare features
print("Number of features after filtering:", df_train_filtered.shape[1])

# Convert to PyTorch tensors and save
filtered_train_tensor = torch.tensor(df_train_filtered.values, dtype=torch.float32)
filtered_validation_tensor = torch.tensor(df_validation_filtered.values, dtype=torch.float32)
filtered_test_tensor = torch.tensor(df_test_filtered.values, dtype=torch.float32)

torch.save(filtered_train_tensor, 'filtered_training_tensor.pt')
torch.save(filtered_validation_tensor, 'filtered_validation_tensor.pt')
torch.save(filtered_test_tensor, 'filtered_test_tensor.pt')