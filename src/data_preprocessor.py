import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Load column names from files
with open('column_names.json', 'r') as file:
    column_names = json.load(file)

with open('encoded_feature_names.json', 'r') as file:
    encoded_feature_names = json.load(file)

# Load the preprocessed tensor
loaded_train_dataset = torch.load('train_dataset.pt')
loaded_validation_dataset = torch.load('validation_dataset.pt')
loaded_test_dataset = torch.load('test_dataset.pt')

# Convert to DataFrames for analysis
df_train = pd.DataFrame(loaded_train_dataset.tensors[0].numpy(), columns=column_names)
df_validation = pd.DataFrame(loaded_validation_dataset.tensors[0].numpy(), columns=column_names)
df_test = pd.DataFrame(loaded_test_dataset.tensors[0].numpy(), columns=column_names)

print(df_train.describe())

# Normalize specified numeric columns in df_outcomes using the Min-Max scaling approach. 
# This method is chosen because it handles negative and zero values well, scaling the data to a [0, 1] range.

columns_to_normalize = ['InitialA1c', 'A1cAfter12Months', 'DaysFromIndexToInitialA1cDate', 
                        'DaysFromIndexToA1cDateAfter12Months', 'DaysFromIndexToFirstEncounterDate', 
                        'DaysFromIndexToLastEncounterDate', 'DaysFromIndexToLatestDate', 
                        'DaysFromIndexToPatientTurns18', 'AgeYears', 'BirthYear', 
                        'NumberEncounters', 'SDI_score']

scaler = MinMaxScaler()

# Fit on training data
df_train[columns_to_normalize] = scaler.fit_transform(df_train[columns_to_normalize])

print(df_train[columns_to_normalize].describe())

# Transform validation and test data
df_validation[columns_to_normalize] = scaler.transform(df_validation[columns_to_normalize])
df_test[columns_to_normalize] = scaler.transform(df_test[columns_to_normalize])

# Calculate sparsity rate for one-hot encoded columns

one_hot_sparsity_rate = 1-df_train[encoded_feature_names].mean()
print("One-hot sparsity rate (training set): ", one_hot_sparsity_rate)

if encoded_feature_names:
    one_hot_sparsity_rate = 1-df_train[encoded_feature_names].mean()

    # Filter out features with a positive sparsity rate
    positive_sparsity_rate = one_hot_sparsity_rate[one_hot_sparsity_rate > 0]

    if not positive_sparsity_rate.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(x=positive_sparsity_rate.index, y=positive_sparsity_rate.values)
        plt.title("Sparsity Rate per One-Hot Encoded Feature in Training Set")
        plt.xlabel("Feature")
        plt.ylabel("Sparsity Rate")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("one_hot_sparsity_rate_positive_plot.png")
    else:
        print("No one-hot encoded features with positive sparsity rate to plot.")
else:
    print("No one-hot encoded columns found in df_train.")


# Filter out extremely rare features (features with sparsity rate >= 99%)
mean_values = df_train[encoded_feature_names].mean()
nan_features = mean_values.index[mean_values.isna()]

rare_features = mean_values[mean_values >= 0.99].index.union(nan_features)
df_train_filtered = df_train.drop(columns=rare_features)
df_validation_filtered = df_validation.drop(columns=rare_features)
df_test_filtered = df_test.drop(columns=rare_features)

print("Number of features after filtering:", df_train_filtered.shape[1])

# Update the encoded_feature_names list to include only those present in df_train_filtered
encoded_feature_names_filtered = [col for col in encoded_feature_names if col in df_train_filtered.columns]

one_hot_sparsity_rate_filtered = 1-df_train_filtered[encoded_feature_names_filtered].mean()
print("One-hot sparsity rate (training set): ", one_hot_sparsity_rate_filtered)

# Calculate sparsity rate for one-hot encoded columns in df_train_filtered
if encoded_feature_names_filtered:
    one_hot_sparsity_rate_filtered = 1- df_train_filtered[encoded_feature_names_filtered].mean()

    # Select features with a positive sparsity rate in the filtered dataset
    positive_sparsity_rate_filtered = one_hot_sparsity_rate_filtered[one_hot_sparsity_rate_filtered > 0]

    # Plotting the sparsity rate for one-hot encoded features with positive sparsity rate in the filtered dataset
    if not positive_sparsity_rate_filtered.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(x=positive_sparsity_rate_filtered.index, y=positive_sparsity_rate_filtered.values)
        plt.title("Sparsity Rate per One-Hot Encoded Feature in Filtered Training Set")
        plt.xlabel("Feature")
        plt.ylabel("Sparsity Rate")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout
        plt.savefig("one_hot_sparsity_rate_positive_filtered_plot.png")  # Save plot to file
    else:
        print("No one-hot encoded features with positive sparsity rate to plot in the filtered dataset.")
else:
    print("No one-hot encoded columns found in df_train_filtered.")

# save training set as text file
np.savetxt('df_train_filtered_values.txt', df_train_filtered.values)

# Convert to PyTorch tensors and save
filtered_train_tensor = torch.tensor(df_train_filtered.values, dtype=torch.float32)
filtered_validation_tensor = torch.tensor(df_validation_filtered.values, dtype=torch.float32)
filtered_test_tensor = torch.tensor(df_test_filtered.values, dtype=torch.float32)

torch.save(filtered_train_tensor, 'filtered_training_tensor.pt')
torch.save(filtered_validation_tensor, 'filtered_validation_tensor.pt')
torch.save(filtered_test_tensor, 'filtered_test_tensor.pt')