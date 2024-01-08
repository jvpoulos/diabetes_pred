import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the preprocessed tensor
tensor = torch.load('preprocessed_tensor.pt')
df = pd.DataFrame(tensor.numpy())  # Convert to DataFrame for analysis

# Print number of unique rows and features
print("Unique rows:", df.drop_duplicates().shape[0])
print("Total features:", df.shape[1])

# Print number of features and patients
print("Number of features:", df.shape[1])
print("Number of patients:", df.shape[0])

# Plot and summarize missing rate for features
missing_rate = df.isna().mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_rate.index, y=missing_rate.values)
plt.title("Missing Rate per Feature")
plt.xlabel("Feature Index")
plt.ylabel("Missing Rate")
plt.show()

# Summary statistics for categorical features
with open('categorical_columns.json', 'r') as file:
    categorical_columns = json.load(file)

categorical_stats = df[categorical_columns].apply(lambda col: col.value_counts(normalize=True))
print(categorical_stats)

# Task 5: Filter out extremely rare features
non_rare_features = missing_rate[missing_rate < 0.99].index
df_filtered = df[non_rare_features]

# Task 6: Print number of features after filtering out rare features
print("Number of features after filtering:", df_filtered.shape[1])

# Convert to PyTorch tensor
filtered_tensor = torch.tensor(df_filtered.values, dtype=torch.float32)

# Save the tensor to a file
torch.save(filtered_tensor, 'filtered_tensor.pt')