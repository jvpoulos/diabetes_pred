import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Load the ICD-9 and ICD-10 codes dataframes
icd9_df = pd.read_excel('data/Section111ValidICD9-Jan2024.xlsx', engine='openpyxl')
icd10_df = pd.read_excel('data/Section111ValidICD10-Jan2024.xlsx', engine='openpyxl')

# Adjust column names based on the structure of the files
icd9_df.columns = ['CODE', 'LONG_DESCRIPTION_ICD9', 'NF_EXCL_ICD9']
icd10_df.columns = ['CODE', 'SHORT_DESCRIPTION_ICD10', 'LONG_DESCRIPTION_ICD10', 'NF_EXCL_ICD10']

# Set the CODE column as the index for faster lookup
icd9_df.set_index('CODE', inplace=True)
icd10_df.set_index('CODE', inplace=True)

def get_icd_description(icd_column_name, icd9_df, icd10_df):
    # Split the column name by the underscore
    parts = icd_column_name.split('_')
    # Ignore columns with 'Date' in the name
    if "Date" in parts:
        return None
    
    # Extract the ICD code and code type from the column name
    icd_code = parts[0]
    code_type = parts[1] if len(parts) > 1 else None

    # Check the code type and select the appropriate DataFrame
    if code_type == 'ICD9':
        description = icd9_df.get(icd_code, 'Description not found')
    elif code_type == 'ICD10':
        description = icd10_df.get(icd_code, 'Description not found')
    else:
        description = 'Invalid code type'

    return description

# Assuming icd9_df and icd10_df are dictionaries or DataFrames with ICD codes as keys or indices and descriptions as values
# Here's how you would call the function for a column name like "001.1_ICD9"
column_name = "001.1_ICD9"
description = get_icd_description(column_name, icd9_df, icd10_df)
print(description)  # Prints the description for the ICD code "001.1" in ICD-9

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

# Identify columns that have all-zero values
all_zero_columns_mask = (df_train[encoded_feature_names] == 0).all(axis=0)
all_zero_columns = all_zero_columns_mask[all_zero_columns_mask].index.tolist()

with open('training_set_all_zero_columns.json', 'w') as file:
    json.dump(all_zero_columns, file)

# Calculate the share of columns with all-zero values
share_of_all_zero_columns = len(all_zero_columns) / len(encoded_feature_names)

print(f"Share of columns with all-zero values: {share_of_all_zero_columns:.2%}")

# Initialize a list to store data for the new DataFrame
data_for_df = []

for col_name in all_zero_columns:
    # Assuming the column name is structured as 'code_type_code'
    parts = col_name.split('_')
    icd_code = parts[1]  # This gets the '001.1' from '001.1_ICD9'
    code_type = parts[2]  # This gets the 'ICD9' from '001.1_ICD9'

    # Call the function with the extracted code, code type, and the respective DataFrame
    description = get_icd_description(icd_code, code_type, icd9_df, icd10_df)
    
    # Append the results to the data list
    data_for_df.append({
        "ICD Code Type": code_type,
        "ICD Code": icd_code,
        "Description": description
    })

# Create a DataFrame from the data list
df_icd_info = pd.DataFrame(data_for_df)

# Convert the DataFrame to an HTML table
html_table = df_icd_info.to_html(index=False)

# Save the HTML table to a file
with open("icd_codes_descriptions.html", "w") as file:
    file.write(html_table)

# Limit the number of features or samples due to visual and performance constraints
sampled_df = df_train[encoded_feature_names].sample(n=min(100, len(df_train)), axis=1, random_state=42) # limits to 100 features

# Calculate sparsity rate directly for features in sampled_df
sparsity_rate_sampled = 1 - sampled_df.mean()
print("One-hot sparsity rate (sampled filtered training set): ", sparsity_rate_sampled)

# Plotting the sparsity rate for one-hot encoded features in sampled_df
plt.figure(figsize=(15, 8))
sns.barplot(x=sparsity_rate_sampled.index, y=sparsity_rate_sampled.values)
plt.title("Sparsity Rate per One-Hot Encoded Feature in Training Set (100 Sampled Features)")
plt.xlabel("Feature")
plt.ylabel("Sparsity Rate")
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout
plt.savefig("one_hot_sparsity_rate_sampled_plot.png")  # Save plot to file

# Generate summary statistics table for df_train[encoded_feature_names]
df_train_summary = df_train[encoded_feature_names].describe(percentiles=[]).T  # Exclude percentiles

# Calculate the sum of each column and add it as a new row in the summary
column_sums = df_train[encoded_feature_names].sum()
df_train_summary['sum'] = column_sums

# Retain only 'count', 'sum', 'mean', 'std', and 'max' in the summary
df_train_summary_filtered = df_train_summary[['count', 'sum', 'mean', 'std']]

# Sort the features by 'sum'
df_train_summary_sorted = df_train_summary_filtered.sort_values(by='sum', ascending=False)

# Convert the sorted summary statistics table to HTML format
df_train_summary_html = df_train_summary_sorted.to_html()

# Save the summary statistics table as an HTML file
with open('df_train_summary_statistics.html', 'w') as f:
    f.write(df_train_summary_html)

# Transpose the DataFrame for the heatmap display (features on y-axis, samples on x-axis)
transposed_df = sampled_df.transpose()

# Create the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(transposed_df, cmap="viridis", cbar_kws={'label': 'Feature Presence (1) or Absence (0)'})
plt.title("Heatmap of One-Hot Encoded Feature Sparsity in Training Set (100 Sampled Features)")
plt.xlabel("Sample Index")
plt.ylabel("One-Hot Encoded Feature")

# Optional: Adjust the layout and aspect ratio for better visibility depending on the number of features
plt.tight_layout()

# Save the plot to a file
plt.savefig("one_hot_encoded_features_sparsity_sampled_heatmap.png")

# Calculate sparsity rate as the proportion of zeros in each feature
sparsity_rate = 1 - df_train[encoded_feature_names].mean()

# Identify features with sparsity rate >= 99%
rare_features = sparsity_rate[sparsity_rate >= 0.99].index

# Filter out these rare features from the datasets
df_train_filtered = df_train.drop(columns=rare_features)
df_validation_filtered = df_validation.drop(columns=rare_features)
df_test_filtered = df_test.drop(columns=rare_features)

print("Number of features after filtering:", df_train_filtered.shape[1])

# Update the encoded_feature_names list to include only those present in df_train_filtered
encoded_feature_names_filtered = [col for col in encoded_feature_names if col in df_train_filtered.columns]

one_hot_sparsity_rate_filtered = 1-df_train_filtered[encoded_feature_names_filtered].mean()
print("One-hot sparsity rate (filtered training set): ", one_hot_sparsity_rate_filtered)

# Limit the number of features or samples due to visual and performance constraints
sampled_df_filtered = df_train_filtered[encoded_feature_names_filtered].sample(n=min(len(encoded_feature_names_filtered), len(df_train_filtered)), axis=1, random_state=42)

# Calculate sparsity rate directly for features in sampled_df
sparsity_rate_sampled_filtered = 1 - sampled_df_filtered.mean()
print("One-hot sparsity rate (sampled filtered training set): ", sparsity_rate_sampled_filtered)

# Plotting the sparsity rate for one-hot encoded features in sampled_df
plt.figure(figsize=(15, 8))
sns.barplot(x=sparsity_rate_sampled_filtered.index, y=sparsity_rate_sampled_filtered.values)
plt.title("Sparsity Rate per One-Hot Encoded Feature in Filtered Training Set (100 Sampled Features)")
plt.xlabel("Feature")
plt.ylabel("Sparsity Rate")
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout
plt.savefig("one_hot_sparsity_rate_sampled_filtered_plot.png")  # Save plot to file

# Transpose the DataFrame for the heatmap display (features on y-axis, samples on x-axis)
transposed_filtered_df = sampled_df_filtered .transpose()

# Create the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(transposed_filtered_df, cmap="viridis", cbar_kws={'label': 'Feature Presence (1) or Absence (0)'})
plt.title("Heatmap of One-Hot Encoded Feature Sparsity in Filtered Training Set (100 Sampled Features)")
plt.xlabel("Sample Index")
plt.ylabel("One-Hot Encoded Feature")

# Optional: Adjust the layout and aspect ratio for better visibility depending on the number of features
plt.tight_layout()

# Save the plot to a file
plt.savefig("one_hot_encoded_features_sparsity_sampled_filtered_heatmap.png")

# Generate summary statistics table for df_train[encoded_feature_names]
df_train_filtered_summary = df_train_filtered[encoded_feature_names_filtered].describe(percentiles=[]).T

# Calculate the sum of each column and add it as a new row in the summary
column_sums_filtered = df_train_filtered[encoded_feature_names_filtered].sum()
df_train_filtered_summary['sum'] = column_sums_filtered 

# Retain only 'count' and 'mean' in the summary, along with std and max if needed
df_train_filtered_summary_filtered = df_train_filtered_summary[['count', 'sum', 'mean', 'std']]

# Sort the features by mean
df_train_filtered_summary_sorted = df_train_filtered_summary_filtered.sort_values(by='sum', ascending=False)

# Convert summary statistics table to HTML format
df_train_filtered_summary_html = df_train_filtered_summary_sorted.to_html()

# Save summary statistics table as HTML file
with open('df_train_filtered_summary_statistics.html', 'w') as f:
    f.write(df_train_filtered_summary_html)

# save training set as text file
np.savetxt('df_train_filtered_values.txt', df_train_filtered.values)

# Convert to PyTorch tensors and save
filtered_train_tensor = torch.tensor(df_train_filtered.values, dtype=torch.float32)
filtered_validation_tensor = torch.tensor(df_validation_filtered.values, dtype=torch.float32)
filtered_test_tensor = torch.tensor(df_test_filtered.values, dtype=torch.float32)

torch.save(filtered_train_tensor, 'filtered_training_tensor.pt')
torch.save(filtered_validation_tensor, 'filtered_validation_tensor.pt')
torch.save(filtered_test_tensor, 'filtered_test_tensor.pt')