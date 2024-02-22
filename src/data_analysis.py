import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import plotly.express as px

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
    try:
        if code_type == 'ICD9':
            description = icd9_df.loc[icd_code, 'Description'] if icd_code in icd9_df.index else 'Description not found'
        elif code_type == 'ICD10':
            description = icd10_df.loc[icd_code, 'Description'] if icd_code in icd10_df.index else 'Description not found'
        else:
            description = 'Invalid code type'
    except KeyError:
        description = 'Description not found'

    return description

def extract_icd_info(col_name):
    # Split the column name based on underscores
    parts = col_name.split('_')
    # Extract the ICD code and type from the parts
    icd_code = parts[1]  # This gets the '001.1' from '001.1_ICD9'
    code_type = parts[2]  # This gets the 'ICD9' or 'ICD10' from '001.1_ICD9'
    return icd_code, code_type

# Load column names from files
with open('column_names.json', 'r') as file:
    column_names = json.load(file)

with open('encoded_feature_names.json', 'r') as file:
    encoded_feature_names = json.load(file)

with open('infrequent_categories.json', 'r') as file:
    infrequent_categories = json.load(file)

# Before loading data, get descriptions of exluded ICD codes
infrequent_categories_df = []

for col_name in infrequent_categories:
    # Use the extract_icd_info function to get ICD code and code type
    icd_code, code_type = extract_icd_info(col_name)

    # Call the function with the extracted code, code type, and the respective DataFrame
    description = get_icd_description(icd_code, code_type, icd9_df, icd10_df)
    
    # Append the results to the data list
    infrequent_categories_df.append({
        "ICD Code Type": code_type,
        "ICD Code": icd_code,
        "Description": description
    })

# Create a DataFrame from the data list
infrequent_categories_icd_info = pd.DataFrame(infrequent_categories_df)

# Convert the DataFrame to an HTML table
infrequent_categories_html_table = infrequent_categories_icd_info.to_html(index=False)

# Save the HTML table to a file
with open("excluded_codes_descriptions.html", "w") as file:
    file.write(infrequent_categories_html_table)

# Load the preprocessed tensor
loaded_train_dataset = torch.load('train_dataset.pt')
loaded_validation_dataset = torch.load('validation_dataset.pt')
loaded_test_dataset = torch.load('test_dataset.pt')

# Convert to DataFrames for analysis
df_train = pd.DataFrame(loaded_train_dataset.tensors[0].numpy(), columns=column_names)
df_validation = pd.DataFrame(loaded_validation_dataset.tensors[0].numpy(), columns=column_names)
df_test = pd.DataFrame(loaded_test_dataset.tensors[0].numpy(), columns=column_names)

print(df_train.describe())

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

# Fetch ICD code descriptions for encoded_feature_names
icd_descriptions = []
for feature in encoded_feature_names:
    # Extract ICD code and type 
    icd_code, code_type = extract_icd_info(feature)
    description = get_icd_description(icd_code, code_type, icd9_df, icd10_df)
    icd_descriptions.append(description)

# Create a column in df_train_summary_sorted for these descriptions
df_train_summary_sorted['ICD Description'] = icd_descriptions

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