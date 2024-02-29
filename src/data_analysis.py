import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import plotly.express as px

# Function to format ICD codes (remove dots and leading zeros)
def format_icd_code(icd_code):
    # Remove dots and leading zeros
    return icd_code.replace('.', '').lstrip('0')

# Get ICD description
def get_icd_description(icd_code, code_type, icd9_df, icd10_df, icd9_txt_df, icd10_txt_df):
    formatted_icd_code = format_icd_code(icd_code)  # Ensure the ICD code is formatted correctly
    description = 'Description not found'
    
    if code_type.upper() == 'ICD9':
        # First attempt to find the description in the ICD-9 dataframe
        description = icd9_df.get(formatted_icd_code, {}).get('LONG DESCRIPTION (VALID ICD-9 FY2024)', 'Description not found')
        
        # If not found, look up in ICD9.txt DataFrame
        if description == 'Description not found' and formatted_icd_code in icd9_txt_df.index:
            description = icd9_txt_df.loc[formatted_icd_code, 'Description']
            
    elif code_type.upper() == 'ICD10':
        # First attempt to find the description in the ICD-10 dataframe
        description = icd10_df.get(formatted_icd_code, {}).get('LONG DESCRIPTION (VALID ICD-10 FY2024)', 'Description not found')
        
        # If not found, look up in ICD10.txt DataFrame
        if description == 'Description not found' and formatted_icd_code in icd10_txt_df.index:
            description = icd10_txt_df.loc[formatted_icd_code, 'Description']
    
    return description

# Extract ICD code and type
def extract_icd_info(col_name):
    parts = col_name.split('_')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

print("Loading ICD .csv files")
# Load the ICD-9 DataFrame with the correct columns
icd9_df = pd.read_excel('data/Section111ValidICD9-Jan2024.xlsx', engine='openpyxl', dtype=str)
# Assuming 'CODE' is the first column and already in the correct format, set it as the index
icd9_df.set_index(icd9_df.columns[0], inplace=True)
# Replace NaN descriptions with a default message
icd9_df.fillna('Description not available', inplace=True)

# Load the ICD-10 DataFrame with the correct columns
icd10_df = pd.read_excel('data/Section111ValidICD10-Jan2024.xlsx', engine='openpyxl', dtype=str)
# Assuming 'CODE' is the first column and already in the correct format, set it as the index
icd10_df.set_index(icd10_df.columns[0], inplace=True)
# Replace NaN descriptions with a default message
icd10_df.fillna('Description not available', inplace=True)

# Load the ICD-9 and ICD-10 descriptions into a dictionary for quick access
icd9_descriptions = icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].to_dict()
icd10_descriptions = icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].to_dict()

# Confirm that 'LONG DESCRIPTION' columns are present and have no NaN values before proceeding
assert 'LONG DESCRIPTION (VALID ICD-9 FY2024)' in icd9_df.columns and not icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].isnull().any()
assert 'LONG DESCRIPTION (VALID ICD-10 FY2024)' in icd10_df.columns and not icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].isnull().any()

# Before accessing 'LONG DESCRIPTION (VALID ICD-9 FY2024)', confirm it exists
if 'LONG DESCRIPTION (VALID ICD-9 FY2024)' in icd9_df.columns:
    # Safe to access the column
    print(icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].head())
else:
    print("'LONG DESCRIPTION (VALID ICD-9 FY2024)' column not found.")

# Apply formatting function to the DataFrame indices
icd9_df.index = icd9_df.index.map(format_icd_code)
icd10_df.index = icd10_df.index.map(format_icd_code)

# Adjust column names based on the actual structure of the DataFrame
icd9_df_column_names = ['CODE', 'LONG DESCRIPTION (VALID ICD-9 FY2024)']
icd10_df_column_names = ['CODE', 'SHORT DESCRIPTION (VALID ICD-10 FY2024)', 'LONG DESCRIPTION (VALID ICD-10 FY2024)']

# Ensure you're setting the correct number of column names
if len(icd9_df.columns) == len(icd9_df_column_names):
    icd9_df.columns = icd9_df_column_names
else:
    print(f"Warning: icd9_df has {len(icd9_df.columns)} columns, but {len(icd9_df_column_names)} names were provided.")

if len(icd10_df.columns) == len(icd10_df_column_names):
    icd10_df.columns = icd10_df_column_names
else:
    print(f"Warning: icd10_df has {len(icd10_df.columns)} columns, but {len(icd10_df_column_names)} names were provided.")

# Continue with setting the index and other operations
icd9_df.set_index('CODE', inplace=True)
icd10_df.set_index('CODE', inplace=True)

print("Loading ICD .txt files.")

# Load the text files into DataFrames
# Try different encodings such as 'ISO-8859-1' or 'cp1252' if 'utf-8' doesn't work
icd9_txt_df = pd.read_csv('data/ICD9.txt', sep=',', header=None, names=['ICD9code', 'Description', 'Type', 'Source', 'Dt'], encoding='ISO-8859-1')
icd10_txt_df = pd.read_csv('data/ICD10.txt', sep=',', header=None, names=['GUID', 'ICD10code', 'Description', 'Type', 'Source', 'Language', 'LastUpdateDTS', 'Dt'], encoding='ISO-8859-1')

# Convert code columns to strings to ensure proper matching
icd9_txt_df['ICD9code'] = icd9_txt_df['ICD9code'].astype(str)
icd10_txt_df['ICD10code'] = icd10_txt_df['ICD10code'].astype(str)

# Index the DataFrames on the code columns for fast lookup
icd9_txt_df.set_index('ICD9code', inplace=True)
icd10_txt_df.set_index('ICD10code', inplace=True)

# Load the preprocessed tensor
loaded_train_dataset = torch.load('train_dataset.pt')
loaded_validation_dataset = torch.load('validation_dataset.pt')
loaded_test_dataset = torch.load('test_dataset.pt')

# Load column names
with open('column_names.json', 'r') as file:
    column_names = json.load(file)

# Convert to DataFrames for analysis
df_train = pd.DataFrame(loaded_train_dataset.tensors[0].numpy(), columns=column_names)
df_validation = pd.DataFrame(loaded_validation_dataset.tensors[0].numpy(), columns=column_names)
df_test = pd.DataFrame(loaded_test_dataset.tensors[0].numpy(), columns=column_names)

# Print dimensions of datasets
# print("Training Data Dimensions:", df_train.shape)
# print("Validation Data Dimensions:", df_validation.shape)
# print("Test Data Dimensions:", df_test.shape)

# Save the training data to disk
#df_train.to_csv('training_data.csv', index=False)

# Print dataset descriptions
print("Training Dataset Description:\n", df_train.describe())

# Calculate sparsity rate for one-hot encoded columns

with open('encoded_feature_names.json', 'r') as file:
    encoded_feature_names = json.load(file)

one_hot_sparsity_rate = 1-df_train[encoded_feature_names].mean()
print("One-hot sparsity rate (training set): ", one_hot_sparsity_rate)

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

df_train_summary = df_train[encoded_feature_names].describe().T

# Now you can add the 'Description' column
df_train_summary['Description'] = df_train_summary.index.map(
    lambda code: get_icd_description(
        code.split('_')[0], 
        'ICD10' if 'ICD10' in code else 'ICD9', 
        icd9_df, 
        icd10_df,
        icd9_txt_df,
        icd10_txt_df
    )
)

# Then you can sort by 'mean' and proceed with the mapping for descriptions.
df_train_summary_sorted = df_train_summary.sort_values(by='mean', ascending=False)

# Save to HTML
df_train_summary_html = df_train_summary_sorted.to_html()
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