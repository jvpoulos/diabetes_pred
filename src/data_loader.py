import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import io
from tqdm import tqdm
import numpy as np
import gc
import json
import re

def optimize_dataframe(df):
    # Convert columns to more efficient types if possible
    # For example, convert object columns to category if they have limited unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Adjust this threshold as needed
                df[col] = df[col].astype('category')
    return df

def read_file(file_path, columns_type, columns_select, parse_dates=None, chunk_size=10000):
    """
    Reads a CSV file with a progress bar, selecting specific columns.

    Parameters:
        file_path (str): Path to the file.
        columns_type (dict): Dictionary of column names and their data types.
        columns_select (list): List of columns to read.
        parse_dates (list or None): List of columns to parse as dates.
        chunk_size (int): Number of rows per chunk.

    Returns:
        DataFrame: The read DataFrame.
    """
    # Filter columns_type to include only those columns in columns_select
    filtered_columns_type = {col: columns_type[col] for col in columns_select if col in columns_type}

    # Initialize a DataFrame to store the data
    data = pd.DataFrame()

    # Estimate the number of chunks
    total_rows = sum(1 for _ in open(file_path))
    total_chunks = (total_rows // chunk_size) + 1

    # Read the file in chunks with a progress bar
    with tqdm(total=total_chunks, desc="Reading CSV") as pbar:
        for chunk in pd.read_csv(file_path, sep='|', dtype=filtered_columns_type, usecols=columns_select,
                                 parse_dates=parse_dates, chunksize=chunk_size, low_memory=False):
            data = pd.concat([data, chunk], ignore_index=True)
            pbar.update(1)

    return data

def encode_batch(batch, encoder):
    """
    Encodes a batch of data using one-hot encoding.

    Parameters:
        batch (DataFrame): The batch of data to encode.
        encoder (OneHotEncoder): The encoder to use for one-hot encoding.

    Returns:
        DataFrame: The encoded data.
    """
   # Ensure all data in batch is string type for consistent encoding
    batch_str = batch.astype(str)
    encoded = encoder.transform(batch_str)
    return pd.DataFrame(encoded, index=batch.index)

# Define the column types for each file type

outcomes_columns = {
    'studyID': 'int64',
    'EMPI': 'int64',
    'IndexDate': 'object',
    'InitialA1c': 'float64',
    'A1cAfter12Months': 'float64',
    'A1cGreaterThan7': 'int64',
    'Female': 'int64',
    'Married': 'int64',
    'GovIns': 'int64',
    'English': 'int64',
    'DaysFromIndexToInitialA1cDate': 'int64',
    'DaysFromIndexToA1cDateAfter12Months': 'int64',
    'DaysFromIndexToFirstEncounterDate': 'int64',
    'DaysFromIndexToLastEncounterDate': 'int64',
    'DaysFromIndexToLatestDate': 'int64',
    'DaysFromIndexToPatientTurns18': 'int64',
    'AgeYears': 'int64',
    'BirthYear': 'int64',
    'NumberEncounters': 'int64',
    'SDI_score': 'float64',
    'Veteran': 'int64'
}

dia_columns = {
    'EMPI': 'int64',
    'Date': 'object', 
    'Code_Type': 'object', 
    'Code': 'object', 
    'IndexDate': 'object', 
    'DiagnosisBeforeOrOnIndexDate': 'object', 
    'CodeWithType': 'object'
}

# Select columns to read in each dataset

outcomes_columns_select = ['studyID', 'EMPI', 'InitialA1c', 'A1cAfter12Months', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English', 'DaysFromIndexToInitialA1cDate', 'DaysFromIndexToA1cDateAfter12Months', 'DaysFromIndexToFirstEncounterDate', 'DaysFromIndexToLastEncounterDate', 'DaysFromIndexToLatestDate', 'DaysFromIndexToPatientTurns18', 'AgeYears', 'BirthYear', 'NumberEncounters', 'SDI_score', 'Veteran']

dia_columns_select = ['EMPI', 'DiagnosisBeforeOrOnIndexDate', 'CodeWithType']

# Define file path and selected columns
outcomes_file_path = 'data/DiabetesOutcomes.txt'
diagnoses_file_path = 'data/Diagnoses.txt'

# Read each file into a DataFrame
df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select)
gc.collect()

# Check and handle duplicates in outcomes data
print("Number of rows before dropping duplicates:", len(df_outcomes))
df_outcomes = df_outcomes.drop_duplicates(subset='EMPI') # keeps first EMPI
print("Number of rows after dropping duplicates:", len(df_outcomes))

print("Total features (original):", df_outcomes.shape[1])

# Optimize dataFrames
df_outcomes = optimize_dataframe(df_outcomes)
df_dia = optimize_dataframe(df_dia)
gc.collect()

# List of columns with known missing values
columns_with_missing_values = ['SDI_score']

with open('columns_with_missing_values.json', 'w') as file:
    json.dump(columns_with_missing_values, file)

# Only replace missing values in these specified columns
for col in columns_with_missing_values:
    df_outcomes.loc[:, col] = df_outcomes[col].fillna(0)

# Handle categorical data in diagnoses data
categorical_columns = df_dia.select_dtypes(include=['object', 'category']).columns.tolist()

# Explicitly add 'CodeWithType' to the list if not already included
if 'CodeWithType' not in categorical_columns:
    categorical_columns.append('CodeWithType')

# Save the list of categorical columns to a JSON file
with open('categorical_columns.json', 'w') as file:
    json.dump(categorical_columns, file)

# Summary statistics for categorical features
categorical_stats = df_dia[categorical_columns].apply(lambda col: col.value_counts(normalize=True))
print(categorical_stats)

# Convert categorical columns to string and limit categories
# filter each categorical column to keep only the top x most frequent categories and replace other categories with a common placeholder like 'Other'
max_categories = 1000  # Adjust this value as needed

# Reduce categories to top 'max_categories' and label others as 'Other'
for col in categorical_columns:
    top_categories = df_dia[col].value_counts().nlargest(max_categories).index
    df_dia[col] = df_dia[col].astype(str)
    df_dia[col] = df_dia[col].where(df_dia[col].isin(top_categories), other='Other')

# Initialize OneHotEncoder with limited categories
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoder.fit(df_dia[categorical_columns].astype(str))

# Process in batches
batch_size = 1000  # Adjust batch size as needed
encoded_batches = []

for start in range(0, len(df_dia), batch_size):
    end = min(start + batch_size, len(df_dia))
    batch = df_dia.iloc[start:end]
    encoded_batch = one_hot_encoder.transform(batch[categorical_columns].astype(str))

    # Get new column names for one-hot encoded variables
    encoded_feature_names = one_hot_encoder.get_feature_names(input_features=categorical_columns)

    # Convert feature names to a more readable format
    encoded_feature_names = [name.replace("x0_", "").replace("_", "-") for name in encoded_feature_names]

    # Convert numpy array to DataFrame with appropriate column names
    encoded_batch_df = pd.DataFrame(encoded_batch, columns=encoded_feature_names, index=batch.index)

    # Add missing value column for each categorical variable
    for col in categorical_columns:
        encoded_batch_df[f'{col}-1'] = (batch[col] == 'Other').astype(int)

    encoded_batches.append(encoded_batch_df)

# Concatenate all encoded batches
encoded_categorical = pd.concat(encoded_batches)

# Combine the original DataFrame with the encoded DataFrame
df_dia = pd.concat([df_dia, encoded_categorical], axis=1)

# Drop the original categorical columns
df_dia.drop(categorical_columns, axis=1, inplace=True)

# Reshape df_dia to have one row per EMPI with one-hot encoded variables as columns

df_dia['count'] = 1 # create a 'count' column to help with pivot_table aggregation

df_dia_wide = pd.pivot_table(df_dia, values='count', index='EMPI', columns='CodeWithType', fill_value=0, aggfunc='max') # pivot table to get one-hot encoding for 'CodeWithType'

df_diagnosis = df_dia.groupby('EMPI')['DiagnosisBeforeOrOnIndexDate'].max().reset_index() # binary indicator aggregated by the maximum value per EMPI, which represents whether any diagnosis was before or on the index date
df_dia_wide = df_dia_wide.merge(df_diagnosis, on='EMPI', how='left')

df_dia_wide.reset_index(inplace=True)  # if 'EMPI' is not already a column

# Merge dataFrames on 'EMPI'

merged_df = df_outcomes.merge(df_dia_wide, on='EMPI', how='outer')
del df_outcomes, df_dia, df_dia_wide
gc.collect()

# Drop 'EMPI'
merged_df.drop(['EMPI'], axis=1, inplace=True)

# Convert all remaining columns to float32
merged_df = merged_df.astype('float32')

# Print column names for verification
print("Column Names after One-Hot Encoding and Adding Missing Value Columns:")
print(merged_df.columns.tolist())

print("Total features (preprocessed):", merged_df.shape[1])

with open('column_names.json', 'w') as file:
    json.dump(merged_df.columns.tolist(), file)

# Convert to PyTorch tensor and save
pytorch_tensor = torch.tensor(merged_df.values, dtype=torch.float32)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')