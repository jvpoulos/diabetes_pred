import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import io
from tqdm import tqdm
import numpy as np
import gc
import json
import re
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
from pandas.api.types import is_categorical_dtype, is_sparse


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

def optimize_column(df, col):
    try:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes.astype('int8', copy=False)

        # Multiply directly for both sparse and non-sparse to keep the operation efficient
        multiplied_result = df[col].multiply(df['DiagnosisBeforeOrOnIndexDate'], axis=0, fill_value=0)
        
        # Determine if the result should be kept sparse
        if isinstance(multiplied_result, pd.Series):
            sparsity_ratio = (multiplied_result == 0).mean()
            if sparsity_ratio > 0.95:  # Adjust this threshold based on your dataset
                df[col] = pd.arrays.SparseArray(multiplied_result, fill_value=0, dtype='float32')
            else:
                df[col] = multiplied_result
        else:
            df[col] = multiplied_result

    except Exception as e:
        print(f"Error processing column {col}: {e}")

    gc.collect()

# Define the column types for each file type

outcomes_columns = {
    'studyID': 'int32',
    'EMPI': 'int32',
    'IndexDate': 'object',
    'InitialA1c': 'float32',
    'A1cAfter12Months': 'float32',
    'A1cGreaterThan7': 'int32',
    'Female': 'int32',
    'Married': 'int32',
    'GovIns': 'int32',
    'English': 'int32',
    'DaysFromIndexToInitialA1cDate': 'int32',
    'DaysFromIndexToA1cDateAfter12Months': 'int32',
    'DaysFromIndexToFirstEncounterDate': 'int32',
    'DaysFromIndexToLastEncounterDate': 'int32',
    'DaysFromIndexToLatestDate': 'int32',
    'DaysFromIndexToPatientTurns18': 'int32',
    'AgeYears': 'int32',
    'BirthYear': 'int32',
    'NumberEncounters': 'int32',
    'SDI_score': 'float32',
    'Veteran': 'int32'
}

dia_columns = {
    'EMPI': 'int32',
    'Date': 'object', 
    'Code_Type': 'object', 
    'Code': 'object', 
    'IndexDate': 'object', 
    'DiagnosisBeforeOrOnIndexDate': 'int32', 
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

# Check dimensions in outcomes data
print("Number of patients:", len(df_outcomes)) # 55667
print("Total features (original):", df_outcomes.shape[1]) # 20

# Optimize dataFrames
df_outcomes = optimize_dataframe(df_outcomes)
df_dia = optimize_dataframe(df_dia)
gc.collect()

# List of columns with known missing values
columns_with_missing_values = ['SDI_score']

with open('columns_with_missing_values.json', 'w') as file:
    json.dump(columns_with_missing_values, file)

print("Replacing missing values in ", columns_with_missing_values)

# Only replace missing values in these specified columns
for col in columns_with_missing_values:
    df_outcomes.loc[:, col] = df_outcomes[col].fillna(0)

# Create categorical feature for 'DiagnosisBeforeOrOnIndexDate' based on interaction with CodeWithType
df_dia['Date'] = df_dia.apply(
    lambda row: row['CodeWithType'] if row['DiagnosisBeforeOrOnIndexDate'] == 1 else None,
    axis=1
)

# Handle categorical columns in diagnoses data
categorical_columns = ['CodeWithType', 'Date']

# Save the list of categorical columns to a JSON file
with open('categorical_columns.json', 'w') as file:
    json.dump(categorical_columns, file)

print("Converting categorical columns to string and handling missing values.")

for col in categorical_columns:
    # Convert to string, fill missing values, then convert back to categorical if needed
    df_dia[col] = df_dia[col].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
 
# Verify no NaN values exist
assert not df_dia[categorical_columns] .isnull().any().any(), "NaN values found in the diagnoses categorical columns"


# Verify the unique values in the 'DiagnosisBeforeOrOnIndexDate' column
print(f"Unique values in 'DiagnosisBeforeOrOnIndexDate': {df_dia['DiagnosisBeforeOrOnIndexDate'].unique()}")

# Print out the unique values of 'CodeWithType' for a sample of data
print(f"Sample unique values in 'CodeWithType': {df_dia['CodeWithType'].sample(5).unique()}")

print("Initializing one-hot encoder.")

# First, validate that all categories have at least one value
for col in categorical_columns:
    assert df_dia[col].nunique() > 0, f"Column {col} has no unique values."

# Initialize OneHotEncoder with limited categories and sparse output
one_hot_encoder = OneHotEncoder(sparse=True, handle_unknown='error')
one_hot_encoder.fit(df_dia[categorical_columns].dropna().astype(str))

print("Processing one-hot encoded features in batches.")
# Process in batches
batch_size = 1000
encoded_batches = []

for start in range(0, len(df_dia), batch_size):
    end = min(start + batch_size, len(df_dia))
    batch = df_dia.loc[start:end-1, categorical_columns]
    encoded_batch = one_hot_encoder.transform(batch.astype(str))
    
    # Ensure that the encoded batch is a 2-D array
    assert encoded_batch.ndim == 2, f"Batch at rows {start} to {end} is not a 2-D array"
    
    # Check if the encoded batch is not all zeros
    assert encoded_batch.getnnz() > 0, f"All-zero batch at rows {start} to {end}"
    
    encoded_batches.append(encoded_batch)

# Check if the encoded_batches list is not empty
assert len(encoded_batches) > 0, "No batches have been encoded. The encoded_batches list is empty."

# After batch processing, inspect one of the batches
print(f"Sample encoded batch: {encoded_batches[0].toarray()[:5]}")

print("Combining batches into a single sparse matrix.")
# Combine all batches into a single sparse matrix
encoded_data = vstack(encoded_batches)

# Get feature names from the encoder
encoded_feature_names = one_hot_encoder.get_feature_names(categorical_columns)

# Process the feature names to remove the prefix
encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in encoded_feature_names]

# Ensure encoded_feature_names is a list
encoded_feature_names = list(encoded_feature_names)

expected_rows = sum(batch.shape[0] for batch in encoded_batches)
print(f"Expected shape of encoded_categorical: ({expected_rows}, {len(encoded_feature_names)})")

# Concatenate all encoded batches into a single sparse matrix
encoded_categorical = sp.vstack(encoded_batches)

print(f"Shape of encoded_categorical: {encoded_categorical.shape}")

# Convert sparse matrix to DataFrame if needed for further processing
assert len(encoded_feature_names) == encoded_categorical.shape[1], "Mismatch in number of feature names and columns in encoded matrix"

print("Combining batches into a single sparse matrix.")
# Convert to DataFrame for further operations if necessary

encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_categorical, columns=encoded_feature_names)

# After creating encoded_df, inspect if it has all zero columns
non_zero_columns = encoded_categorical.getnnz(axis=0) > 0  # This returns a boolean array
all_zero_columns = ~non_zero_columns  # Invert to get all-zero columns

# Use the boolean index to filter column names
all_zero_column_names = [encoded_feature_names[i] for i, is_zero in enumerate(all_zero_columns) if is_zero]

print(f"Columns with all zeros: {all_zero_column_names}")

print("Combining the original DataFrame with the encoded DataFrame.")

# Concatenate using a method that preserves the sparse structure
df_dia = pd.concat([df_dia, encoded_df], axis=1, sort=False)

print("Binarizing one-hot encoded variables based on 'DiagnosisBeforeOrOnIndexDate'.")

# Identify all one-hot encoded columns for 'Date'
encoded_columns = [col for col in df_dia.columns if 'Date' in col]

# Convert 'DiagnosisBeforeOrOnIndexDate' to a numeric type if it's categorical
if is_categorical_dtype(df_dia['DiagnosisBeforeOrOnIndexDate']):
    df_dia['DiagnosisBeforeOrOnIndexDate'] = df_dia['DiagnosisBeforeOrOnIndexDate'].astype('int8', copy=False)

# Before processing, ensure df_dia is optimized for memory usage
df_dia = optimize_dataframe(df_dia)  # This function needs to ensure optimal memory usage

print("Preprocess encoded columns.")
# Process each encoded column, skipping 'DiagnosisBeforeOrOnIndexDate'
for col in encoded_columns:
    if col != 'DiagnosisBeforeOrOnIndexDate':
        optimize_column(df_dia, col)

# After binarization, inspect the result
print(f"Sample data after binarization: {df_dia[encoded_columns].sample(5)}")

# Identify columns with missing values in df_dia for the specified encoded_feature_names
# Initialize an empty list to hold the names of columns with missing values
columns_with_missing_values = []

# Iterate over each column in encoded_feature_names to check for missing values
for col in encoded_feature_names:
    # If the column has any missing values, append it to the list
    if df_dia[col].isnull().any():
        columns_with_missing_values.append(col)

# Print columns with missing values before filling them
print("Columns with missing values before filling:", columns_with_missing_values)

# Fill missing values with 0
df_dia[columns_with_missing_values] = df_dia[columns_with_missing_values].fillna(0)

# Verify by printing the means of the columns to ensure there are no NaN values
means_after_filling = df_dia[encoded_feature_names].mean()
print("Means of columns after filling missing values with 0:\n", means_after_filling)
        
# Drop the original categorical columns
df_dia.drop(categorical_columns, axis=1, inplace=True)

# Deduplicate diagnoses so there is one record per patient
df_dia.drop_duplicates(subset='EMPI', inplace=True)

assert not df_dia.isnull().any().any(), "NaN values found in the diagnoses DataFrame"

# Merge dataFrames on 'EMPI'

merged_df = df_outcomes.merge(df_dia, on='EMPI', how='outer')
del df_outcomes, df_dia
gc.collect()

# Drop unnecessary columns
merged_df.drop(['EMPI', "studyID", 'DiagnosisBeforeOrOnIndexDate'], axis=1, inplace=True)

# Fill missing values with a specified value
merged_df.fillna(0, inplace=True)

# Filter columns with all zeros
columns_with_all_zeros = [col for col in encoded_feature_names if (merged_df[col] == 0).all()]

merged_df.drop(columns=columns_with_all_zeros, inplace=True)

encoded_feature_names = [col for col in encoded_feature_names if col not in columns_with_all_zeros]

with open('columns_with_all_zeros.json', 'w') as file:
    json.dump(columns_with_all_zeros, file)

assert not merged_df.isnull().any().any(), "NaN values found in the merged DataFrame"

print("Total patients (preprocessed):", merged_df.shape[0])
print("Total features (preprocessed):", merged_df.shape[1])

with open('encoded_feature_names.json', 'w') as file:
    json.dump(encoded_feature_names, file)

with open('column_names.json', 'w') as file:
    json.dump(merged_df.columns.tolist(), file)

# Convert to PyTorch tensor and save
pytorch_tensor = torch.tensor(merged_df.values, dtype=torch.float32)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')