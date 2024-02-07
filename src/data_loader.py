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
df_dia['CodeWithType_Date'] = df_dia.apply(
    lambda row: row['CodeWithType'] if row['DiagnosisBeforeOrOnIndexDate'] == 1 else None,
    axis=1
)

# Handle categorical columns in diagnoses data
categorical_columns = ['CodeWithType', 'CodeWithType_Date']

# Save the list of categorical columns to a JSON file
with open('categorical_columns.json', 'w') as file:
    json.dump(categorical_columns, file)

# Convert categorical columns to string and limit categories
print("Reducing categories to those appearing at least once.")

for col in categorical_columns:
    # Calculate the frequency of each category
    category_counts = df_dia[col].value_counts()
    
    # Identify categories that appear at least once
    categories_to_keep = category_counts[category_counts >= 1].index
    
    # Convert the column to string type
    df_dia[col] = df_dia[col].astype(str)
    
    # Keep categories that appear at least once, label others as 'Other'
    df_dia[col] = df_dia[col].where(df_dia[col].isin(categories_to_keep), other='Other')

# Preprocess to replace NaN and empty strings
df_dia[categorical_columns] = df_dia[categorical_columns].fillna('missing').replace({'': 'missing'})

# Initialize OneHotEncoder with limited categories and sparse output
one_hot_encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
# Fit the encoder to the preprocessed data
one_hot_encoder.fit(df_dia[categorical_columns].astype(str))

# Process in batches
batch_size = 1000
encoded_batches = []

for start in range(0, len(df_dia), batch_size):
    end = min(start + batch_size, len(df_dia))
    batch = df_dia.loc[start:end-1, categorical_columns].fillna('missing').replace({'': 'missing'})
    encoded_batch = one_hot_encoder.transform(batch.astype(str))
    encoded_batches.append(encoded_batch)

# Combine all batches into a single sparse matrix
encoded_data = vstack(encoded_batches)

# Get feature names from the encoder
encoded_feature_names = one_hot_encoder.get_feature_names(categorical_columns)

# Ensure encoded_feature_names is a list
encoded_feature_names = list(encoded_feature_names)

# Convert to DataFrame for further operations if necessary
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoded_feature_names)

# Verify no NaN values exist
#assert not encoded_df.isnull().any().any(), "NaN values found in the encoded DataFrame"

expected_rows = sum(batch.shape[0] for batch in encoded_batches)
print(f"Expected shape of encoded_categorical: ({expected_rows}, {len(encoded_feature_names)})")

# Concatenate all encoded batches into a single sparse matrix
encoded_categorical = sp.vstack(encoded_batches)

print(f"Shape of encoded_categorical: {encoded_categorical.shape}")

# Convert sparse matrix to DataFrame if needed for further processing
#assert len(encoded_feature_names) == encoded_categorical.shape[1], "Mismatch in number of feature names and columns in encoded matrix"

encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_categorical, columns=encoded_feature_names)

#assert not encoded_df.isnull().any().any(), "NaN values found in the encoded DataFrame"

# Combine the original DataFrame with the encoded DataFrame
df_dia = pd.concat([df_dia, encoded_df], axis=1)

# Binarize one-hot encoded variables based on 'DiagnosisBeforeOrOnIndexDate'

# Identify all one-hot encoded columns for 'CodeWithType_Date'
encoded_columns = [col for col in df_dia.columns if 'CodeWithType_Date' in col]

# Create a mask where 'DiagnosisBeforeOrOnIndexDate' is 0, which should apply to all encoded columns simultaneously
mask = df_dia['DiagnosisBeforeOrOnIndexDate'] == 0

for col in encoded_columns:
    print(f"Processing column: {col}, type: {type(df_dia[col])}")
    try:
        # Retrieve the series from df_dia DataFrame
        series = df_dia[col]
        
        # Check if the series is of a sparse data type
        if pd.api.types.is_sparse(series.dtype):
            print(f"Processing column: {col}")  # Debug print
            
            # Convert the sparse series to a dense format
            dense_array = series.sparse.to_dense()

            # Apply the mask. Ensure the mask is correctly aligned with the DataFrame's index
            updated_data = np.where(mask, 0, dense_array)

            # Convert the updated dense data back to a SparseArray with the original fill_value
            fill_value = series.sparse.fill_value
            df_dia[col] = pd.arrays.SparseArray(updated_data, fill_value=fill_value, dtype='float32')
        else:
            print(f"Column {col} is not sparse. Skipping.")  # Debug print for non-sparse columns
    except AttributeError as e:
        print(f"Error processing column {col}: {e}")

# Identify columns with missing values in df_dia for the specified encoded_feature_names
columns_with_missing_values = df_dia[encoded_feature_names].columns[df_dia[encoded_feature_names].isnull().any()].tolist()

# Print columns with missing values before filling them
print("Columns with missing values before filling:", columns_with_missing_values)

# Fill missing values with 0
df_dia[columns_with_missing_values] = df_dia[columns_with_missing_values].fillna(0)

# Verify by printing the means of the columns to ensure there are no NaN values
means_after_filling = df_dia[encoded_feature_names].mean()
print("Means of columns after filling missing values with 0:\n", means_after_filling)
        
# Drop the original categorical columns
df_dia.drop(categorical_columns, axis=1, inplace=True)

# Deduplicate diagnoses so there is one records per patient
df_dia.drop_duplicates(subset='EMPI', inplace=True)

#assert not df_dia.isnull().any().any(), "NaN values found in the diagnoses DataFrame"

# Merge dataFrames on 'EMPI'

merged_df = df_outcomes.merge(df_dia, on='EMPI', how='outer')
del df_outcomes, df_dia
gc.collect()

# Drop unnecessary columns
merged_df.drop(['EMPI', "studyID", 'DiagnosisBeforeOrOnIndexDate','CodeWithType__ICD9', 'CodeWithType_Date__ICD9'], axis=1, inplace=True)
encoded_feature_names = [name for name in encoded_feature_names if name not in ['CodeWithType__ICD9','CodeWithType_Date__ICD9']]

# Fill missing values with a specified value
merged_df.fillna(0, inplace=True)

# # Drop one-hot encoded columns with all zeros
# for col in encoded_feature_names:
#     try:
#         # Accessing a single column as a Series
#         series = merged_df[col]
        
#         # Now series.dtype is valid, and series should not inadvertently be a DataFrame
#         if pd.api.types.is_sparse(series.dtype):
#             print(f"Processing sparse column: {col}")
#         else:
#             print(f"Processing non-sparse column: {col}")

#         # No error should be raised by this line if col is correctly a single column name
#         if (merged_df[col] == 0).all():
#             print(f"Column {col} has all zeros.")
#     except Exception as e:
#         print(f"Error processing column {col}: {e}")

# Filter columns with all zeros
columns_with_all_zeros = [col for col in encoded_feature_names if (merged_df[col] == 0).all()]
print("Columns with all zeros:", columns_with_all_zeros)

merged_df.drop(columns=columns_with_all_zeros, inplace=True)

encoded_feature_names = [col for col in encoded_feature_names if col not in columns_with_all_zeros]

print("Dropped columns with all zeros:", columns_with_all_zeros)

#assert not merged_df.isnull().any().any(), "NaN values found in the merged DataFrame"

print("Total patients (preprocessed):", merged_df.shape[0])
print("Total features (preprocessed):", merged_df.shape[1])

with open('encoded_feature_names.json', 'w') as file:
    json.dump(encoded_feature_names, file)

with open('column_names.json', 'w') as file:
    json.dump(merged_df.columns.tolist(), file)

# Convert to PyTorch tensor and save
pytorch_tensor = torch.tensor(merged_df.values, dtype=torch.float32)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')