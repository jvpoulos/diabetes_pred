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
from scipy.sparse import csr_matrix, hstack

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
df_dia['CodeWithType_Interaction'] = df_dia.apply(
    lambda row: f"{row['CodeWithType']}_BeforeOrOnIndex" if row['DiagnosisBeforeOrOnIndexDate'] == 1 else None,
    axis=1
)

# Handle categorical columns in diagnoses data
categorical_columns = ['CodeWithType', 'CodeWithType_Interaction']

# Save the list of categorical columns to a JSON file
with open('categorical_columns.json', 'w') as file:
    json.dump(categorical_columns, file)

# Convert categorical columns to string and limit categories
# filter each categorical column to keep only the top x most frequent categories and replace other categories with a common placeholder like 'Other'
max_categories = 3000 # will not restrict diagnoses
print("One-hot encoding with a max. of", max_categories, "most frequent categories")

# Reduce categories to top 'max_categories' and label others as 'Other'
for col in categorical_columns:
    top_categories = df_dia[col].value_counts().nlargest(max_categories).index
    df_dia[col] = df_dia[col].astype(str)
    df_dia[col] = df_dia[col].where(df_dia[col].isin(top_categories), other='Other')

# Initialize OneHotEncoder with limited categories and sparse output
one_hot_encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
one_hot_encoder.fit(df_dia[categorical_columns].astype(str))

# Process in batches
batch_size = 1000
encoded_batches = []

for start in range(0, len(df_dia), batch_size):
    end = min(start + batch_size, len(df_dia))
    batch = df_dia.iloc[start:end]
    encoded_batch = one_hot_encoder.transform(batch[categorical_columns].astype(str))

    # Get new column names for one-hot encoded variables
    encoded_feature_names = one_hot_encoder.get_feature_names(categorical_columns)

    # Ensure encoded_feature_names is a list
    encoded_feature_names = list(encoded_feature_names)

    # Proceed with appending new column names for missing value indicators
    for col in categorical_columns:
        missing_value_indicator = (batch[col] == 'Other').astype(int).values.reshape(-1, 1)
        missing_value_column_name = f"{col}_missing_indicator"
        encoded_feature_names.append(missing_value_column_name)
        missing_value_column = csr_matrix(missing_value_indicator)
        encoded_batch = hstack([encoded_batch, missing_value_column])

    encoded_batches.append(encoded_batch)

expected_rows = sum(batch.shape[0] for batch in encoded_batches)
print(f"Expected shape of encoded_categorical: ({expected_rows}, {len(encoded_feature_names)})")

# Concatenate all encoded batches into a single sparse matrix
encoded_categorical = sp.vstack(encoded_batches)

print(f"Shape of encoded_categorical: {encoded_categorical.shape}")

# Convert sparse matrix to DataFrame if needed for further processing
# Note: This step may significantly increase memory usage depending on the sparsity of the matrix
# Ensure the encoded_feature_names list matches the number of columns in encoded_categorical
assert len(encoded_feature_names) == encoded_categorical.shape[1], "Mismatch in number of feature names and columns in encoded matrix"

encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_categorical, columns=encoded_feature_names)

# Combine the original DataFrame with the encoded DataFrame
df_dia = pd.concat([df_dia, encoded_df], axis=1)

# Drop the original categorical columns
df_dia.drop(categorical_columns, axis=1, inplace=True)

# Deduplicate diagnoses so there is one records per patient
df_dia.drop_duplicates(subset='EMPI', inplace=True)

# Merge dataFrames on 'EMPI'

merged_df = df_outcomes.merge(df_dia, on='EMPI', how='outer')
del df_outcomes, df_dia
gc.collect()

# Drop unnecessary columns
merged_df.drop(['EMPI','CodeWithType_missing_indicator', 'CodeWithType_Interaction_missing_indicator', 'CodeWithType__ICD9', ], axis=1, inplace=True)
encoded_feature_names = [name for name in encoded_feature_names if name not in ['CodeWithType_missing_indicator', 'CodeWithType_Interaction_missing_indicator', 'CodeWithType__ICD9']]

# Convert all remaining columns to float32
merged_df = merged_df.astype('float32')

print("Total patients (preprocessed):", merged_df.shape[0])
print("Total features (preprocessed):", merged_df.shape[1])

with open('encoded_feature_names.json', 'w') as file:
    json.dump(encoded_feature_names, file)

with open('column_names.json', 'w') as file:
    json.dump(merged_df.columns.tolist(), file)

# Convert to PyTorch tensor and save
pytorch_tensor = torch.tensor(merged_df.values, dtype=torch.float32)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')