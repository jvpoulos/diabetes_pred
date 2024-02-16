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
import logging
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import dask.dataframe as dd
from dask.cache import Cache
cache = Cache(1e9)  # 1e9 bytes == 1 GB
cache.register()  # Register cache globally

# Determine the total number of available cores
total_cores = multiprocessing.cpu_count()
print("total_cores:", total_cores)

# Don't use all available cores
n_jobs = total_cores - 12

print("n_jobs:", n_jobs)

logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def df_to_tensor_in_chunks(df, chunk_size=50000):
    """
    Processes a DataFrame in chunks and converts each chunk into a PyTorch tensor.

    Parameters:
    - df: pandas.DataFrame, the DataFrame to process.
    - chunk_size: int, the size of each chunk to process.

    Returns:
    - A PyTorch tensor containing all data from the DataFrame.
    """
    # Placeholder list for tensors
    tensor_list = []

    # Calculate the number of chunks
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

    for chunk_start in range(0, len(df), chunk_size):
        # Determine the end of the chunk
        chunk_end = min(chunk_start + chunk_size, len(df))
        
        # Convert the chunk to a numpy array of type float32
        chunk_array = df.iloc[chunk_start:chunk_end].to_numpy(dtype=np.float32)
        
        # Convert the numpy array to a PyTorch tensor
        chunk_tensor = torch.tensor(chunk_array, dtype=torch.float32)
        
        # Append the tensor to the list
        tensor_list.append(chunk_tensor)
    
    # Concatenate all tensors in the list along the first dimension
    combined_tensor = torch.cat(tensor_list, dim=0)
    
    return combined_tensor

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

def read_file(file_path, columns_type, columns_select, parse_dates=None, chunk_size=50000):
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

def process_batch(start, end, df, categorical_columns, one_hot_encoder):
    batch = df.loc[start:end-1, categorical_columns]
    encoded_batch = one_hot_encoder.transform(batch.astype(str))
    assert encoded_batch.ndim == 2, f"Batch at rows {start} to {end} is not a 2-D array"
    assert encoded_batch.getnnz() > 0, f"All-zero batch at rows {start} to {end}"
    return encoded_batch

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

def optimize_column(df, col, start_idx, end_idx):
    try:
        # Convert categorical columns to numerical codes
        if not pd.api.types.is_numeric_dtype(df[col]):
            df.loc[start_idx:end_idx, col] = df[col].astype('category').cat.codes
        
        # Perform the multiplication
        multiplied_result = df.loc[start_idx:end_idx, col] * df.loc[start_idx:end_idx, 'DiagnosisBeforeOrOnIndexDate']
        
        # Calculate sparsity ratio
        sparsity_ratio = (multiplied_result == 0).mean()

        # If the column is mostly zeros, convert to a sparse type to save memory
        if sparsity_ratio > 0.95:
            # Create a SparseArray and replace the existing column with it
            df[col] = pd.arrays.SparseArray(df[col], fill_value=0, dtype=df[col].dtype)
        else:
            # Directly assign the result to the DataFrame in the specified slice
            df.loc[start_idx:end_idx, col] = multiplied_result
    except Exception as e:
        print(f"Error processing column {col}: {e}")

def process_column(df, col, chunk_size=500000):
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        optimize_column(df, col, chunk_start, chunk_end)

def process_chunk(start, df_dia, encoded_columns, agg_dict, chunk_size=500000):
    end = start + chunk_size
    chunk = df_dia.iloc[start:end].copy()
    for col in encoded_columns:
        if pd.api.types.is_sparse(chunk[col].dtype):
            chunk[col] = chunk[col].sparse.to_dense()
    agg_chunk = chunk.groupby('EMPI').agg(agg_dict).reset_index()
    return agg_chunk

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

# Verify the unique values in the 'CodeWithType' column
print(f"Unique values in 'CodeWithType': {df_dia['CodeWithType'].unique()}")

# Verify the unique values in the 'DiagnosisBeforeOrOnIndexDate' column
print(f"Unique values in 'DiagnosisBeforeOrOnIndexDate': {df_dia['DiagnosisBeforeOrOnIndexDate'].unique()}")

print("Initializing one-hot encoder.")

# First, validate that all categories have at least one value
for col in categorical_columns:
    assert df_dia[col].nunique() > 0, f"Column {col} has no unique values."

# Initialize OneHotEncoder with limited categories and sparse output
one_hot_encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
one_hot_encoder.fit(df_dia[categorical_columns].dropna().astype(str))

print("Processing one-hot encoded features in batches.")

batch_size = 5000
encoded_batches = []

# Execute the processing in parallel
encoded_batches = Parallel(n_jobs=n_jobs)(
    delayed(process_batch)(start, min(start + batch_size, len(df_dia)), df_dia, categorical_columns, one_hot_encoder)
    for start in range(0, len(df_dia), batch_size)
)

# Check if the encoded_batches list is not empty
assert len(encoded_batches) > 0, "No batches have been encoded. The encoded_batches list is empty."

print("Combining batches into a single sparse matrix.")
# Combine all batches into a single sparse matrix
encoded_data = vstack(encoded_batches)

# Get feature names from the encoder
encoded_feature_names = one_hot_encoder.get_feature_names(categorical_columns)

print("Dropping the original categorical columns ", categorical_columns)
# Drop the original categorical columns
df_dia.drop(categorical_columns, axis=1, inplace=True)

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

# Preprocess encoded columns
print("Preprocess encoded columns.")
filtered_columns = [col for col in encoded_columns if col != 'DiagnosisBeforeOrOnIndexDate']
tasks = [delayed(process_column)(df_dia, col) for col in filtered_columns]

if tasks:
    with tqdm(total=len(tasks), desc="Processing Columns") as progress_bar:
        results = Parallel(n_jobs=n_jobs)(tasks)
        for _ in results:
            progress_bar.update()
else:
    print("No columns meet the criteria for processing.")

print("Aggregate one-hot encoded features by EMPI.")
# Setup the aggregation dictionary
agg_dict = {col: 'max' for col in encoded_columns}
print("Aggregation dictionary set up.")
print("Starting aggregation by EMPI.")
df_dia = dd.from_pandas(df_dia, npartitions=n_jobs)
delayed_aggs = [dask.delayed(lambda x: x.groupby('EMPI').agg(agg_dict))(df_dia.get_partition(n)) for n in range(df_dia.npartitions)]
agg_results = dask.compute(*delayed_aggs, scheduler='processes')[0]
df_dia = dd.concat(agg_results).groupby('EMPI').agg(agg_dict).compute()

print("Merge diagnoses and outcomes datasets")
df_outcomes = dd.from_pandas(df_outcomes, npartitions=n_jobs)
df_outcomes = df_outcomes.set_index('EMPI', sorted=True)
merged_df = df_outcomes.join(df_dia, how='outer').compute()

if merged_df.isnull().any().any():
    logging.warning('Merged DataFrame contains null values.')

# Reset index if EMPI is needed as a column
merged_df.reset_index(inplace=True)

del df_outcomes, df_dia
gc.collect()

# Drop unnecessary columns
merged_df.drop(['EMPI', 'DiagnosisBeforeOrOnIndexDate'], axis=1, inplace=True)

# Identify columns with all zeros
columns_with_all_zeros = [col for col in encoded_feature_names if (merged_df[col] == 0).all()]
print(f"Columns with all zeros: {columns_with_all_zeros}")

merged_df.drop(columns=columns_with_all_zeros, inplace=True)

encoded_feature_names = [col for col in encoded_feature_names if col not in columns_with_all_zeros]

with open('columns_with_all_zeros.json', 'w') as file:
    json.dump(columns_with_all_zeros, file)

print("Total patients (preprocessed):", merged_df.shape[0])
print("Total features (preprocessed):", merged_df.shape[1])

with open('encoded_feature_names.json', 'w') as file:
    json.dump(encoded_feature_names, file)

with open('column_names.json', 'w') as file:
    json.dump(merged_df.columns.tolist(), file)

# Convert to PyTorch tensor and save
pytorch_tensor = df_to_tensor_in_chunks(merged_df, chunk_size=50000)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')