import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset 
import io
from tqdm import tqdm
import numpy as np
import gc
from math import ceil
import json
import re
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_sparse
import logging
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import argparse
import dask
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, as_completed

logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


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

def dask_df_to_tensor(dask_df, chunk_size=1024):
    """
    Convert a Dask DataFrame to a PyTorch tensor in chunks.
    
    Args:
    dask_df (dask.dataframe.DataFrame): The Dask DataFrame to convert.
    chunk_size (int): The size of chunks to use when processing.

    Returns:
    torch.Tensor: The resulting PyTorch tensor.
    """
    # Convert the Dask DataFrame to a Dask Array
    dask_array = dask_df.to_dask_array(lengths=True)

    # Define a function to process a chunk of the array
    def process_chunk(dask_chunk):
        numpy_chunk = dask_chunk.compute()  # Compute the chunk to a NumPy array
        tensor_chunk = torch.from_numpy(numpy_chunk)  # Convert the NumPy array to a PyTorch tensor
        return tensor_chunk

    # Create a list to hold tensor chunks
    tensor_chunks = []

    # Iterate over chunks of the Dask array
    for i in range(0, dask_array.shape[0], chunk_size):
        chunk = dask_array[i:i + chunk_size]
        # Use map_blocks to apply the processing function to each chunk
        tensor_chunk = chunk.map_blocks(process_chunk, dtype=float)
        tensor_chunks.append(tensor_chunk)

    # Use da.concatenate to combine chunks into a single Dask array, then compute
    combined_tensor = da.concatenate(tensor_chunks, axis=0).compute()

    return combined_tensor

# Define the column types for each file type

def main(use_dask=False):
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

    outcomes_columns_select = ['studyID','EMPI', 'InitialA1c', 'A1cAfter12Months', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English','AgeYears', 'BirthYear', 'SDI_score', 'Veteran']

    dia_columns_select = ['EMPI', 'DiagnosisBeforeOrOnIndexDate', 'CodeWithType']

    # Define file path and selected columns
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'

    # Read each file into a DataFrame
    df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select)

   # Check dimensions in diagnoses data
    print("Number of diagnoses, unconditional on Index date:", len(df_dia)) # 8768424

    # Keep only rows where 'DiagnosisBeforeOrOnIndexDate' equals 1
    df_dia = df_dia[df_dia['DiagnosisBeforeOrOnIndexDate'] == 1]

    # Drop the 'DiagnosisBeforeOrOnIndexDate' column
    df_dia.drop('DiagnosisBeforeOrOnIndexDate', axis=1, inplace=True) 

    # Check dimensions in diagnoses data
    print("Number of diagnoses before or on Index date:", len(df_dia)) # 2881686

    # Check dimensions in outcomes data
    print("Number of patients:", len(df_outcomes)) # 55667
    print("Total features (original):", df_outcomes.shape[1]) # 20

    print("Preprocessing outcomes data")

    # Create a new column 'A1cLessThan7' based on the condition
    df_outcomes['A1cLessThan7'] = np.where(df_outcomes['A1cAfter12Months'] < 7, 1, 0)

    # Drop the 'A1cAfter12Months' column
    df_outcomes.drop('A1cAfter12Months', axis=1, inplace=True)

    print("Splitting outcomes data")
    # Splitting the data into train, validation, and test sets
    train_df, temp_df = train_test_split(df_outcomes, test_size=0.3, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)

    del df_outcomes

    print("Training set size:", train_df.shape[0])
    print("Validation set size:", validation_df.shape[0])
    print("Test set size:", test_df.shape[0])

    # Extracting 'EMPI' identifiers for each set
    train_empi = train_df['EMPI']
    validation_empi = validation_df['EMPI']
    test_empi = test_df['EMPI']

    # Save the 'EMPI' identifiers to files
    train_empi.to_csv('train_empi.csv', index=False)
    validation_empi.to_csv('validation_empi.csv', index=False)
    test_empi.to_csv('test_empi.csv', index=False)

    # List of columns with known missing values
    columns_with_missing_values = ['SDI_score']

    with open('columns_with_missing_values.json', 'w') as file:
        json.dump(columns_with_missing_values, file)

    print("Replacing missing values in ", columns_with_missing_values)

    # Create the imputer object
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    # Fit the imputer on the training data and transform the training data
    train_df[columns_with_missing_values] = imputer.fit_transform(train_df[columns_with_missing_values])

    # Apply the same imputation to validation and test datasets without fitting again
    validation_df[columns_with_missing_values] = imputer.transform(validation_df[columns_with_missing_values])
    test_df[columns_with_missing_values] = imputer.transform(test_df[columns_with_missing_values])

    # Normalize specified numeric columns in outcomes data using the Min-Max scaling approach. 
    # Handles negative and zero values well, scaling the data to a [0, 1] range.
    columns_to_normalize = ['InitialA1c', 'AgeYears', 'BirthYear', 'SDI_score']

    print("Normalizing numeric colums: ", columns_to_normalize)

    with open('columns_to_normalize.json', 'w') as file:
        json.dump(columns_to_normalize, file)

    scaler = MinMaxScaler()

    # Fit on training data
    train_df[columns_to_normalize] = scaler.fit_transform(train_df[columns_to_normalize])

    # Transform validation and test data
    validation_df[columns_to_normalize] = scaler.transform(validation_df[columns_to_normalize])
    test_df[columns_to_normalize] = scaler.transform(test_df[columns_to_normalize])

    print("Preprocessing diagnoses data")

    # Handle categorical columns in diagnoses data
    categorical_columns = ['CodeWithType']

    # Save the list of categorical columns to a JSON file
    with open('categorical_columns.json', 'w') as file:
        json.dump(categorical_columns, file)

    print("Converting categorical columns to string and handling missing values.")

    for col in categorical_columns:
        # Convert to string, fill missing values, then convert back to categorical if needed
        df_dia[col] = df_dia[col].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_dia[categorical_columns] .isnull().any().any(), "NaN values found in the diagnoses categorical columns"

    print("Initializing one-hot encoder for diagnoses data.")

    # Splitting the DataFrame while preserving the order
    train_rows = df_dia[df_dia['EMPI'].isin(train_empi)].copy()
    validation_rows = df_dia[df_dia['EMPI'].isin(validation_empi)].copy()
    test_rows = df_dia[df_dia['EMPI'].isin(test_empi)].copy()

    # Add an 'order' column to each subset based on the original DataFrame's index or another unique identifier
    train_rows['order'] = train_rows.index
    validation_rows['order'] = validation_rows.index
    test_rows['order'] = test_rows.index

    # Verify the unique values in the 'CodeWithType' column
    print(f"Unique values in training set 'CodeWithType': {train_rows['CodeWithType'].unique()}")

    # Initialize OneHotEncoder with limited categories and sparse output
    one_hot_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore', min_frequency=ceil(37908*0.01))

    # Fit the encoder on the training subset
    one_hot_encoder.fit(train_rows[categorical_columns].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded = one_hot_encoder.transform(train_rows[categorical_columns].astype(str))
    validation_encoded = one_hot_encoder.transform(validation_rows[categorical_columns].astype(str))
    test_encoded = one_hot_encoder.transform(test_rows[categorical_columns].astype(str))

    print("Extracting infrequent categories.")

    infrequent_categories = one_hot_encoder.infrequent_categories_

    # Prepare a dictionary to hold the infrequent categories for each feature
    infrequent_categories_dict = {
        categorical_columns[i]: list(infrequent_categories[i])
        for i in range(len(categorical_columns))
    }

    # Save the infrequent categories to a JSON file
    with open('infrequent_categories.json', 'w') as f:
        json.dump(infrequent_categories_dict, f)

    print("Infrequent categories saved to infrequent_categories.json.")

    # Get feature names from the encoder
    encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    # Process the feature names to remove the prefix
    encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in encoded_feature_names]

    # Ensure encoded_feature_names is a list
    encoded_feature_names = list(encoded_feature_names)

    print("Combining encoded splits into a single sparse matrix.")

    # Combine the encoded train, validation, and test data into a single sparse matrix
    encoded_data = vstack([train_encoded, validation_encoded, test_encoded])

    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoded_feature_names)

    # Concatenate the 'order' column to encoded_df to preserve original row order
    orders = pd.concat([train_rows['order'], validation_rows['order'], test_rows['order']])
    encoded_df['order'] = orders.values

    # Sort encoded_df by 'order' to match the original df_dia row order and reset the index
    encoded_df.sort_values(by='order', inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)

    # Drop the 'order' column if it's no longer needed
    encoded_df.drop(columns=['order'], inplace=True)

    # Drop the infrequent columns from encoded_df
    infrequent_sklearn_columns = ["infrequent_sklearn"]
    encoded_df = encoded_df.drop(columns=infrequent_sklearn_columns)

    encoded_feature_names = [col for col in encoded_feature_names if col not in infrequent_sklearn_columns]

    with open('encoded_feature_names.json', 'w') as file:
        json.dump(encoded_feature_names, file)

    print("Combining the original DataFrame with the encoded DataFrame.")

    print("Number of one-hot encoded diagnoses:", len(encoded_df)) 
    print("Number of diagnoses prior to concatenatation:", len(df_dia)) 

    # Reset the index of both DataFrames to ensure alignment
    df_dia = df_dia.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    # Verify that the number of rows matches to ensure a logical one-to-one row correspondence
    assert len(df_dia) == len(encoded_df), "Row counts do not match."

    # Check if the indexes are aligned
    assert df_dia.index.equals(encoded_df.index), "Indexes are not aligned."

    df_dia = pd.concat([df_dia, encoded_df], axis=1, sort=False)

    print("Number of diagnoses after concatenatation:", len(df_dia))
    # Verify row counts match expected transformations
    assert len(df_dia) == len(encoded_df), f"Unexpected row count. Expected: {len(encoded_df)}, Found: {len(df_dia)}"
 
    print("Dropping the original categorical columns ", categorical_columns)
    df_dia.drop(categorical_columns, axis=1, inplace=True)

    print("Starting aggregation by EMPI.")

    agg_dict = {col: 'max' for col in encoded_feature_names}
    if use_dask:
        print("Starting aggregation by EMPI directly with Dask DataFrame operations.")
        print("Converting diagnoses to Dask DataFrame.")
        df_dia = dd.from_pandas(df_dia, npartitions=npartitions)

        print("Perform the groupby and aggregation in parallel.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict)

        print("Convert splits to Dask DataFrame.")
        train_df = dd.from_pandas(train_df, npartitions=npartitions)
        validation_df = dd.from_pandas(validation_df, npartitions=npartitions)
        test_df = dd.from_pandas(test_df, npartitions=npartitions)
    else:
        print("Perform groupby and aggregation using pandas.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict)

    print("Number of diagnoses before or on Index , after aggregation:", len(df_dia_agg))

    print("Merge outcomes splits with aggregated diagnoses.")
    merged_train_df = train_df.merge(df_dia_agg, on='EMPI', how='inner')
    merged_validation_df = validation_df.merge(df_dia_agg, on='EMPI', how='inner')
    merged_test_df = test_df.merge(df_dia_agg, on='EMPI', how='inner')

    print("Select numeric columns and drop EMPI using Pandas.")
    numeric_train_df = merged_train_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')
    numeric_validation_df = merged_validation_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')
    numeric_test_df = merged_test_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')

    print("Merged training set size:", numeric_train_df.shape[0])
    print("Merged validation set size:", numeric_validation_df.shape[0])
    print("Merged test set size:", numeric_test_df.shape[0])

    column_names = numeric_train_df.columns.tolist()

    # Save column names to a JSON file
    with open('column_names.json', 'w') as file:
        json.dump(column_names, file)

    print("Clean up unused dataframes.")
    del df_dia, df_dia_agg
    gc.collect()
    
    if use_dask:
        print("Parallel conversion of training set Dask DataFrame to PyTorch tensor.")
        train_tensor = dask_df_to_tensor(numeric_train_df)
        print("Parallel conversion of validation set Dask DataFrame to PyTorch tensor.")
        validation_tensor = dask_df_to_tensor(numeric_validation_df)
        print("Parallel conversion of test set Dask DataFrame to PyTorch tensor.")
        test_tensor = dask_df_to_tensor(numeric_test_df)
        client.close()
    else:
        print("Convert to PyTorch tensors using Pandas.")
        train_tensor = torch.tensor(numeric_train_df.values, dtype=torch.float32)
        validation_tensor = torch.tensor(numeric_validation_df.values, dtype=torch.float32)
        test_tensor = torch.tensor(numeric_test_df.values, dtype=torch.float32)

    print("Wrap tensors in TensorDataset.")
    train_dataset = TensorDataset(train_tensor)
    validation_dataset = TensorDataset(validation_tensor)
    test_dataset = TensorDataset(test_tensor)

    print("Save datasets to file.")
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(validation_dataset, 'validation_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
    args = parser.parse_args()
    if args.use_dask:
        # Adjust timeout settings
        dask.config.set({'distributed.comm.timeouts.connect': '120s'})

        # Determine the total number of available cores
        total_cores = multiprocessing.cpu_count()

        # Don't use all available cores
        n_jobs = total_cores // 2

        npartitions = int(n_jobs*7) # number of partitions for Dask - Aim for partitions with around 100 MB of data each.

        client = Client(processes=True)  # Use processes instead of threads for potentially better CPU utilization
    main(use_dask=args.use_dask)