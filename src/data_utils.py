import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
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
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import argparse
import dask
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, as_completed

def preprocess_data(df_dia, df_prc, df_outcomes):
    # Limit diagnoses and procedures to those on or before patients' index date
    print("Number of diagnoses, unconditional on Index date:", len(df_dia))

    # Keep only rows where 'DiagnosisBeforeOrOnIndexDate' equals 1
    df_dia = df_dia[df_dia['DiagnosisBeforeOrOnIndexDate'] == 1]

    # Drop the 'DiagnosisBeforeOrOnIndexDate' column
    df_dia.drop('DiagnosisBeforeOrOnIndexDate', axis=1, inplace=True)

    print("Number of diagnoses before or on Index date:", len(df_dia))

    # Procedures data processing
    print("Number of procedures, unconditional on Index date:", len(df_prc))

    # Keep only rows where 'ProcedureBeforeOrOnIndexDate' equals 1
    df_prc = df_prc[df_prc['ProcedureBeforeOrOnIndexDate'] == 1]

    # Drop the 'ProcedureBeforeOrOnIndexDate' column
    df_prc.drop('ProcedureBeforeOrOnIndexDate', axis=1, inplace=True)

    print("Number of procedures before or on Index date:", len(df_prc))

    # Outcomes data processing
    print("Number of patients:", len(df_outcomes))
    print("Total features (original):", df_outcomes.shape[1])

    print("Preprocessing outcomes data")

    # Create a new column 'A1cLessThan7' based on the condition
    df_outcomes['A1cLessThan7'] = np.where(df_outcomes['A1cAfter12Months'] < 7, 1, 0)

    # Drop the 'A1cAfter12Months' column
    df_outcomes.drop('A1cAfter12Months', axis=1, inplace=True)

    # Drop the 'BirthYear' column
    df_outcomes.drop('BirthYear', axis=1, inplace=True)

    # Return the modified dataframes
    return df_dia, df_prc, df_outcomes


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