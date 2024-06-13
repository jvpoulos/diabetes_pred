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
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix
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
import polars as pl
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.data.dataset_config import DatasetConfig
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import tempfile
import shutil
import pickle

def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Type {type(obj)} not serializable')

def add_to_container(key: str, val: any, cont: dict[str, any]):
    if key in cont:
        if cont[key] == val:
            print(f"WARNING: {key} is specified twice with value {val}.")
        else:
            raise ValueError(f"{key} is specified twice ({val} v. {cont[key]})")
    else:
        cont[key] = val

def read_parquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df

# Create a function to generate time intervals
def generate_time_intervals(start_date, end_date, interval_days):
    current_date = start_date
    intervals = []
    while current_date < end_date:
        next_date = current_date + timedelta(days=interval_days)
        intervals.append((current_date, next_date))
        current_date = next_date
    return intervals

class CustomPytorchDataset(PytorchDataset):
    def __init__(self, config: DatasetConfig, split: str, dl_reps_dir: str, task_df: pl.DataFrame = None):
        self.dl_reps_dir = Path(dl_reps_dir)  # Convert dl_reps_dir to a Path object
        super().__init__(config, split, task_df=task_df, dl_reps_dir=self.dl_reps_dir)
        self.load_cached_data()

    def load_cached_data(self):
        if not self.dl_reps_dir:
            raise ValueError("The 'dl_reps_dir' attribute must be set to a valid directory path.")
        
        cached_path = self.dl_reps_dir / f"{self.split}*.parquet"
        parquet_files = list(cached_path.parent.glob(cached_path.name))

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found for split '{self.split}' in directory '{self.dl_reps_dir}'")

        print(f"Loading Parquet files for split '{self.split}':")
        for parquet_file in parquet_files:
            print(f"File: {parquet_file}")
            try:
                df = pd.read_parquet(parquet_file)
                print(df.head())  # Print the first few rows of each Parquet file
                print(df.dtypes)  # Print the data types of each column
            except Exception as e:
                print(f"Error reading Parquet file: {parquet_file}")
                print(f"Error message: {str(e)}")
                continue

        self.cached_data = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

        # Ensure all required columns are present and handle missing values
        required_columns = ["dynamic_indices", "dynamic_counts"]
        for col in required_columns:
            if col not in self.cached_data.columns:
                self.cached_data[col] = None

        # Handle categorical columns separately
        categorical_columns = ["dynamic_indices_event_type"]
        for col in categorical_columns:
            if col in self.cached_data.columns:
                self.cached_data[col] = self.cached_data[col].apply(lambda x: x.to_list() if isinstance(x, list) else [])

        # Ensure no None values are present in the DataFrame to avoid unwrap errors
        self.cached_data = self.cached_data.fillna({
            col: 0 if self.cached_data[col].dtype in [np.float64, np.float32, np.int64, np.int32] else ""
            for col in self.cached_data.columns if col not in categorical_columns
        })

        print(f"Loaded cached data from {self.dl_reps_dir} for split '{self.split}'.")
        print(f"Cached data shape: {self.cached_data.shape}")
        print(f"Cached data columns: {self.cached_data.columns}")

def save_plot(data, x_col, y_col, gender_col, title, y_label, x_range, file_path):
    """
    Creates a scatter plot with gender-based coloring and saves it to a file.

    Parameters:
    - data: DataFrame with the data to plot.
    - x_col: Column name to use for the x-axis.
    - y_col: Column name to use for the y-axis.
    - gender_col: Column name for gender representation.
    - title: Title for the plot.
    - y_label: Label for the y-axis.
    - x_range: Tuple specifying the range of the x-axis (min, max).
    - file_path: Path to save the plot as a PNG file.
    """
    # Map gender to "Female" or "Male"
    data[gender_col] = data[gender_col].map({1: "Female", 0: "Male"})

    # Create the scatter plot
    fig = px.scatter(data, x=x_col, y=y_col, color=gender_col,
                     title=title,
                     labels={x_col: 'Age', y_col: y_label, gender_col: 'Gender'},
                     range_x=x_range)

    # Save to file
    fig.write_image(file_path, format="png", width=600, height=350, scale=2)

def save_plot_line(df, x, y, filename, xlabel, ylabel, title, x_range=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x], df[y])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=90)

    # Add vertical dashed lines
    date1 = mdates.date2num(datetime(2000, 1, 1))
    date2 = mdates.date2num(datetime(2022, 6, 30))
    ax.axvline(date1, color='r', linestyle='--', label='Start of Study')
    ax.axvline(date2, color='r', linestyle='-', label='End of Study')
    ax.legend()

    if x_range:
        ax.set_xlim(x_range)

    plt.savefig(f'data_summaries/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def temporal_dist_pd(df, measure, event_type_col=None):
    df = df.sort('timestamp').to_pandas()
    df['day'] = df['timestamp'].dt.date
    if event_type_col is None:
        daily_counts = df.groupby('day')[measure].count().reset_index()
        daily_counts.columns = ['day', 'count']
    else:
        daily_counts = df.groupby(['day', event_type_col])[measure].count().reset_index()
        if event_type_col == 'CodeWithType':
            daily_counts.columns = ['day', 'CodeWithType', 'count']
        else:
            daily_counts.columns = ['day', event_type_col, 'count']
    return daily_counts

def save_scatter_plot(data, x_col, y_col, title, y_label, file_path, x_range=None, x_label=None, x_scale=None):
    """
    Creates a scatter plot and saves it to a file.

    Parameters:
    - data: DataFrame with the data to plot.
    - x_col: Column name to use for the x-axis.
    - y_col: Column name to use for the y-axis.
    - title: Title for the plot.
    - y_label: Label for the y-axis.
    - file_path: Path to save the plot as a PNG file.
    - x_range: Tuple specifying the range of the x-axis (min, max). Default is None.
    - x_label: Label for the x-axis. Default is None.
    - x_scale: Tuple specifying the scale for the x-axis (min, max). Default is None.
    """
    # Create the scatter plot
    fig = px.scatter(data, x=x_col, y=y_col, title=title, labels={x_col: x_label or x_col, y_col: y_label})

    # Set the x-axis range if provided
    if x_range is not None:
        fig.update_layout(xaxis_range=x_range)

    # Set the x-axis scale if provided
    if x_scale is not None:
        fig.update_xaxes(range=x_scale)

    # Save to file
    fig.write_image(file_path, format="png", width=600, height=350, scale=2)
    
def save_plot_measurements_per_subject(data, x_col, gender_col, title, x_label, file_path):
    """
    Creates a box plot for the distribution of measurements per subject, grouped by gender, and saves it to a file.

    Parameters:
    - data: DataFrame with the data to plot.
    - x_col: Column name to use for the x-axis (number of measurements).
    - gender_col: Column name for gender representation.
    - title: Title for the plot.
    - x_label: Label for the x-axis.
    - file_path: Path to save the plot as a PNG file.
    """
    # Map gender to "Female" or "Male"
    data[gender_col] = data[gender_col].map({1: "Female", 0: "Male"})

    # Create the box plot with specified colors
    fig = px.box(data, x=x_col, color=gender_col, title=title, labels={x_col: x_label, gender_col: 'Gender'},
                 color_discrete_map={"Female": "blue", "Male": "red"})

    # Save to file
    fig.write_image(file_path, format="png", width=600, height=350, scale=2)

def save_plot_heatmap(data, x_col, y_col, title, file_path):
    """
    Creates a heatmap plot and saves it to a file.

    Parameters:
    - data: DataFrame or 2D array with the data to plot.
    - x_col: Column names to use for the x-axis.
    - y_col: Column names to use for the y-axis.
    - title: Title for the plot.
    - file_path: Path to save the plot as a PNG file.
    """
    # Create the heatmap plot
    fig = px.imshow(data, x=x_col, y=y_col, color_continuous_scale='RdBu', zmin=-1, zmax=1, title=title)

    # Save to file
    fig.write_image(file_path, format="png", width=600, height=350, scale=2)

def print_covariate_metadata(covariates, ESD):
    """
    Prints the measurement metadata for each covariate in the given list.
    
    Parameters:
    - covariates: List of covariates to print metadata for.
    - ESD: An object or module that contains the measurement configs.
    """
    for covariate in covariates:
        if covariate in ESD.measurement_configs:
            metadata = ESD.measurement_configs[covariate].measurement_metadata
            print(f"{covariate} Metadata:", metadata)
        else:
            print(f"{covariate} is not available in measurement configs.")

def preprocess_dataframe(df_name, file_path, columns, selected_columns, chunk_size=50000, min_frequency=None):
    df = read_file(file_path, columns, selected_columns, chunk_size=chunk_size)
    
    print(f"{df_name} DataFrame shape: {df.shape}")
    
    df = pl.from_pandas(df)
    
    print(f"{df_name} Polars DataFrame shape: {df.shape}")
    
    print(f"Preprocess {df_name.lower()} data")
    print(f"Original {df_name} DataFrame shape: {df.shape}")
    
    # Convert 'A1cGreaterThan7' to a float
    if df_name == 'Outcomes':
        df = df.with_columns(pl.col('A1cGreaterThan7').cast(pl.Float64))
    
    if df_name in ['Diagnoses', 'Procedures', 'Labs']:
        # Parse the 'Date' column as datetime
        df = df.with_columns(pl.col('Date').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    
    if df_name in ['Diagnoses', 'Procedures']:
        # Drop rows with missing/null values in Date or CodeWithType
        df = df.drop_nulls(subset=['Date', 'CodeWithType'])
        
        if min_frequency is not None:
            # Count the occurrences of each CodeWithType
            code_counts = df.select('CodeWithType').groupby('CodeWithType').count().sort('count', descending=True)
            
            # Determine the infrequent categories based on min_frequency
            if isinstance(min_frequency, int):
                infrequent_categories = code_counts.filter(pl.col('count') < min_frequency)['CodeWithType']
            else:
                min_count_threshold = min_frequency * len(df)
                infrequent_categories = code_counts.filter(pl.col('count') < min_count_threshold)['CodeWithType']
            
            # Filter out infrequent categories
            df = df.filter(~pl.col('CodeWithType').is_in(infrequent_categories))
    
    elif df_name == 'Labs':
        # Drop rows with missing/null values in Date, Code, or Result
        df = df.drop_nulls(subset=['Date', 'Code', 'Result'])
        
        if min_frequency is not None:
            # Count the occurrences of each CodeWithType
            code_counts = df.select('Code').groupby('Code').count().sort('count', descending=True)
            
            # Determine the infrequent categories based on min_frequency
            if isinstance(min_frequency, int):
                infrequent_categories = code_counts.filter(pl.col('count') < min_frequency)['Code']
            else:
                min_count_threshold = min_frequency * len(df)
                infrequent_categories = code_counts.filter(pl.col('count') < min_count_threshold)['Code']
            
            # Filter out infrequent categories
            df = df.filter(~pl.col('Code').is_in(infrequent_categories))
    
    if min_frequency is not None:
        print(f"Reduced {df_name} DataFrame shape: {df.shape}")
    
    return df

def custom_one_hot_encoder(df, scaler=None, fit=True, use_dask=False, chunk_size=10000, min_frequency=None):
    """
    One-hot encodes the 'Code' column, while scaling the 'Result' column.
    The processing can be done using Dask for parallel computation or in chunks to avoid OOM errors.
    """
    if use_dask:
        # Convert the input DataFrame to a Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=len(df) // chunk_size + 1)

        # Create a Dask Series for 'Code' and 'Result' columns
        code_series = ddf['Code'].astype(str)
        result_series = ddf['Result']

        # Compute the category frequencies
        category_counts = code_series.value_counts().compute()

        # Determine the infrequent categories based on min_frequency
        if min_frequency is not None:
            if isinstance(min_frequency, int):
                infrequent_categories = category_counts[category_counts < min_frequency].index
            else:
                infrequent_categories = category_counts[category_counts < min_frequency * len(df)].index
            code_series = code_series.map_partitions(lambda x: x.where(~x.isin(infrequent_categories), 'infrequent_sklearn'))

        # Create a Dask DataFrame for the encoded features
        encoded_features = code_series.map_partitions(
            lambda x: pd.get_dummies(x, sparse=True),
            meta=pd.DataFrame(columns=[], dtype=float),
            token='encode'
        )

        # Get the feature names from the encoded columns
        feature_names = encoded_features.columns.tolist()

        # Scale the 'Result' column if scaler is provided
        if scaler is not None:
            if fit:
                def fit_scaler(partition):
                    result_values = partition.values
                    result_values = np.nan_to_num(result_values)  # Replace NaN with 0
                    scaler.fit(result_values.reshape(-1, 1))
                    return scaler

                fitted_scaler = result_series.map_partitions(fit_scaler).compute()

                def transform_scaler(partition):
                    result_values = partition.values
                    result_values = np.nan_to_num(result_values)  # Replace NaN with 0
                    return fitted_scaler.transform(result_values.reshape(-1, 1)).flatten()

                result_series = result_series.map_partitions(transform_scaler, meta=float)
            else:
                def transform_scaler(partition):
                    result_values = partition.values
                    result_values = np.nan_to_num(result_values)  # Replace NaN with 0
                    return scaler.transform(result_values.reshape(-1, 1)).flatten()

                result_series = result_series.map_partitions(transform_scaler, meta=float)

        # Multiply the encoded features with the scaled 'Result' column
        def multiply_features(encoded_partition, result_partition):
            encoded_partition = encoded_partition.multiply(result_partition.values[:, None], axis=0)
            return encoded_partition

        encoded_features = encoded_features.map_partitions(multiply_features, result_series, meta=encoded_features._meta)

        # Concatenate the original DataFrame with the encoded features
        encoded_df = dd.concat([ddf[['EMPI', 'Code', 'Result']], encoded_features], axis=1)

        # Return the encoded Dask DataFrame, the fitted scaler if fit=True, and the feature names
        if fit:
            return encoded_df, fitted_scaler, feature_names
        else:
            return encoded_df, feature_names
    else:
        # Initialize the fitted scaler
        fitted_scaler = None

        # Compute the category frequencies
        category_counts = df['Code'].value_counts()

        # Determine the infrequent categories based on min_frequency
        if min_frequency is not None:
            if isinstance(min_frequency, int):
                infrequent_categories = category_counts[category_counts < min_frequency].index
            else:
                infrequent_categories = category_counts[category_counts < min_frequency * len(df)].index
        else:
            infrequent_categories = []

        # Initialize an empty list to store the encoded chunks
        encoded_chunks = []
        feature_names = []

        # Iterate over chunks of the DataFrame
        for start in range(0, len(df), chunk_size):
            end = start + chunk_size
            chunk = df.iloc[start:end].copy()  # Make a copy to avoid SettingWithCopyWarning

            # Convert 'Code' to string type to handle missing values
            chunk['Code'] = chunk['Code'].astype(str)

            # Replace infrequent categories with 'infrequent_sklearn'
            chunk['Code'] = chunk['Code'].where(~chunk['Code'].isin(infrequent_categories), 'infrequent_sklearn')

            # Replace NaN values in 'Result' column with 0
            chunk['Result'] = chunk['Result'].fillna(0)

            # Get the unique codes in the current chunk
            unique_codes_chunk = chunk['Code'].unique()

            # Convert 'Result' to a NumPy array
            results = chunk['Result'].values

            # Initialize a sparse matrix to store the one-hot encoded features
            encoded_features = lil_matrix((len(chunk), len(unique_codes_chunk)), dtype=float)

            # Loop over each unique code in the chunk
            for i, code in enumerate(unique_codes_chunk):
                # Create a boolean mask for the current code
                mask = (chunk['Code'] == code)

                # Extract the 'Result' values for the current code
                code_results = results[mask]

                # Scale the results if scaler is provided
                if scaler is not None:
                    if fit:
                        fitted_scaler = scaler.fit(code_results.reshape(-1, 1))
                        code_results = fitted_scaler.transform(code_results.reshape(-1, 1)).ravel()
                    else:
                        code_results = scaler.transform(code_results.reshape(-1, 1)).ravel()

                # Update the sparse matrix with the scaled 'Result' values
                encoded_features[mask, i] = code_results

            # Convert the lil_matrix to a csr_matrix for efficient row slicing
            encoded_features = encoded_features.tocsr()

            # Concatenate the original chunk with the encoded features
            encoded_chunk = pd.concat([chunk[['EMPI', 'Code', 'Result']], pd.DataFrame(encoded_features.toarray())], axis=1)

            # Append the encoded chunk to the list
            encoded_chunks.append(encoded_chunk)

            # Update the feature names
            feature_names.extend([f"Code_{code}" for code in unique_codes_chunk])

        # Concatenate all the encoded chunks into a single DataFrame
        encoded_df = pd.concat(encoded_chunks, ignore_index=True)

        # Return the encoded DataFrame, the fitted scaler if fit=True, and the feature names
        if fit:
            return encoded_df, fitted_scaler, feature_names
        else:
            return encoded_df, feature_names

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