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
import math
from math import ceil
import json
import re
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_sparse
from joblib import Parallel, delayed
import argparse
import dask
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
from typing import Dict, Set, List, Any
import dill
import logging
import ast
import psutil

def inspect_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {file_path}")
        print(f"Type of loaded object: {type(data)}")
        if isinstance(data, dict):
            print("Keys in the loaded dictionary:")
            for key in data.keys():
                print(f"- {key}: {type(data[key])}")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

def load_with_dill(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        print(f"Successfully loaded {file_path} with dill")
        return data
    except Exception as e:
        print(f"Error loading {file_path} with dill: {str(e)}")
        return None

def map_codes_to_indices(df: pl.DataFrame, code_to_index: Dict[str, int]) -> pl.DataFrame:
    """
    Map the CodeWithType column to indices based on the provided mapping.
    """
    return df.with_columns([
        pl.col('CodeWithType').map_dict(code_to_index).alias('dynamic_indices')
    ])

def create_inverse_mapping(code_mapping: Dict[str, int]) -> Dict[int, str]:
    """
    Create an inverse mapping from indices to codes.
    """
    return {idx: code for code, idx in code_mapping.items()}

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

import json
import logging
from pathlib import Path
import torch
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CustomPytorchDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, dl_reps_dir, subjects_df, df_dia, df_prc, task_df=None, device=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initializing CustomPytorchDataset for split: {split}")

        self.split = split
        self.dl_reps_dir = Path(dl_reps_dir)
        self.subjects_df = subjects_df
        self.task_df = task_df
        self.df_dia = df_dia
        self.df_prc = df_prc
        self.device = device

        self.has_task = task_df is not None
        if self.has_task:
            task_columns = [col for col in task_df.columns if col != 'subject_id']
            if len(task_columns) == 1:
                self.tasks = ['label']
                self.task_types = {'label': 'binary_classification'}
            else:
                self.tasks = task_columns
                self.task_types = {task: 'binary_classification' for task in self.tasks}
        else:
            self.tasks = []
            self.task_types = {}

        self.create_code_mapping()  # Create code mapping before loading cached data
        self.load_cached_data()

        self.logger.info(f"CustomPytorchDataset initialized for split: {split}")
        self.logger.info(f"Dataset length: {len(self.cached_data)}")
        self.logger.info(f"Has task: {self.has_task}")

    def create_code_mapping(self):
        all_codes = set(self.df_dia['CodeWithType'].unique()) | set(self.df_prc['CodeWithType'].unique())
        
        sorted_codes = sorted(all_codes)
        self.code_to_index = {code: idx for idx, code in enumerate(sorted_codes, start=1)}
        
        self.logger.info(f"Created code mapping with {len(self.code_to_index)} unique codes")
        sample_codes = list(self.code_to_index.items())[:10]
        self.logger.debug(f"Code mapping sample: {dict(sample_codes)}")
        
        self.logger.debug(f"Full code mapping: {self.code_to_index}")

        # Save the code mapping to a file
        with open("data/code_mapping.json", 'w') as f:
            json.dump(self.code_to_index, f)

    def load_cached_data(self):
        self.logger.info(f"Loading cached data for split: {self.split}")
        
        if not self.dl_reps_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.dl_reps_dir}")

        parquet_files = list(self.dl_reps_dir.glob(f"{self.split}*.parquet"))
        self.logger.debug(f"Found {len(parquet_files)} Parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found for split '{self.split}' in directory '{self.dl_reps_dir}'")
        
        dfs = []
        total_rows = 0
        for parquet_file in parquet_files:
            self.logger.debug(f"Reading file: {parquet_file}")
            try:
                df = pl.read_parquet(parquet_file)
                self.logger.debug(f"File {parquet_file} shape: {df.shape}")
                if self.task_df is not None:
                    df = df.join(self.task_df, on='subject_id', how='inner')
                dfs.append(df)
                total_rows += df.shape[0]
            except Exception as e:
                self.logger.error(f"Error reading Parquet file: {parquet_file}")
                self.logger.error(f"Error message: {str(e)}")
                continue

        if not dfs:
            self.logger.error(f"No data loaded for split: {self.split}")
            raise ValueError(f"No data loaded for split: {self.split}")

        self.logger.info(f"Loaded {total_rows} rows of cached data")

        if total_rows == 0:
            raise ValueError(f"No matching data found for split: {self.split}")

        self.cached_data = pl.concat(dfs)

        # Print some information about the loaded data
        self.logger.info(f"Loaded {self.cached_data.shape[0]} rows of cached data")
        self.logger.info(f"Cached data columns: {self.cached_data.columns}")
        self.logger.info(f"Sample of dynamic_indices: {self.cached_data['dynamic_indices'].head(5)}")

        # Filter out rows with empty dynamic_indices
        self.cached_data = self.cached_data.filter(pl.col('dynamic_indices').is_not_null() & (pl.col('dynamic_indices') != '[]') & (pl.col('dynamic_indices') != ''))
        
        self.logger.info(f"After filtering, {self.cached_data.shape[0]} rows remain")
        self.logger.info(f"Sample of dynamic_indices after filtering: {self.cached_data['dynamic_indices'].head(5)}")

    def __len__(self):
        return self.cached_data.shape[0]

    def __getitem__(self, idx):
        try:
            row = self.cached_data.row(idx)
            subject_id = row[self.cached_data.columns.index('subject_id')]

            raw_dynamic_indices = row[self.cached_data.columns.index('dynamic_indices')]
            # self.logger.debug(f"Raw dynamic_indices for subject {subject_id}: {raw_dynamic_indices}")

            dynamic_indices = self.process_dynamic_indices(raw_dynamic_indices)
            # self.logger.debug(f"Processed dynamic_indices for subject {subject_id}: {dynamic_indices}")

            dynamic_counts = self.process_dynamic_counts(row[self.cached_data.columns.index('dynamic_counts')])

            # Ensure dynamic_indices is not empty
            if dynamic_indices.numel() == 0:
                dynamic_indices = torch.tensor([0], dtype=torch.long)  # Use 0 for padding
                dynamic_counts = torch.tensor([1.0], dtype=torch.float32)

            item = {
                'subject_id': subject_id,
                'dynamic_indices': dynamic_indices,
                'dynamic_counts': dynamic_counts,
                'labels': torch.tensor(row[self.cached_data.columns.index('label')], dtype=torch.float32) if self.has_task else None,
            }

            # Add static features
            static_features = ['InitialA1c', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']
            for feature in static_features:
                item[feature] = self.process_static_feature(row, feature)

            if 'A1cGreaterThan7' in self.cached_data.columns:
                item['A1cGreaterThan7'] = self.process_static_feature(row, 'A1cGreaterThan7', dtype=torch.float32)
            else:
                item['A1cGreaterThan7'] = torch.tensor(0.0, dtype=torch.float32)

            return self.handle_nan_values(item)

        except Exception as e:
            self.logger.error(f"Error getting item at index {idx}: {str(e)}")
            return self.get_default_item()

    def process_dynamic_indices(self, indices):
        if indices is None or (isinstance(indices, (list, str)) and len(indices) == 0):
            self.logger.warning(f"Received empty or None for dynamic_indices, using default value")
            return torch.tensor([0], dtype=torch.long)
        
        if isinstance(indices, str):
            codes = [indices]  # Single code as a string
        elif isinstance(indices, list):
            codes = indices
        else:
            self.logger.error(f"Unexpected type for dynamic_indices: {type(indices)}")
            return torch.tensor([0], dtype=torch.long)
        
        processed_indices = []
        for code in codes:
            code_str = str(code).strip()
            index = self.code_to_index.get(code_str, 0)
            # self.logger.debug(f"Code: {code_str}, Index: {index}")
            processed_indices.append(index)
        
        # self.logger.debug(f"Raw codes: {codes}")
        # self.logger.debug(f"Processed indices: {processed_indices}")
        return torch.tensor(processed_indices, dtype=torch.long)

    def process_dynamic_counts(self, counts):
        if counts is None:
            return torch.tensor([0.0], dtype=torch.float32)
        if isinstance(counts, str):
            try:
                counts = json.loads(counts)
            except json.JSONDecodeError:
                counts = [float(x) for x in counts.split(',')]
        if isinstance(counts, (int, float)):
            counts = [float(counts)]
        return torch.tensor(counts, dtype=torch.float32)

    def process_static_feature(self, row, feature, dtype=None):
        if dtype is None:
            dtype = torch.float32 if feature in ['InitialA1c', 'AgeYears', 'SDI_score'] else torch.long
        value = row[self.cached_data.columns.index(feature)] if feature in self.cached_data.columns else None
        if value is None:
            value = 0.0 if dtype == torch.float32 else 0
        return torch.tensor(value, dtype=dtype)

    def handle_nan_values(self, item):
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                self.logger.warning(f"NaN values found in {key} for subject_id {item['subject_id']}")
                item[key] = torch.nan_to_num(value, nan=0.0)
        return item

    def get_default_item(self):
        return {
            'subject_id': -1,
            'dynamic_indices': torch.tensor([1], dtype=torch.long),
            'dynamic_counts': torch.tensor([1.0], dtype=torch.float32),
            'labels': torch.tensor(0.0, dtype=torch.float32) if self.has_task else None,
            'InitialA1c': torch.tensor(0.0, dtype=torch.float32),
            'Female': torch.tensor(0, dtype=torch.long),
            'Married': torch.tensor(0, dtype=torch.long),
            'GovIns': torch.tensor(0, dtype=torch.long),
            'English': torch.tensor(0, dtype=torch.long),
            'AgeYears': torch.tensor(0.0, dtype=torch.float32),
            'SDI_score': torch.tensor(0.0, dtype=torch.float32),
            'Veteran': torch.tensor(0, dtype=torch.long),
            'A1cGreaterThan7': torch.tensor(0.0, dtype=torch.float32)
        }

    def get_max_index(self):
        return max(self.code_to_index.values())

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

def plot_avg_measurements_per_patient_per_month(dynamic_measurements_df):
    # Convert timestamp to month
    dynamic_measurements_df = dynamic_measurements_df.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month")
    )

    # Calculate the number of measurements per patient per month
    measurements_per_patient_per_month = (
        dynamic_measurements_df.groupby(["subject_id", "month"])
        .agg(pl.count("measurement_id").alias("num_measurements"))
        .groupby("month")
        .agg(pl.mean("num_measurements").alias("avg_measurements_per_patient"))
    )

    # Convert to pandas DataFrame and sort by month
    measurements_per_patient_per_month_pd = measurements_per_patient_per_month.sort("month").to_pandas()
    measurements_per_patient_per_month_pd['month'] = pd.to_datetime(measurements_per_patient_per_month_pd['month'])

    # Create and save the plot
    save_plot_line(
        measurements_per_patient_per_month_pd, 
        'month', 
        'avg_measurements_per_patient', 
        'avg_measurements_per_patient_per_month.png', 
        'Month', 
        'Average Measurements per Patient', 
        'Average Measurements per Patient per Month',
        x_range=(mdates.date2num(datetime(1990, 1, 1)), mdates.date2num(datetime(2022, 12, 31)))
    )

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

def temporal_dist_pd_monthly(df, measure):
    df = df.sort('timestamp')
    df = df.with_columns(pl.col('timestamp').dt.strftime('%Y-%m').alias('month'))
    monthly_counts = df.groupby('month').agg(pl.count(measure).alias('count'))
    return monthly_counts.sort('month')

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

def preprocess_dataframe(df_name, file_path, columns, selected_columns, chunk_size=10000, min_frequency=None):
    chunks = []
    for chunk in pd.read_csv(file_path, sep='|', chunksize=chunk_size):
        chunk = read_file(file_path, columns, selected_columns, chunk_size=chunk_size, parse_dates=None)
        chunk = process_chunk(df_name, chunk, columns, selected_columns, min_frequency)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return pl.from_pandas(df)

def process_chunk(df_name, chunk, columns, selected_columns, min_frequency):
    chunk = chunk[selected_columns]

    if df_name == 'Outcomes':
        chunk['A1cGreaterThan7'] = chunk['A1cGreaterThan7'].astype(float)

    if df_name in ['Diagnoses', 'Procedures', 'Labs']:
        chunk['Date'] = pd.to_datetime(chunk['Date'], format="%Y-%m-%d %H:%M:%S")

    if df_name in ['Diagnoses', 'Procedures']:
        chunk = chunk.dropna(subset=['Date', 'CodeWithType'])
        if min_frequency is not None:
            code_counts = chunk['CodeWithType'].value_counts()
            if isinstance(min_frequency, int):
                infrequent_categories = code_counts[code_counts < min_frequency].index
            else:
                min_count_threshold = min_frequency * len(chunk)
                infrequent_categories = code_counts[code_counts < min_count_threshold].index
            chunk = chunk[~chunk['CodeWithType'].isin(infrequent_categories)]

    elif df_name == 'Labs':
        chunk = chunk.dropna(subset=['Date', 'Code', 'Result'])
        if min_frequency is not None:
            code_counts = chunk['Code'].value_counts()
            if isinstance(min_frequency, int):
                infrequent_categories = code_counts[code_counts < min_frequency].index
            else:
                min_count_threshold = min_frequency * len(chunk)
                infrequent_categories = code_counts[code_counts < min_count_threshold].index
            chunk = chunk[~chunk['Code'].isin(infrequent_categories)]

    return chunk

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