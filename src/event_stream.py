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
from typing import Any
import polars as pl
from polars.datatypes import DataType
import pickle
from pathlib import Path
import polars.expr as expr
import pyarrow.parquet as pq

from EventStream.data.config import (
    InputDFSchema,
    MeasurementConfig,
)
from EventStream.data.dataset_config import DatasetConfig
from EventStream.data.dataset_schema import DatasetSchema
from EventStream.data.dataset_polars import Dataset
from EventStream.data.types import (
    DataModality,
    InputDataType,
    InputDFType,
    TemporalityType,
)
from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals, create_code_mapping, map_codes_to_indices, create_inverse_mapping, optimize_labs_data
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select
from collections import defaultdict
from EventStream.data.preprocessing.standard_scaler import StandardScaler

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import tempfile
import shutil
import pyarrow.parquet as pq

import psutil

def try_convert_to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None
        
def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def main(use_labs=False, save_schema=False, debug=False):
    print("Building dataset config...")

    if use_labs:
        if not os.path.exists("data/labs"):
            os.makedirs("data/labs")
        data_dir = Path("data/labs")
    else:
        data_dir = Path("data")

    # Build Config
    split = (0.7, 0.2, 0.1)
    seed = 42
    do_overwrite = True
    DL_chunk_size = 20000

    config = DatasetConfig(
        measurement_configs={
            'dynamic_indices': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
        },
        normalizer_config={'cls': 'standard_scaler'},
        save_dir=data_dir
    )

    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'
    if use_labs:
        labs_file_path = 'data/Labs.txt'

    subjects_df = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select)

    # Aggregate subjects_df to have one row per StudyID
    subjects_df = subjects_df.group_by('StudyID').agg([
        pl.col('InitialA1c').first(),
        pl.col('A1cGreaterThan7').first(),
        pl.col('Female').first(),
        pl.col('Married').first(),
        pl.col('GovIns').first(),
        pl.col('English').first(),
        pl.col('AgeYears').first(),
        pl.col('SDI_score').first(),
        pl.col('Veteran').first()
    ])

    # Create a mapping from StudyID to integer subject_id
    subject_id_mapping = {study_id: idx for idx, study_id in enumerate(subjects_df['StudyID'].unique(), start=1)}

    # Add the subject_id column
    subjects_df = subjects_df.with_columns([
        pl.col('StudyID').replace(subject_id_mapping).alias('subject_id')
    ])

    subjects_df = subjects_df.with_columns([pl.col('subject_id').cast(pl.UInt32)])

    print("Shape of subjects_df after aggregation:", subjects_df.shape)
    print("Number of unique EMPIs:", subjects_df['StudyID'].n_unique())

    print("Processing diagnosis and procedure data...")
    df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select, min_frequency=ceil(subjects_df.height*0.01))
    df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select, min_frequency=ceil(subjects_df.height*0.01))

    # Add subject_id to df_dia and df_prc
    df_dia = df_dia.join(subjects_df.select(['StudyID', 'subject_id']), on='StudyID', how='inner')
    df_prc = df_prc.join(subjects_df.select(['StudyID', 'subject_id']), on='StudyID', how='inner')

    if use_labs:
        print("Processing labs data...")
        df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, min_frequency=ceil(subjects_df.height*0.01), debug=debug)
        df_labs = optimize_labs_data(df_labs)
        df_labs = df_labs.join(subjects_df.select(['StudyID', 'subject_id']), on='StudyID', how='inner')

    # Add print statements for the number of unique mapped codes
    print(f"Number of unique mapped diagnosis codes: {df_dia['CodeWithType'].n_unique()}")
    print(f"Number of unique mapped procedures codes: {df_prc['CodeWithType'].n_unique()}")
    if use_labs:
        print(f"Number of unique mapped lab codes: {df_labs['Code'].n_unique()}")

    print("Creating code mapping...")
    code_to_index = create_code_mapping(df_dia, df_prc, df_labs if use_labs else None)

    # Create the inverse mapping
    index_to_code = {idx: code for code, idx in code_to_index.items()}

    print(f"Sample of index_to_code: {dict(list(index_to_code.items())[:5])}")

    print("Creating events dataframe...")
    # Create an instance of the Dataset class
    dataset = Dataset(config=config)

    # Process each dataframe separately
    df_dia_events, df_dia_dynamic = dataset._process_events_and_measurements_df(df_dia, "DIAGNOSIS", dia_columns_select, code_to_index, subject_id_mapping)
    df_prc_events, df_prc_dynamic = dataset._process_events_and_measurements_df(df_prc, "PROCEDURE", prc_columns_select, code_to_index, subject_id_mapping)
    if use_labs:
        df_labs_events, df_labs_dynamic = dataset._process_events_and_measurements_df(df_labs, "LAB", labs_columns_select, code_to_index, subject_id_mapping)

    # Concatenate events dataframes
    if use_labs:
        events_df = pl.concat([df_dia_events, df_prc_events, df_labs_events], how='vertical')
    else:
        events_df = pl.concat([df_dia_events, df_prc_events], how='vertical')

    # Sort events by subject_id and timestamp
    events_df = events_df.sort(['subject_id', 'timestamp'])

    # First, let's check the content of the event_type column
    print(events_df.select(pl.col("event_type")).head())

    # Now, let's modify our aggregation to handle potential null values and list structure
    events_df = events_df.group_by(['subject_id', 'timestamp', 'event_id']).agg([
        pl.col('event_type').filter(pl.col('event_type').is_not_null()).alias('event_types')
    ])

    # Join event types with '&' and keep only unique combinations
    events_df = events_df.with_columns([
        pl.when(pl.col('event_types').list.len() > 0)
        .then(
            pl.col('event_types')
            .cast(pl.List(pl.Utf8))  # Convert Categorical to String within the list
            .list.unique()
            .list.join('&')
        )
        .otherwise(None)
        .alias('event_type')
    ]).drop('event_types')

    # Check the result
    print(events_df.head())

    print("Event types after combining:")
    print(events_df['event_type'].value_counts())

    print("Columns in df_dia_dynamic:", df_dia_dynamic.columns)
    print("Columns in df_prc_dynamic:", df_prc_dynamic.columns)
    if use_labs:
        print("Columns in df_labs_dynamic:", df_labs_dynamic.columns)

    def align_columns(df, subject_id_mapping):
        required_columns = ['subject_id', 'timestamp', 'dynamic_indices', 'dynamic_counts']
        if use_labs:
            required_columns.append('dynamic_values')
        
        # If 'subject_id' is not in the dataframe, create it from 'StudyID'
        if 'subject_id' not in df.columns:
            if 'StudyID' in df.columns:
                df = df.with_columns([
                    pl.col('StudyID').cast(pl.Utf8).replace(subject_id_mapping).alias('subject_id').cast(pl.UInt32)
                ])
            else:
                raise ValueError("Neither 'subject_id' nor 'StudyID' column found in the dataframe")
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'dynamic_values' and use_labs:
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
                elif col == 'dynamic_counts':
                    df = df.with_columns(pl.lit(1).cast(pl.UInt32).alias(col))
        
        return df.select(required_columns)

    df_dia_dynamic = align_columns(df_dia_dynamic, subject_id_mapping)
    df_prc_dynamic = align_columns(df_prc_dynamic, subject_id_mapping)
    if use_labs:
        df_labs_dynamic = align_columns(df_labs_dynamic, subject_id_mapping)

    print("df_dia_dynamic:")
    print(f"Shape: {df_dia_dynamic.shape}")
    print(f"Columns: {df_dia_dynamic.columns}")
    print(df_dia_dynamic.head())

    print("\ndf_prc_dynamic:")
    print(f"Shape: {df_prc_dynamic.shape}")
    print(f"Columns: {df_prc_dynamic.columns}")
    print(df_prc_dynamic.head())

    if use_labs:
        print("\ndf_labs_dynamic:")
        print(f"Shape: {df_labs_dynamic.shape}")
        print(f"Columns: {df_labs_dynamic.columns}")
        print(df_labs_dynamic.head())

    # Concatenate dynamic measurements dataframes
    if use_labs:
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic, df_labs_dynamic], how='vertical')
    else:
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic], how='vertical')

    print("After concatenation:")
    print(f"Shape: {dynamic_measurements_df.shape}")
    print(f"Columns: {dynamic_measurements_df.columns}")
    print(dynamic_measurements_df.head())

    print("Columns in concatenated dynamic_measurements_df:", dynamic_measurements_df.columns)

    print("Sample of dynamic_indices before mapping:")
    print(dynamic_measurements_df.select('dynamic_indices').head())
    print("Sample of code_to_index:")
    print(dict(list(code_to_index.items())[:10]))

    # Sort and add event_id
    dynamic_measurements_df = (
        dynamic_measurements_df
        .sort(['subject_id', 'timestamp'])
        .with_row_count("event_id")
        .with_columns(pl.col('event_id').cast(pl.Int64))
    )

    print("Final columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Ensure event_id is unique across the entire events_df
    if 'event_id' not in events_df.columns:
        events_df = events_df.with_row_count("event_id")
    events_df = events_df.with_columns(pl.col('event_id').cast(pl.Int64))

    if use_labs:
        CHUNK_SIZE = 1_000_000  # Adjust this based on your memory constraints

        def process_in_chunks(df, chunk_size, process_func, *args):
            processed_chunks = []
            for chunk in df.iter_slices(chunk_size):
                processed_chunk = process_func(chunk, *args)
                processed_chunks.append(processed_chunk)
            return pl.concat(processed_chunks)
                   
        def join_with_events(chunk_df, events_df):
            return chunk_df.join(
                events_df.select('subject_id', 'timestamp', 'event_id'),
                on=['subject_id', 'timestamp'],
                how='left'
            )

        print("Processing dynamic_measurements_df in chunks...")
        dynamic_measurements_df = process_in_chunks(
            dynamic_measurements_df, 
            CHUNK_SIZE, 
            join_with_events, 
            events_df
        )

        if 'event_id' in dynamic_measurements_df.columns:
            dynamic_measurements_df = dynamic_measurements_df.drop('event_id')

        # Sort and add new event_id
        print("Sorting and adding new event_id...")
        dynamic_measurements_df = (
            dynamic_measurements_df
            .sort(['subject_id', 'timestamp'])
            .with_row_count("event_id")
            .with_columns(pl.col('event_id').cast(pl.Int64))
        )
    else:
        # Update dynamic_measurements_df with the new event_ids
        dynamic_measurements_df = dynamic_measurements_df.join(
            events_df.select('subject_id', 'timestamp', 'event_id'),
            on=['subject_id', 'timestamp'],
            how='left'
        )

    print("Creating event types index map...")
    event_types = events_df['event_type'].unique().to_list()
    event_types_idxmap = {event_type: idx for idx, event_type in enumerate(event_types, start=1)}

    print("Updating config with event types index map...")
    config.event_types_idxmap = event_types_idxmap

    print("Event types index map:")
    print(event_types_idxmap)

    print("Events dataframe created.")
    print(f"Shape of events_df: {events_df.shape}")
    print(f"Columns of events_df: {events_df.columns}")

    print("Dynamic measurements dataframe created.")
    print(f"Shape of dynamic_measurements_df: {dynamic_measurements_df.shape}")
    print(f"Columns of dynamic_measurements_df: {dynamic_measurements_df.columns}")

    if use_labs:
        # Ensure dynamic_values column exists and is of the correct type
        if 'dynamic_values' not in dynamic_measurements_df.columns:
            dynamic_measurements_df = dynamic_measurements_df.with_columns([
                pl.lit(None).cast(pl.Utf8).alias('dynamic_values')
            ])
        else:
            dynamic_measurements_df = dynamic_measurements_df.with_columns([
                pl.col('dynamic_values').cast(pl.Utf8)
            ])

    # Ensure dynamic_counts column exists and is of the correct type
    if 'dynamic_counts' not in dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.lit(1).cast(pl.UInt32).alias('dynamic_counts')
        ])
    else:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('dynamic_counts').fill_null(1).cast(pl.UInt32)
        ])

    print("Final sample of dynamic_measurements_df:")
    print(dynamic_measurements_df.head())
    print("\nFinal columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    def process_df(df, code_col, value_col=None, event_type=None):
        base_cols = [
            'subject_id',
            pl.col('Date').alias('timestamp'),
            pl.col(code_col).alias('code'),  # Keep original code
            pl.lit(1).cast(pl.UInt32).alias('dynamic_counts'),
            pl.col(value_col).cast(pl.Utf8).alias('dynamic_values') if value_col else pl.lit(None).cast(pl.Utf8).alias('dynamic_values'),
            pl.lit(event_type).alias('event_type')
        ]
        return df.select(base_cols)

    print("Creating dynamic measurements dataframe...")
    df_dia_dynamic = process_df(df_dia, 'CodeWithType', event_type='DIAGNOSIS')
    df_prc_dynamic = process_df(df_prc, 'CodeWithType', event_type='PROCEDURE')
    if use_labs:
        df_labs_dynamic = process_df(df_labs, 'Code', 'Result', event_type='LAB')

    # Concatenate the dataframes
    if use_labs:
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic, df_labs_dynamic], how='vertical')
    else:
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic], how='vertical')

    # Create a unified code_to_index mapping
    all_codes = set(dynamic_measurements_df['code'].unique())
    code_to_index = {code: idx for idx, code in enumerate(sorted(all_codes), start=1)}

    # Ensure UNKNOWN is in the mapping
    if 'UNKNOWN' not in code_to_index:
        code_to_index['UNKNOWN'] = len(code_to_index) + 1

    # Map the codes to indices
    unknown_index = code_to_index['UNKNOWN']
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('code').replace(code_to_index, default=unknown_index).alias('dynamic_indices')
    ])

    # Sort and add event_id
    dynamic_measurements_df = (
        dynamic_measurements_df
        .sort(['subject_id', 'timestamp'])
        .with_row_count("event_id")
        .with_columns(pl.col('event_id').cast(pl.Int64))
    )

    # Ensure correct data types
    dynamic_measurements_columns = [
        pl.col('event_id').cast(pl.Int64),
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('dynamic_indices').cast(pl.UInt32),
        pl.col('dynamic_counts').fill_null(1).cast(pl.UInt32),
        pl.col('timestamp').cast(pl.Datetime),
    ]
    if use_labs:
        dynamic_measurements_columns.append(pl.col('dynamic_values').cast(pl.Utf8))

    dynamic_measurements_df = dynamic_measurements_df.with_columns(dynamic_measurements_columns)

    # Drop the original 'code' column
    dynamic_measurements_df = dynamic_measurements_df.drop('code')

    print("Final sample of dynamic_measurements_df:")
    print(dynamic_measurements_df.head())
    print("\nFinal columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    print("Add epsilon to concurrent timestamps")
    eps = np.finfo(float).eps
    default_timestamp = datetime(2000, 1, 1)

    # Apply timestamp adjustment to events_df
    events_df = events_df.with_columns([
        pl.col('timestamp').fill_null(default_timestamp).alias('timestamp'),
        pl.col('event_id').cast(pl.Int64),
        pl.col('event_type').cast(pl.Utf8)  # Convert Categorical to String
    ]).with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp')
    ])

    # Apply timestamp adjustment to dynamic_measurements_df
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('timestamp').fill_null(default_timestamp).alias('timestamp'),
        pl.col('event_id').cast(pl.Int64),
        pl.col('event_type').cast(pl.Utf8)  # Convert Categorical to String
    ]).with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp')
    ])

    # Ensure event_type is not a combination
    events_df = events_df.with_columns([
        pl.col('event_type').str.split('&').list.first().alias('event_type')
    ])

    # Count of events by type
    event_type_counts = dynamic_measurements_df.group_by('event_type').count()
    print("\nCount of events by type:")
    print(event_type_counts)

    print("Sample of dynamic_measurements_df after processing:")
    print(dynamic_measurements_df.head())
    print("\nColumns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Update vocabulary configuration
    vocab_sizes_by_measurement = {
        'event_type': len(set(dynamic_measurements_df['event_type'].to_list())),
        'dynamic_indices': len(code_to_index),
    }

    vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': vocab_sizes_by_measurement['event_type'] + 1,
    }

    config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

    print("Updated Vocabulary sizes by measurement:")
    print(vocab_sizes_by_measurement)
    print("Updated Vocabulary offsets by measurement:")
    print(vocab_offsets_by_measurement)

    # Save mappings to disk as JSON files
    print("Saving mappings to disk...")
    with open(data_dir / 'code_to_index.json', 'w') as f:
        json.dump(code_to_index, f)

    with open(data_dir / 'index_to_code.json', 'w') as f:
        json.dump(index_to_code, f)

    print("Mappings saved successfully.")

    print("Saving dataframes as Parquet files")
    dynamic_measurements_df.write_parquet(data_dir / "dynamic_measurements_df.parquet")
    events_df.write_parquet(data_dir / "events_df.parquet")
    subjects_df.write_parquet(data_dir / "subjects_df.parquet")

    print("Processing events and measurements data...")
    temp_dataset = Dataset(config, subjects_df=subjects_df, events_df=events_df, dynamic_measurements_df=dynamic_measurements_df)

    print("Columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)

    print("Unique values in dynamic_indices:")
    print(dynamic_measurements_df['dynamic_indices'].unique().sort().head(20))

    # Ensure event_id is Int64 in both DataFrames
    dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('event_id').cast(pl.Int64))
    events_df = events_df.with_columns(pl.col('event_id').cast(pl.Int64))

    print("Final sample of dynamic_measurements_df:")
    print(dynamic_measurements_df.head())
    print("\nFinal columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Ensure 'subject_id' is present before further processing
    if 'subject_id' not in dynamic_measurements_df.columns:
        raise ValueError("'subject_id' column is missing from dynamic_measurements_df")

    print("Sorting dynamic_measurements_df...")
    dynamic_measurements_df = dynamic_measurements_df.sort(['subject_id', 'timestamp', 'event_id'])

    print("Sample of sorted dynamic_measurements_df:")
    print(dynamic_measurements_df.head())

    temp_dataset.dynamic_measurements_df = dynamic_measurements_df

    print("Before _convert_dynamic_indices_to_indices")
    print("Data types in dynamic_measurements_df:")
    for col in temp_dataset.dynamic_measurements_df.columns:
        print(f"{col}: {temp_dataset.dynamic_measurements_df[col].dtype}")

    if not hasattr(temp_dataset, 'code_mapping') or temp_dataset.code_mapping is None:
        temp_dataset._create_code_mapping()

    print("Code mapping sample:")
    print(dict(list(temp_dataset.code_mapping.items())[:5]))  # Print first 5 items of code_mapping

    print("Sample of dynamic_indices before conversion:")
    print(temp_dataset.dynamic_measurements_df.select('dynamic_indices').head())

    temp_dataset._convert_dynamic_indices_to_indices()

    print("After _convert_dynamic_indices_to_indices")
    print("Data types in dynamic_measurements_df:")
    for col in temp_dataset.dynamic_measurements_df.columns:
        print(f"{col}: {temp_dataset.dynamic_measurements_df[col].dtype}")

    print("Sample of dynamic_measurements_df:")
    print(temp_dataset.dynamic_measurements_df.head())

    # Create a mapping from StudyID (UUID) to numeric subject_id
    unique_study_ids = subjects_df['StudyID'].unique()
    subject_id_mapping = {str(uuid): idx for idx, uuid in enumerate(unique_study_ids, start=1)}

    print("Columns in events_df:")
    print(events_df.columns)

    if 'subject_id' not in events_df.columns:
        print("Adding subject_id to events_df...")
        events_df = events_df.with_columns([
            pl.col('StudyID').cast(pl.Utf8).replace(subject_id_mapping).alias('subject_id').cast(pl.UInt32)
        ])

    print("Columns in events_df after potential addition of subject_id:")
    print(events_df.columns)

    # Make sure to drop the original 'StudyID' column from subjects_df
    subjects_df = subjects_df.drop('StudyID')

    print("Sample of updated subjects_df:")
    print(subjects_df.head())

    # Print sample of subjects_df to verify the changes
    print("Sample of updated subjects_df:")
    print(subjects_df.head())

    # Create the dynamic_measurements_df
    columns_to_select = ['event_id', 'timestamp', 'dynamic_indices']
    if 'Result' in temp_dataset.dynamic_measurements_df.columns:
        columns_to_select.append('Result')
    if 'subject_id' in temp_dataset.dynamic_measurements_df.columns:
        columns_to_select.append('subject_id')

    dynamic_measurements_df = temp_dataset.dynamic_measurements_df.select(columns_to_select)

    print("Adding dynamic_counts column...")
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.lit(1).cast(pl.UInt32).alias('dynamic_counts')
    ])

    if use_labs:
        if 'Result' in dynamic_measurements_df.columns:
            dynamic_measurements_df = dynamic_measurements_df.with_columns([
                pl.col('Result').alias('dynamic_values'),
                pl.col("Result").apply(try_convert_to_float, return_dtype=pl.Float32).alias("dynamic_values_numeric")
            ])
        else:
            dynamic_measurements_df = dynamic_measurements_df.with_columns([
                pl.lit(None).cast(pl.Utf8).alias('dynamic_values'),
                pl.lit(None).cast(pl.Float32).alias('dynamic_values_numeric')
            ])

        # Ensure correct data types
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('dynamic_values').cast(pl.Utf8),
            pl.col('dynamic_values_numeric').cast(pl.Float32)
        ])

    print("Sample of dynamic_measurements_df after adding dynamic_values:")
    print(dynamic_measurements_df.head())
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Convert StudyID to numeric subject_id and cast columns
    if 'subject_id' not in dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('event_id').replace(events_df.select('event_id', 'subject_id').collect().to_dict(as_series=False)).alias('subject_id').cast(pl.UInt32)
        ])

    columns_to_cast = [
        pl.col('event_id').cast(pl.UInt32),
        pl.col('dynamic_indices').cast(pl.UInt32),
        pl.col('dynamic_counts').cast(pl.UInt32),
        pl.col('timestamp').cast(pl.Datetime)
    ]

    if 'dynamic_values' in dynamic_measurements_df.columns:
        columns_to_cast.append(pl.col('dynamic_values').cast(pl.Float32))

    dynamic_measurements_df = dynamic_measurements_df.with_columns(columns_to_cast)

    # Ensure 'subject_id' is present
    if 'subject_id' not in dynamic_measurements_df.columns:
        raise ValueError("'subject_id' column is missing from dynamic_measurements_df")

    print("Sample of final dynamic_measurements_df:")
    print(dynamic_measurements_df.head())
    print("Shape of subjects_df:", subjects_df.shape)
    print("Columns of subjects_df:", subjects_df.columns)
    print("Data types in subjects_df:")
    for col in subjects_df.columns:
        print(f"{col}: {subjects_df[col].dtype}")

    if use_labs:
        # Prepare labs data
        labs_df = df_labs.select('StudyID', 'Code', 'Result', 'Date').rename({
            'StudyID': 'subject_id', 
            'Date': 'timestamp',
            'Code': 'dynamic_indices',
            'Result': 'dynamic_values'
        })

        # Map the 'Code' to indices using the code_to_index mapping
        labs_df = labs_df.with_columns([
            pl.col('dynamic_indices').replace(code_to_index).cast(pl.UInt32)
        ])

        # Add dynamic_counts and event_type
        labs_df = labs_df.with_columns([
            pl.lit(1).cast(pl.UInt32).alias('dynamic_counts'),
            pl.lit('LAB').alias('event_type')
        ])

        # Convert subject_id from UUID to numeric ID using subject_id_mapping
        labs_df = labs_df.with_columns([
            pl.col('subject_id').replace(subject_id_mapping).cast(pl.UInt32)
        ])

        # Ensure all columns have the correct data types
        labs_df = labs_df.with_columns([
            pl.col('timestamp').cast(pl.Datetime),
            pl.col('dynamic_values').cast(pl.Utf8),
            pl.col('dynamic_values').map_elements(try_convert_to_float).alias('dynamic_values_numeric')
        ])

        # Add event_id column and ensure it's Int64
        labs_df = labs_df.with_row_count("event_id").with_columns([
            pl.col('event_id').cast(pl.Int64)
        ])

        # Get the exact column structure from dynamic_measurements_df
        dynamic_columns = dynamic_measurements_df.columns

        # Ensure labs_df has all the columns from dynamic_measurements_df
        for col in dynamic_columns:
            if col not in labs_df.columns:
                labs_df = labs_df.with_columns(pl.lit(None).alias(col))

        # Select only the columns that are in dynamic_measurements_df
        labs_df = labs_df.select(dynamic_columns)

        print("Columns in labs_df after alignment:", labs_df.columns)

        # Ensure all columns have the correct data types in both DataFrames
        columns_to_cast = {
            'event_id': pl.Int64,
            'subject_id': pl.UInt32,
            'dynamic_indices': pl.UInt32,
            'dynamic_counts': pl.UInt32,
            'timestamp': pl.Datetime,
        }
        if use_labs:
            columns_to_cast.update({
                'dynamic_values': pl.Utf8,
                'dynamic_values_numeric': pl.Float32
            })

        for col, dtype in columns_to_cast.items():
            if col in dynamic_measurements_df.columns:
                dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col(col).cast(dtype))
            if use_labs and col in labs_df.columns:
                labs_df = labs_df.with_columns(pl.col(col).cast(dtype))

        print("Columns in dynamic_measurements_df:", dynamic_measurements_df.columns)
        print("Columns in labs_df:", labs_df.columns)

        # Combine existing dynamic_measurements_df with labs_df
        dynamic_measurements_df = pl.concat([dynamic_measurements_df, labs_df])

        if 'event_id' in dynamic_measurements_df.columns:
            dynamic_measurements_df = dynamic_measurements_df.rename({'event_id': 'old_event_id'})

        # Sort and add new event_id
        print("Sorting and adding new event_id...")
        dynamic_measurements_df = (
            dynamic_measurements_df
            .sort(['subject_id', 'timestamp'])
            .with_row_count("event_id")
            .with_columns(pl.col('event_id').cast(pl.Int64))
        )

    # Ensure all columns have the correct data types
    columns_to_cast = [
        pl.col('event_id').cast(pl.Int64),
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('dynamic_indices').cast(pl.UInt32),
        pl.col('dynamic_counts').cast(pl.UInt32),
        pl.col('timestamp').cast(pl.Datetime),
    ]

    if use_labs:
        columns_to_cast.append(pl.col('dynamic_values').cast(pl.Utf8))

    dynamic_measurements_df = dynamic_measurements_df.with_columns(columns_to_cast)

    # If using labs, ensure dynamic_values_numeric is also cast correctly
    if use_labs and 'dynamic_values_numeric' in dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('dynamic_values_numeric').cast(pl.Float32)
        ])

    print("Final columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("Data types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Update vocabulary configuration
    event_types = events_df['event_type'].unique().to_list()
    event_types_idxmap = {event_type: idx for idx, event_type in enumerate(event_types, start=1)}

    vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap),
        'dynamic_indices': len(code_to_index),
    }

    vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': len(event_types_idxmap) + 1,
    }

    config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

    print("Event types index map:")
    print(event_types_idxmap)
    print("Vocabulary sizes by measurement:")
    print(vocab_sizes_by_measurement)
    print("Vocabulary offsets by measurement:")
    print(vocab_offsets_by_measurement)

    print(f"Number of unique mapped diagnosis codes: {df_dia['CodeWithType'].n_unique()}")
    print(f"Number of unique mapped procedures codes: {df_prc['CodeWithType'].n_unique()}")

    print("Convert temporal dfs to lazy frames..")
    events_df = events_df.lazy()
    dynamic_measurements_df = dynamic_measurements_df.lazy()

    if save_schema: 
        print("Building dataset schema...")
        dataset_schema = DatasetSchema(
            static=InputDFSchema(
                input_df=subjects_df,  # Use subjects_df directly
                subject_id_col='subject_id',
                data_schema={
                    'InitialA1c': InputDataType.FLOAT,
                    'Female': InputDataType.CATEGORICAL,
                    'Married': InputDataType.CATEGORICAL,
                    'GovIns': InputDataType.CATEGORICAL,
                    'English': InputDataType.CATEGORICAL,
                    'AgeYears': InputDataType.CATEGORICAL,
                    'SDI_score': InputDataType.FLOAT,
                    'Veteran': InputDataType.CATEGORICAL
                },
                type=InputDFType.STATIC
            ),
            dynamic=dynamic_input_schemas
        )
        print("Saving input_schema to file...")
        if config.save_dir is not None:
            dataset_schema_dict = dataset_schema.to_dict()
            with open(config.save_dir / "input_schema.json", "w") as f:
                json.dump(dataset_schema_dict, f, default=json_serial)

    # Add this block before creating the Dataset object
    print("Columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)

    # Create the Dataset object with eager DataFrames
    print("Creating Dataset object...")

    print("Preparing data for Dataset creation...")
    # Ensure all DataFrames are collected (not lazy)
    subjects_df = subjects_df.collect() if isinstance(subjects_df, pl.LazyFrame) else subjects_df
    events_df = events_df.collect() if isinstance(events_df, pl.LazyFrame) else events_df
    dynamic_measurements_df = dynamic_measurements_df.collect() if isinstance(dynamic_measurements_df, pl.LazyFrame) else dynamic_measurements_df

    # Check the type of timestamp column
    if dynamic_measurements_df['timestamp'].dtype == pl.Datetime:
        print("Timestamp is already in datetime format")
    else:
        print("Converting timestamp to datetime format")
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S.%f", strict=False)
            .fill_null(pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
            .alias('timestamp')
        ])

    # Ensure correct data types
    subjects_df = subjects_df.with_columns([pl.col('subject_id').cast(pl.UInt32)])
    events_df = events_df.with_columns([
        pl.col('event_id').cast(pl.Int64),
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('timestamp').cast(pl.Datetime),
        pl.col('event_type').cast(pl.Utf8)  # Ensure event_type is String
    ])
    # Ensure all columns have the correct data types
    columns_to_cast = [
        pl.col('event_id').cast(pl.Int64),
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('dynamic_indices').cast(pl.UInt32),
        pl.col('dynamic_counts').cast(pl.UInt32),
        pl.col('timestamp').cast(pl.Datetime),
    ]

    # Only add event_type if it exists in the DataFrame
    if 'event_type' in dynamic_measurements_df.columns:
        columns_to_cast.append(pl.col('event_type').cast(pl.Utf8))

    if use_labs:
        columns_to_cast.append(pl.col('dynamic_values').cast(pl.Utf8))

    dynamic_measurements_df = dynamic_measurements_df.with_columns(columns_to_cast)

    # If using labs, ensure dynamic_values_numeric is also cast correctly
    if use_labs and 'dynamic_values_numeric' in dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('dynamic_values_numeric').cast(pl.Float32)
        ])

    print("Final data types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    print("Before creating Dataset object:")
    print(f"Shape of dynamic_measurements_df: {dynamic_measurements_df.shape}")
    print(dynamic_measurements_df.head())

    # If event_type is not in dynamic_measurements_df, we need to add it from events_df
    if 'event_type' not in dynamic_measurements_df.columns:
        print("Adding event_type to dynamic_measurements_df from events_df")
        dynamic_measurements_df = dynamic_measurements_df.join(
            events_df.select('event_id', 'event_type'),
            on='event_id',
            how='left'
        )
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('event_type').cast(pl.Utf8)
        ])

    ESD = Dataset(
        config=config,
        subjects_df=subjects_df,
        events_df=events_df,
        dynamic_measurements_df=dynamic_measurements_df,
        code_mapping=code_to_index
    )
    print("After creating Dataset object:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")
    print(ESD.dynamic_measurements_df.head())

    print("Dataset object created.")
    print_memory_usage()

    print("Splitting dataset...")
    ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
    print("Dataset split.")

    print("Preprocessing dataset...")
    print("Before preprocessing:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")

    ESD.preprocess()

    print("After preprocessing:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")
    print("Dataset preprocessed.")

    print("Caching deep learning representation...")

    print("Checking dynamic_indices in ESD.dynamic_measurements_df")
    null_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
    zero_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices') == 0).shape[0]
    unknown_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices') == ESD.code_mapping.get('UNKNOWN', len(ESD.code_mapping) + 1)).shape[0]
    print(f"Null values in dynamic_indices: {null_count}")
    print(f"Zero values in dynamic_indices: {zero_count}")
    print(f"Unknown values in dynamic_indices: {unknown_count}")

    if null_count > 0 or zero_count > 0:
        raise ValueError(f"Found {null_count} null values and {zero_count} zero values in dynamic_indices after preprocessing")
    else:
        print("No null or zero values found in dynamic_indices after preprocessing")
        print(f"Number of unknown values in dynamic_indices: {unknown_count}")

    # We don't need to check for 'UNKNOWN_CODE' as it should now have a valid index
    print(f"Sample of ESD.dynamic_measurements_df:\n{ESD.dynamic_measurements_df.head()}")

    print(f"Sample of ESD.dynamic_measurements_df:\n{ESD.dynamic_measurements_df.head()}")
    print("Before caching deep learning representation:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")

    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)

    print("After caching deep learning representation:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")
    print("Deep learning representation cached.")
    print_memory_usage()

    # Update the vocabulary configuration
    ESD.vocabulary_config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    ESD.vocabulary_config.vocab_offsets_by_measurement = vocab_offsets_by_measurement
    ESD.vocabulary_config.event_types_idxmap = event_types_idxmap

    print("Updated Vocabulary sizes by measurement:")
    print(ESD.vocabulary_config.vocab_sizes_by_measurement)
    print("Updated Vocabulary offsets by measurement:")
    print(ESD.vocabulary_config.vocab_offsets_by_measurement)
    print("Updated Event types index map:")
    print(ESD.vocabulary_config.event_types_idxmap)

    print("Saving dataset...")
    print("Just before saving the dataset:")
    print(f"Shape of ESD.dynamic_measurements_df: {ESD.dynamic_measurements_df.shape}")
    print(ESD.dynamic_measurements_df.head())
    print(type(ESD))
    ESD.save(do_overwrite=do_overwrite)
    print("Dataset saved.")

    # Print final vocabulary information
    print("Final Vocabulary sizes by measurement:", ESD.vocabulary_config.vocab_sizes_by_measurement)
    print("Final Vocabulary offsets by measurement:", ESD.vocabulary_config.vocab_offsets_by_measurement)
    print("Final Event types index map:", ESD.vocabulary_config.event_types_idxmap)

    # Print the contents of ESD
    print("Contents of ESD after saving:")
    print("Config:", ESD.config)
    print("Subjects DF:")
    print(ESD.subjects_df.head())
    print("Events DF:")
    print(ESD.events_df.head())
    print("Dynamic Measurements DF:")
    print(ESD.dynamic_measurements_df.head())

    print("Contents of Parquet files in data directory:")
    for parquet_file in data_dir.glob("*.parquet"):
        print(f"File: {parquet_file}")
        df = pl.read_parquet(parquet_file)
        print("Columns:", df.columns)
        print("Top 5 rows:")
        print(df.head(5))
        print()

    print("Contents of Parquet files in task_dfs directory:")
    task_dfs_dir = data_dir / "task_dfs"
    for parquet_file in task_dfs_dir.glob("*.parquet"):
        print(f"File: {parquet_file}")
        df = pl.read_parquet(parquet_file)
        print("Columns:", df.columns)
        print("Top 5 rows:")
        print(df.head(5))
        print()

    # Read and display the contents of the Parquet files
    print("Contents of Parquet files in DL_reps directory:")
    dl_reps_dir = ESD.config.save_dir / "DL_reps"
    if dl_reps_dir.exists():
        for split in ["train", "tuning", "held_out"]:
            parquet_files = list(dl_reps_dir.glob(f"{split}*.parquet"))
            if parquet_files:
                print(f"Parquet files for split '{split}':")
                for parquet_file in parquet_files:
                    print(f"File: {parquet_file}")
                    try:
                        df = read_parquet_file(parquet_file)
                        print(df.head())
                    except Exception as e:
                        print(f"Error reading Parquet file: {parquet_file}")
                        print(f"Error message: {str(e)}")
            else:
                print(f"No Parquet files found for split '{split}'.")
    else:
        print("DL_reps directory not found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Event Stream Data.')
    parser.add_argument('--use_labs', action='store_true', help='Include labs data in processing.')
    parser.add_argument('--save_schema', action='store_true', help='Save dataset schema to file.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: import only 1% of observations')
    args = parser.parse_args()

    main(use_labs=args.use_labs, save_schema=args.save_schema, debug=args.debug)