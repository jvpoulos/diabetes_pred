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
print(f"Polars version: {pl.__version__}")
from polars.datatypes import DataType
import pickle
from pathlib import Path
import polars.expr as expr
import pyarrow.parquet as pq
from pandas import Series

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
from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals, create_code_mapping, map_codes_to_indices, create_inverse_mapping, optimize_labs_data, process_events_and_measurements_df, try_convert_to_float, print_memory_usage, ConcreteDataset, create_static_vocabularies, split_event_type, create_static_indices_and_measurements, fit_scaler_on_training_data
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select
from collections import defaultdict
from EventStream.data.preprocessing.standard_scaler import StandardScaler
from EventStream.data.types import NumericDataModalitySubtype

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import tempfile
import shutil
import pyarrow.parquet as pq

import psutil

from visualize_describe import main as visualize_main
from build_task import main as build_task_main

def main(use_labs=False, debug=False):
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
            'InitialA1c': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.Series(),
            ),
            'AgeYears': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.Series(),
            ),
            'SDI_score': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
                _measurement_metadata=pd.Series(),
            ),
            'static_indices': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'static_measurement_indices': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'dynamic_indices': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'dynamic_values': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column='dynamic_values',
                _measurement_metadata=None,
            ),
            'Female': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'Married': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'GovIns': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'English': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
            'Veteran': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                _measurement_metadata=None,
            ),
        },
        normalizer_config={'cls': 'standard_scaler'},
        save_dir=data_dir
    )

    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/DiagnosesICD10.txt'
    procedures_file_path = 'data/ProceduresICD10.txt'
    if use_labs:
        labs_file_path = 'data/Labs_numeric.txt'

    # Process subjects data
    subjects_df = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select)
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

    # Create subject_id mapping
    subject_id_mapping = {study_id: idx for idx, study_id in enumerate(subjects_df['StudyID'].unique(), start=1)}
    subjects_df = subjects_df.with_columns([
        pl.col('StudyID').replace(subject_id_mapping).alias('subject_id').cast(pl.UInt32)
    ])

    # Create static vocabularies
    static_indices_vocab, static_measurement_indices_vocab = create_static_vocabularies(subjects_df)
    
    static_columns = ['InitialA1c', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']

    def map_to_index(col):
        mapping_dict = {str(k): v for k, v in static_indices_vocab[col].items()}
        expr = pl.when(pl.col(col).is_null()).then(None)
        for k, v in mapping_dict.items():
            expr = expr.when(pl.col(col).cast(pl.Utf8) == k).then(v)
        return expr.otherwise(None)

    # Create expressions for static_indices and static_measurement_indices
    static_indices_expr = pl.concat_list([map_to_index(col) for col in static_columns]).alias("static_indices")
    static_measurement_indices_expr = pl.concat_list([
        pl.when(pl.col(col).is_not_null())
        .then(pl.lit(static_measurement_indices_vocab[col]))
        .otherwise(None)
        for col in static_columns
    ]).alias("static_measurement_indices")

    # Apply the expressions to the DataFrame
    subjects_df = subjects_df.with_columns([
        static_indices_expr,
        static_measurement_indices_expr
    ])

    # Filter out None values and cast to UInt32
    subjects_df = subjects_df.with_columns([
        pl.col("static_indices").list.eval(pl.element().filter(pl.element().is_not_null())).cast(pl.List(pl.UInt32)),
        pl.col("static_measurement_indices").list.eval(pl.element().filter(pl.element().is_not_null())).cast(pl.List(pl.UInt32))
    ])

    print("Data types after processing:")
    print(subjects_df.dtypes)
    
    print("Shape of subjects_df after aggregation:", subjects_df.shape)
    print("Number of unique EMPIs:", subjects_df['StudyID'].n_unique())
    print("Columns in subjects_df:", subjects_df.columns)

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

    print("Saving diagnosis and procedure DataFrames as Parquet files...")
    df_dia.write_parquet(data_dir / "df_dia.parquet")
    df_prc.write_parquet(data_dir / "df_prc.parquet")

    if use_labs:
        print("Saving labs DataFrame as Parquet file...")
        df_labs.write_parquet(data_dir / "df_labs.parquet")


    print("Creating events dataframe...")
    # Process events and measurements
    df_dia_events, df_dia_dynamic = process_events_and_measurements_df(df_dia, "DIAGNOSIS", dia_columns_select, code_to_index, subject_id_mapping)
    df_prc_events, df_prc_dynamic = process_events_and_measurements_df(df_prc, "PROCEDURE", prc_columns_select, code_to_index, subject_id_mapping)

    if use_labs:
        df_labs_events, df_labs_dynamic = process_events_and_measurements_df(df_labs, "LAB", labs_columns_select, code_to_index, subject_id_mapping)
        
        print("Columns in df_dia_dynamic:", df_dia_dynamic.columns)
        print("Columns in df_prc_dynamic:", df_prc_dynamic.columns)
        print("Columns in df_labs_dynamic:", df_labs_dynamic.columns)
        
        # Add 'Result' column to diagnosis and procedure DataFrames
        df_dia_dynamic = df_dia_dynamic.with_columns(pl.lit(None).cast(pl.Float64).alias("Result"))
        df_prc_dynamic = df_prc_dynamic.with_columns(pl.lit(None).cast(pl.Float64).alias("Result"))
        
        # Ensure all DataFrames have the same column order and data types
        columns_order = ['event_id', 'subject_id', 'timestamp', 'dynamic_indices', 'Result', 'dynamic_values']
        df_dia_dynamic = df_dia_dynamic.select(columns_order).with_columns(pl.col("dynamic_values").cast(pl.Float64))
        df_prc_dynamic = df_prc_dynamic.select(columns_order).with_columns(pl.col("dynamic_values").cast(pl.Float64))
        df_labs_dynamic = df_labs_dynamic.select(columns_order).with_columns(pl.col("dynamic_values").cast(pl.Float64))
        
        events_df = pl.concat([df_dia_events, df_prc_events, df_labs_events], how='vertical')
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic, df_labs_dynamic], how='vertical')
    else:
        # Add 'Result' column to diagnosis and procedure DataFrames
        df_dia_dynamic = df_dia_dynamic.with_columns(pl.lit(None).cast(pl.Float64).alias("Result"))
        df_prc_dynamic = df_prc_dynamic.with_columns(pl.lit(None).cast(pl.Float64).alias("Result"))
        
        # Ensure both DataFrames have the same column order and data types
        columns_order = ['event_id', 'subject_id', 'timestamp', 'dynamic_indices', 'Result', 'dynamic_values']
        df_dia_dynamic = df_dia_dynamic.select(columns_order).with_columns(pl.col("dynamic_values").cast(pl.Float64))
        df_prc_dynamic = df_prc_dynamic.select(columns_order).with_columns(pl.col("dynamic_values").cast(pl.Float64))
        
        events_df = pl.concat([df_dia_events, df_prc_events], how='vertical')
        dynamic_measurements_df = pl.concat([df_dia_dynamic, df_prc_dynamic], how='vertical')

    # Print the final schema of dynamic_measurements_df
    print("Final schema of dynamic_measurements_df:")
    print(dynamic_measurements_df.schema)

    # Handle null timestamps
    events_df = events_df.with_columns([
        pl.col('timestamp').fill_null(pl.min('timestamp')).alias('timestamp'),
        pl.col('event_id').cast(pl.Int64),
        pl.col('event_type').cast(pl.Utf8)  # Convert Categorical to String
    ])

    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('timestamp').fill_null(pl.min('timestamp')).alias('timestamp'),
        pl.col('event_id').cast(pl.Int64)
    ])

    # Ensure event_type is not a combination in events_df
    events_df = events_df.with_columns([
        pl.col('event_type').str.split('&').list.first().alias('event_type')
    ])

    # Rename existing 'event_id' column
    events_df = events_df.rename({"event_id": "original_event_id"})
    dynamic_measurements_df = dynamic_measurements_df.rename({"event_id": "original_event_id"})

    # Sort and add new event_id
    events_df = events_df.sort(['subject_id', 'timestamp', 'original_event_id']).with_row_index("event_id")
    dynamic_measurements_df = dynamic_measurements_df.sort(['subject_id', 'timestamp', 'original_event_id']).with_row_index("event_id")

    # Make sure to cast the new event_id column to Int64:
    events_df = events_df.with_columns(pl.col('event_id').cast(pl.Int64))
    dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('event_id').cast(pl.Int64))

    # Optionally, drop the original_event_id column if it's no longer needed
    events_df = events_df.drop('original_event_id')
    dynamic_measurements_df = dynamic_measurements_df.drop('original_event_id')

    # Convert dynamic_values to float, replacing non-numeric values with 0
    dynamic_measurements_df = dynamic_measurements_df.with_columns(
        pl.col('dynamic_values').cast(pl.Float64, strict=False).fill_null(0)
    )

    # Fill null with 0 for Result column as well
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('Result').fill_null(0),
        pl.col('dynamic_values').fill_null(0)
    ])

    print("Data type of dynamic_values after conversion:", dynamic_measurements_df['dynamic_values'].dtype)

    # Update event types
    unique_event_types = set()
    for event_type in events_df['event_type'].unique():
        unique_event_types.update(event_type.split('&'))
    event_types = sorted(list(unique_event_types))
    event_types_idxmap = {event_type: idx for idx, event_type in enumerate(event_types, start=1)}

    events_df = events_df.with_columns([
        pl.col('event_type').map_elements(split_event_type, return_dtype=pl.Utf8).alias('event_type')
    ])

    # Update vocabulary configuration
    config.vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap),
        'dynamic_indices': len(code_to_index),
        'static_indices': len(static_indices_vocab),  # You'll need to create this vocabulary
        'static_measurement_indices': len(static_measurement_indices_vocab),  # You'll need to create this vocabulary
    }

    config.vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': len(event_types_idxmap) + 1,
        'static_indices': len(event_types_idxmap) + len(code_to_index) + 1,
        'static_measurement_indices': len(event_types_idxmap) + len(code_to_index) + len(static_indices_vocab) + 1,
    }

    config.measurements_idxmap = {
        'event_type': event_types_idxmap,
        'dynamic_indices': code_to_index,
        'static_indices': static_indices_vocab,
        'static_measurement_indices': static_measurement_indices_vocab,
    }

    config.measurements_per_generative_mode = {
        DataModality.MULTI_LABEL_CLASSIFICATION: ['dynamic_indices', 'static_indices', 'static_measurement_indices'],
        DataModality.MULTIVARIATE_REGRESSION: ['dynamic_values'],
        DataModality.UNIVARIATE_REGRESSION: ['InitialA1c', 'AgeYears', 'SDI_score'],
        DataModality.SINGLE_LABEL_CLASSIFICATION: ['Female', 'Married', 'GovIns', 'English', 'Veteran'],
    }
    config.event_types_idxmap = event_types_idxmap
    
    # Count of events by type
    event_type_counts = events_df.group_by('event_type').len()
    print("\nCount of events by type:")
    print(event_type_counts)

    # Save DataFrames as Parquet files
    subjects_df.write_parquet(data_dir / "subjects_df.parquet")
    events_df.write_parquet(data_dir / "events_df.parquet")
    dynamic_measurements_df.write_parquet(data_dir / "dynamic_measurements_df.parquet")

    # Save mappings to disk as JSON files
    with open(data_dir / 'code_to_index.json', 'w') as f:
        json.dump(code_to_index, f)
    with open(data_dir / 'index_to_code.json', 'w') as f:
        json.dump(index_to_code, f)

    # Now create the Dataset object after saving the files
    ESD = ConcreteDataset(
        config=config,
        subjects_df=subjects_df,
        events_df=events_df,
        dynamic_measurements_df=dynamic_measurements_df,
        code_mapping=code_to_index,
        static_indices_vocab=static_indices_vocab,
        static_measurement_indices_vocab=static_measurement_indices_vocab
    )

    # Split dataset
    ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)

    # Preprocess dataset
    ESD.preprocess()

    # Cache deep learning representation
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)

    # Save dataset
    ESD.save(do_overwrite=do_overwrite)

    # Print final information
    print("Final Vocabulary sizes by measurement:", ESD.vocabulary_config.vocab_sizes_by_measurement)
    print("Final Vocabulary offsets by measurement:", ESD.vocabulary_config.vocab_offsets_by_measurement)
    print("Final Event types index map:", ESD.vocabulary_config.event_types_idxmap)

    print("Final sample of dynamic_measurements_df:")
    print(dynamic_measurements_df.head())
    print("\nFinal columns in dynamic_measurements_df:")
    print(dynamic_measurements_df.columns)
    print("\nData types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    # Print the contents of ESD
    print("Contents of ESD after saving:")
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

    print("\nSummary statistics for DL_reps/train_0.parquet:")
    train_df = pl.read_parquet(data_dir / 'DL_reps' / 'train_0.parquet')
    print(train_df.describe())
    print("\nNumber of non-null dynamic_values:")
    print(train_df.filter(pl.col('dynamic_values').is_not_null()).shape[0])
    print("\nSample of non-null dynamic_values:")
    print(train_df.filter(pl.col('dynamic_values').is_not_null()).select('dynamic_values').head())

    print("\nSummary statistics for dynamic_measurements_df.parquet:")
    dynamic_measurements_df = pl.read_parquet(data_dir / 'dynamic_measurements_df.parquet')
    print(dynamic_measurements_df.describe())
    print("\nNumber of non-null dynamic_values:")
    print(dynamic_measurements_df.filter(pl.col('dynamic_values').is_not_null()).shape[0])
    print("\nSample of non-null dynamic_values:")
    print(dynamic_measurements_df.filter(pl.col('dynamic_values').is_not_null()).select('dynamic_values').head())

    # Build task and visualize
    build_task_main(ESD, use_labs=use_labs)
    visualize_main(ESD, use_labs=use_labs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Event Stream Data.')
    parser.add_argument('--use_labs', action='store_true', help='Include labs data in processing.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: import only 1% of observations')
    args = parser.parse_args()

    main(use_labs=args.use_labs, debug=args.debug)