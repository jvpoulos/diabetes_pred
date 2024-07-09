# Jason's working version


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

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def main(use_labs=True, save_schema=False, debug=False):
    print("Building dataset config...")
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
        save_dir=Path("data")
    )

    outcomes_file_path = '/home/sbanerjee/diabetes_pred/data/DiabetesOutcomes.txt'
    diagnoses_file_path = '/home/sbanerjee/diabetes_pred/data/Diagnoses_3yr.txt'
    procedures_file_path = '/home/sbanerjee/diabetes_pred/data/Procedures_3yr.txt'
    if use_labs:
        labs_file_path = '/home/sbanerjee/diabetes_pred/data/Labs_3yr.txt'

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
    subject_id_mapping = {study_id: idx for idx, study_id in enumerate(subjects_df['StudyID'].unique())}

    # Add the subject_id column
    #switching map_dict to replace
    subjects_df = subjects_df.with_columns([
        pl.col('StudyID').replace(subject_id_mapping, default=None).alias('subject_id')
    ])

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


    # Add event_type column
    df_dia = df_dia.with_columns([pl.lit('DIAGNOSIS').alias('event_type').cast(pl.Categorical)])
    df_prc = df_prc.with_columns([pl.lit('PROCEDURE').alias('event_type').cast(pl.Categorical)])
    if use_labs:
        df_labs = df_labs.with_columns([pl.lit('LAB').alias('event_type').cast(pl.Categorical)])


    # Add print statements for the number of unique mapped codes
    print(f"Number of unique mapped diagnosis codes: {df_dia['CodeWithType'].n_unique()}")
    print(f"Number of unique mapped procedures codes: {df_prc['CodeWithType'].n_unique()}")
    if use_labs:
        print(f"Number of unique mapped lab codes: {df_labs['Code'].n_unique()}")

    print("Unique mapped codes in df_dia:")
    print(df_dia['CodeWithType'].unique().sort().to_list())

    print("Unique mapped codes in df_prc:")
    print(df_prc['CodeWithType'].unique().sort().to_list())

    if use_labs:
        print("Unique mapped codes in df_labs:")
        print(df_labs['Code'].unique().sort().to_list())

    print("Creating events dataframe...")
    events_df = pl.concat([df_dia, df_prc], how='diagonal')

    if use_labs:
        # Ensure df_labs has the same structure as events_df
        df_labs_events = df_labs.with_columns([
            pl.col('Code').alias('CodeWithType'),
            pl.lit('LAB').alias('event_type').cast(pl.Categorical)
        ])
        events_df = pl.concat([events_df, df_labs_events], how='diagonal')

    events_df = events_df.with_columns([
        pl.col('Date').alias('timestamp'),
        pl.when(pl.col('event_type').is_null())
        .then(pl.when(pl.col('StudyID').is_in(df_dia['StudyID']))
              .then(pl.lit('DIAGNOSIS'))
              .otherwise(pl.lit('PROCEDURE')))
        .otherwise(pl.col('event_type'))
        .alias('event_type').cast(pl.Categorical)
    ])

    events_df = events_df.with_row_count('event_id')

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

    print("Creating code mapping...")
    code_to_index = create_code_mapping(df_dia, df_prc, df_labs if use_labs else None)

    print("Creating inverse mapping...")
    index_to_code = {idx: code for code, idx in code_to_index.items()}

    print(f"Total unique codes: {len(code_to_index)}")

    print("Mapping codes to indices...")
    df_dia = map_codes_to_indices(df_dia, code_to_index)
    df_prc = map_codes_to_indices(df_prc, code_to_index)
    if use_labs:
        df_labs = map_codes_to_indices(df_labs, code_to_index)

    print("Creating dynamic measurements dataframe...")
    # Add event_id if it doesn't exist, then ensure it's UInt32
    df_dia = df_dia.with_row_index("event_id") if "event_id" not in df_dia.columns else df_dia
    df_dia = df_dia.with_columns([
        pl.col('event_id').cast(pl.UInt32),
        pl.col('subject_id').cast(pl.Utf8) #string
    ])
    
    df_prc = df_prc.with_row_index("event_id") if "event_id" not in df_prc.columns else df_prc
    df_prc = df_prc.with_columns([
        pl.col('event_id').cast(pl.UInt32),
        pl.col('subject_id').cast(pl.Utf8) #string
    ])
    
    if use_labs:
        df_labs = df_labs.with_row_index("event_id") if "event_id" not in df_labs.columns else df_labs
        df_labs = df_labs.with_columns([
            pl.col('event_id').cast(pl.UInt32),
            pl.col('subject_id').cast(pl.Utf8) #string
        ])

    dynamic_measurements_df = pl.concat([df_dia, df_prc], how='diagonal')
    if use_labs:
        dynamic_measurements_df = pl.concat([dynamic_measurements_df, df_labs], how='diagonal')

    # Create a mapping dictionary from StudyID to subject_id
    subject_id_map = subjects_df.select(['StudyID', 'subject_id']).to_dict(as_series=False)
    subject_id_map = dict(zip(subject_id_map['StudyID'], subject_id_map['subject_id']))

    # Convert subject_id_map values to UInt32 using Polars Series
    subject_id_series = pl.Series(list(subject_id_map.values())).cast(pl.UInt32)
    subject_id_map = dict(zip(subject_id_map.keys(), subject_id_series))

    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('StudyID').map_dict(subject_id_map).alias('subject_id'), #not Uint32
        pl.col('Date').alias('timestamp'),
        pl.col('CodeWithType').alias('dynamic_indices'),
        pl.when(pl.col('StudyID').is_in(df_dia['StudyID']))
        .then(pl.lit('DIAGNOSIS'))
        .otherwise(pl.when(pl.col('StudyID').is_in(df_prc['StudyID']))
                   .then(pl.lit('PROCEDURE'))
                   .otherwise(pl.lit('LAB')))
        .alias('dynamic_indices_event_type').cast(pl.Categorical)
    ])

    # Ensure event_id is present and UInt32
    if 'event_id' not in dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_row_index('event_id')
    
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('event_id').cast(pl.UInt32),
        pl.col('subject_id').cast(pl.Utf8) #not Uint32
    ])

    print("Save measurements data as Parquet files")
    df_dia.write_parquet("data/df_dia.parquet")
    df_prc.write_parquet("data/df_prc.parquet")
    if use_labs:
        df_labs.write_parquet("data/df_labs.parquet")

    # Update these definitions
    vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap),
        'dynamic_indices': len(code_to_index),  # Changed from code_mapping to code_to_index
    }

    vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': len(event_types_idxmap),
    }

    print("Updating config with vocabulary sizes and offsets...")
    config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

    print("Updated Vocabulary sizes by measurement:", vocab_sizes_by_measurement)
    print("Updated Vocabulary offsets by measurement:", vocab_offsets_by_measurement)

    # Save mappings to disk as JSON files
    print("Saving mappings to disk...")
    with open('data/code_to_index.json', 'w') as f:
        json.dump(code_to_index, f)

    with open('data/index_to_code.json', 'w') as f:
        json.dump(index_to_code, f)

    print("Mappings saved successfully.")
    
    if use_labs:
        print("Processing labs data...")
        # Check if 'event_id' already exists in df_labs
        if 'event_id' not in df_labs.columns:
            df_labs = df_labs.with_row_index('event_id')
        
        # Ensure 'event_id' is of type UInt32
        df_labs = df_labs.with_columns([
            pl.col('event_id').cast(pl.UInt32),
            pl.col('subject_id').cast(pl.Utf8) #not Uint32
        ])

        df_labs = df_labs.with_columns([
            pl.col('StudyID'),
            pl.col('Date'),
            pl.col('Code').alias('CodeWithType'),
            pl.lit('LAB').alias('event_type').cast(pl.Categorical)
        ])
        df_labs = df_labs.join(subjects_df.select(['StudyID', 'subject_id']), on='StudyID', how='inner')

        # Ensure column names and types match with events_df
        df_labs = df_labs.select([
            pl.col('event_id').cast(events_df['event_id'].dtype),
            'StudyID',
            pl.col('Date').cast(events_df['Date'].dtype),
            'CodeWithType',
            pl.col('subject_id').cast(events_df['subject_id'].dtype),
            'Code',
            'Result',
            pl.col('event_type').cast(events_df['event_type'].dtype),
            pl.col('Date').alias('timestamp').cast(events_df['timestamp'].dtype)
        ])

        # Add labs data to events_df and dynamic_measurements_df
        events_df = pl.concat([events_df, df_labs])
        
        # Update dynamic_measurements_df with labs data
        df_labs_measurements = df_labs.with_columns([
            pl.col('CodeWithType').alias('dynamic_indices'),
            pl.col('event_type').alias('dynamic_indices_event_type')
        ])
        dynamic_measurements_df = pl.concat([
            dynamic_measurements_df, 
            df_labs_measurements.select(dynamic_measurements_df.columns)
        ], how='diagonal')

    # Build measurement_configs and track input schemas
    subject_id_col = 'StudyID'
    measurements_by_temporality = {
        TemporalityType.STATIC: {
            'outcomes': {
                DataModality.UNIVARIATE_REGRESSION: ['InitialA1c', 'SDI_score'],
                DataModality.SINGLE_LABEL_CLASSIFICATION: ['Female', 'Married', 'GovIns', 'English', 'AgeYears', 'Veteran']
            }
        },
        TemporalityType.DYNAMIC: {
            'diagnoses': {
                DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
            },
            'procedures': {
                DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
            },
        }
    }
    static_sources = defaultdict(dict)
    dynamic_sources = defaultdict(dict)
    measurement_configs = {}

    for temporality, sources_by_modality in measurements_by_temporality.items():
        schema_source = static_sources if temporality == TemporalityType.STATIC else dynamic_sources
        for source_name, modalities_by_measurement in sources_by_modality.items():
            for modality, measurements in modalities_by_measurement.items():
                if not measurements:
                    continue
                data_schema = schema_source[source_name]
                for m in measurements:
                    measurement_config_kwargs = {
                        "temporality": temporality,
                        "modality": modality,
                    }

                    if isinstance(m, dict):
                        m_dict = m
                        measurement_config_kwargs["name"] = m_dict.pop("name")
                        if m.get("values_column", None):
                            values_column = m_dict.pop("values_column")
                            m = [measurement_config_kwargs["name"], values_column]
                        else:
                            m = measurement_config_kwargs["name"]

                        measurement_config_kwargs.update(m_dict)

                    if isinstance(m, str) and modality == DataModality.UNIVARIATE_REGRESSION:
                        add_to_container(m, InputDataType.FLOAT, data_schema)
                    elif isinstance(m, list) and len(m) == 2 and isinstance(m[0], str) and isinstance(m[1], str) and modality == DataModality.MULTIVARIATE_REGRESSION:
                        add_to_container(m[0], InputDataType.CATEGORICAL, data_schema)
                        add_to_container(m[1], InputDataType.FLOAT, data_schema)
                        measurement_config_kwargs["values_column"] = m[1]
                        measurement_config_kwargs["name"] = m[0]
                    elif isinstance(m, str) and modality == DataModality.SINGLE_LABEL_CLASSIFICATION:
                        add_to_container(m, InputDataType.CATEGORICAL, data_schema)
                    elif isinstance(m, str) and modality == DataModality.MULTI_LABEL_CLASSIFICATION:
                        add_to_container(m, InputDataType.CATEGORICAL, data_schema)
                    else:
                        raise ValueError(f"{m}, {modality} invalid! Must be in {DataModality.values()}!")

                    if m in measurement_configs:
                        old = {k: v for k, v in measurement_configs[m].to_dict().items() if v is not None}
                        if old != measurement_config_kwargs:
                            raise ValueError(
                                f"{m} differs across input sources!\n{old}\nvs.\n{measurement_config_kwargs}"
                            )
                    else:
                        measurement_configs[m] = MeasurementConfig(**measurement_config_kwargs)

    # Add the columns to the 'col_schema' dictionary for the 'outcomes' input schema
    static_sources['outcomes']['StudyID'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['InitialA1c'] = InputDataType.FLOAT
    static_sources['outcomes']['Female'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['Married'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['GovIns'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['English'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['AgeYears'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['SDI_score'] = InputDataType.FLOAT
    static_sources['outcomes']['Veteran'] = InputDataType.CATEGORICAL

    static_sources['diagnoses']['CodeWithType'] = InputDataType.CATEGORICAL
    static_sources['procedures']['CodeWithType'] = InputDataType.CATEGORICAL

    static_sources['procedures']['StudyID'] = InputDataType.CATEGORICAL
    static_sources['diagnoses']['StudyID'] = InputDataType.CATEGORICAL

    # Build DatasetSchema
    connection_uri = None

    def build_schema(
        col_schema: dict[str, InputDataType],
        source_schema: dict[str, Any],
        schema_name: str,
        **extra_kwargs,
    ) -> InputDFSchema:
        input_schema_kwargs = {}

        if "input_df" in source_schema:
            if schema_name == 'outcomes':
                input_schema_kwargs["input_df"] = subjects_df
            elif schema_name == 'diagnoses':
                input_schema_kwargs["input_df"] = df_dia
            elif schema_name == 'procedures':
                input_schema_kwargs["input_df"] = df_prc
            elif schema_name == 'labs':
                input_schema_kwargs["input_df"] = df_labs
            else:
                raise ValueError(f"Unknown schema name: {schema_name}")
        else:
            raise ValueError("Must specify an input dataframe!")

        for param in (
            "start_ts_col",
            "end_ts_col",
            "ts_col",
            "event_type",
            "start_ts_format",
            "end_ts_format",
            "ts_format",
        ):
            if param in source_schema:
                input_schema_kwargs[param] = source_schema[param]

        if source_schema.get("start_ts_col", None):
            input_schema_kwargs["type"] = InputDFType.RANGE
        elif source_schema.get("ts_col", None):
            input_schema_kwargs["type"] = InputDFType.EVENT
        else:
            input_schema_kwargs["type"] = InputDFType.STATIC

        if input_schema_kwargs["type"] != InputDFType.STATIC and "event_type" not in input_schema_kwargs:
            event_type = schema_name.upper()
            input_schema_kwargs["event_type"] = event_type

        cols_covered = []
        any_schemas_present = False
        for n, cols_n in (
            ("start_data_schema", "start_columns"),
            ("end_data_schema", "end_columns"),
            ("data_schema", "columns"),
        ):
            if cols_n not in source_schema:
                continue
            cols = source_schema[cols_n]
            data_schema = {}

            for col in cols.items():
                in_name, out_val = col
                if isinstance(out_val, tuple):
                    out_name, out_type = out_val
                else:
                    out_name, out_type = out_val, col_schema.get(out_val)

                if out_type is None:
                    raise ValueError(f"Column {out_val} not found in col_schema: {col_schema}")

                cols_covered.append(out_name)
                add_to_container(in_name, (out_name, out_type), data_schema)

            input_schema_kwargs[n] = data_schema
            any_schemas_present = True

        if not any_schemas_present and (len(col_schema) > len(cols_covered)):
            input_schema_kwargs["data_schema"] = {}

        for col, dt in col_schema.items():
            if col in cols_covered:
                continue

            if col == 'SDI_score':
                col_schema[col] = InputDataType.FLOAT
            else:
                col_schema[col] = dt

            for schema in ("start_data_schema", "end_data_schema", "data_schema"):
                if schema in input_schema_kwargs:
                    input_schema_kwargs[schema][col] = dt

        must_have = source_schema.get("must_have", None)
        if must_have is None:
            pass
        elif isinstance(must_have, list):
            input_schema_kwargs["must_have"] = must_have
        elif isinstance(must_have, dict):
            must_have_processed = []
            for k, v in must_have.items():
                if v is True:
                    must_have_processed.append(k)
                elif isinstance(v, list):
                    must_have_processed.append((k, v))
                else:
                    raise ValueError(f"{v} invalid for `must_have`")
            input_schema_kwargs["must_have"] = must_have_processed
        else:
            raise ValueError("Unhandled `must_have` type")
       
        return InputDFSchema(**input_schema_kwargs, **extra_kwargs)

    inputs = {
        'outcomes': {
            'input_df': outcomes_file_path,
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'InitialA1c': ('InitialA1c', InputDataType.FLOAT),
                'Female': ('Female', InputDataType.CATEGORICAL),
                'Married': ('Married', InputDataType.CATEGORICAL),
                'GovIns': ('GovIns', InputDataType.CATEGORICAL),
                'English': ('English', InputDataType.CATEGORICAL),
                'AgeYears': ('AgeYears', InputDataType.CATEGORICAL),
                'SDI_score': ('SDI_score', InputDataType.FLOAT),
                'Veteran': ('Veteran', InputDataType.CATEGORICAL)
            }
        },
        'diagnoses': {
            'input_df': diagnoses_file_path,
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
            }
        },
        'procedures': {
            'input_df': procedures_file_path,
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
            }
        },
        'labs': {
            'input_df': labs_file_path,
            'event_type': 'LAB',
            'ts_col': 'Date',
            'ts_format': '%Y-%m-%d %H:%M:%S',
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'Code': ('Code', InputDataType.CATEGORICAL),
                'Result': ('Result', InputDataType.CATEGORICAL),
                'Date': ('Date', InputDataType.TIMESTAMP)
            }
        } if use_labs else []
    }

    dynamic_sources['diagnoses']['dynamic_indices'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['dynamic_indices'] = InputDataType.CATEGORICAL

    # Add the 'StudyID' column to the 'col_schema' dictionary for all dynamic input schemas
    dynamic_sources['diagnoses']['StudyID'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['StudyID'] = InputDataType.CATEGORICAL

    # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
    dynamic_sources['diagnoses']['Date'] = InputDataType.TIMESTAMP
    dynamic_sources['procedures']['Date'] = InputDataType.TIMESTAMP

    if use_labs:
        # Add the 'StudyID' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['StudyID'] = InputDataType.CATEGORICAL

        # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['Date'] = InputDataType.TIMESTAMP

        # Add the 'Result' column to the 'col_schema' dictionary for the 'labs' input schema
        dynamic_sources['labs']['Result'] = InputDataType.CATEGORICAL

        # Add the 'Code' column to the 'col_schema' dictionary for the 'labs' input schema
        dynamic_sources['labs']['Code'] = InputDataType.CATEGORICAL

    # Build DatasetSchema
    dynamic_input_schemas = []

    dynamic_input_schemas.append(
        InputDFSchema(
            input_df=df_dia,
            subject_id_col=None,  # Set to None for dynamic input schemas
            data_schema={
                'Date': InputDataType.TIMESTAMP
            },
            event_type='DIAGNOSIS',
            ts_col='Date',
            ts_format='%Y-%m-%d %H:%M:%S',
            type=InputDFType.EVENT
        )
    )

    dynamic_input_schemas.extend([
        InputDFSchema(
            input_df=df_dia,
            subject_id_col=None,
            data_schema={'CodeWithType': InputDataType.CATEGORICAL, 'Date': InputDataType.TIMESTAMP},
            event_type='DIAGNOSIS',
            ts_col='Date',
            ts_format='%Y-%m-%d %H:%M:%S',
            type=InputDFType.EVENT
        ),
        InputDFSchema(
            input_df=df_prc,
            subject_id_col=None,
            data_schema={'CodeWithType': InputDataType.CATEGORICAL, 'Date': InputDataType.TIMESTAMP},
            event_type='PROCEDURE',
            ts_col='Date',
            ts_format='%Y-%m-%d %H:%M:%S',
            type=InputDFType.EVENT
        )
    ])

    if use_labs:
        dynamic_input_schemas.append(
            InputDFSchema(
                input_df=df_labs,
                subject_id_col=None,
                data_schema={
                    'Code': InputDataType.CATEGORICAL,
                    'Result': InputDataType.CATEGORICAL,
                    'timestamp': InputDataType.TIMESTAMP
                },
                event_type='LAB',
                ts_col='timestamp',
                ts_format='%Y-%m-%d %H:%M:%S',
                type=InputDFType.EVENT
            )
        )
    
    if use_labs:
        df_labs_empi = df_labs['StudyID'].unique()

    print("Saving dataframes as Parquet files")
    dynamic_measurements_df.write_parquet("data/dynamic_measurements_df.parquet")
    events_df.write_parquet("data/events_df.parquet")
    subjects_df.write_parquet("data/subjects_df.parquet")

    print("Processing events and measurements data...")
    temp_dataset = Dataset(config, subjects_df=subjects_df, events_df=events_df, dynamic_measurements_df=dynamic_measurements_df)

    events_df_dia, dynamic_measurements_df_dia = temp_dataset._process_events_and_measurements_df(
        df_dia, "DIAGNOSIS", {"CodeWithType": ("CodeWithType", InputDataType.CATEGORICAL)}
    )
    events_df_prc, dynamic_measurements_df_prc = temp_dataset._process_events_and_measurements_df(
        df_prc, "PROCEDURE", {"CodeWithType": ("CodeWithType", InputDataType.CATEGORICAL)}
    )
    if use_labs:
        events_df_lab, dynamic_measurements_df_lab = temp_dataset._process_events_and_measurements_df(
            df_labs, "LAB", {"Code": ("Code", InputDataType.CATEGORICAL), "Result": ("Result", InputDataType.CATEGORICAL)}
        )
        # Rename 'Code' to 'CodeWithType' for consistency if 'Code' exists
        if 'Code' in dynamic_measurements_df_lab.columns:
            dynamic_measurements_df_lab = dynamic_measurements_df_lab.rename({"Code": "CodeWithType"})
        dynamic_measurements_df = pl.concat([dynamic_measurements_df_dia, dynamic_measurements_df_prc, dynamic_measurements_df_lab], how="diagonal")
    else:
        dynamic_measurements_df = pl.concat([dynamic_measurements_df_dia, dynamic_measurements_df_prc], how="diagonal")

    print("Columns in final dynamic_measurements_df:", dynamic_measurements_df.columns)
    print("Data types in final dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    print("Sample of StudyID column:")
    print(dynamic_measurements_df.select('subject_id').head()) #subject_id

    temp_dataset.dynamic_measurements_df = dynamic_measurements_df
    
    print("Before _convert_dynamic_indices_to_indices")
    print("Data types in dynamic_measurements_df:")
    for col in temp_dataset.dynamic_measurements_df.columns:
        print(f"{col}: {temp_dataset.dynamic_measurements_df[col].dtype}")
    
    temp_dataset._convert_dynamic_indices_to_indices()
    
    print("After _convert_dynamic_indices_to_indices")
    print("Data types in dynamic_measurements_df:")
    for col in temp_dataset.dynamic_measurements_df.columns:
        print(f"{col}: {temp_dataset.dynamic_measurements_df[col].dtype}")

    print("Sample of dynamic_measurements_df:")
    print(temp_dataset.dynamic_measurements_df.head())

    # Instead of using with_columns, let's create a new DataFrame with the desired columns and types
    dynamic_measurements_df = pl.DataFrame({
        'event_id': temp_dataset.dynamic_measurements_df['event_id'],
        'subject_id': temp_dataset.dynamic_measurements_df['subject_id'], #subject_id
        'timestamp': temp_dataset.dynamic_measurements_df['timestamp'],
        'dynamic_indices': temp_dataset.dynamic_measurements_df['dynamic_indices'].cast(pl.UInt32),
        'CodeWithType': temp_dataset.dynamic_measurements_df['CodeWithType'], #CodeWithType
    })

    # Add dynamic_counts column if it exists, otherwise create it with default value 1
    if 'dynamic_counts' in temp_dataset.dynamic_measurements_df.columns:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.col('dynamic_counts').fill_null(1).cast(pl.UInt32)
        ])
    else:
        dynamic_measurements_df = dynamic_measurements_df.with_columns([
            pl.lit(1).cast(pl.UInt32).alias('dynamic_counts')
        ])

    print("Final data types in dynamic_measurements_df:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    print("Sample of final dynamic_measurements_df:")
    print(dynamic_measurements_df.head())

    dynamic_measurements_df = pl.concat([
        df_dia.select('StudyID', 'CodeWithType', 'Date'),
        df_prc.select('StudyID', 'CodeWithType', 'Date')
    ])

    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('StudyID').alias('subject_id').cast(pl.Utf8), #not Unit32
        pl.col('Date').alias('timestamp'),
        pl.col('CodeWithType').alias('dynamic_indices'),  # Keep as string
        pl.when(pl.col('StudyID').is_in(df_dia['StudyID']))
        .then(pl.lit('DIAGNOSIS'))
        .otherwise(pl.lit('PROCEDURE'))
        .alias('dynamic_indices_event_type').cast(pl.Categorical)
    ])

    dynamic_measurements_df = dynamic_measurements_df.join(subjects_df.select(['StudyID', 'subject_id']), on='StudyID', how='inner')

    dynamic_measurements_df = dynamic_measurements_df.with_row_index('event_id')

    subjects_df = subjects_df.with_columns(pl.col('StudyID').alias('subject_id').cast(pl.Utf8)) #not Unit32

    print("Shape of subjects_df:", subjects_df.shape)
    print("Columns of subjects_df:", subjects_df.columns)

    if use_labs:
        dynamic_measurements_df = df_labs.select('StudyID', 'Code', 'Result', 'Date').rename({'StudyID': 'subject_id', 'Date': 'timestamp'})
        dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('subject_id').cast(pl.Utf8)) #not Unit32
        dynamic_measurements_df = (
            dynamic_measurements_df
            .group_by(['subject_id', 'timestamp'])
            .agg(pl.col('Result').count().alias('tmp_event_id'))
            .drop('tmp_event_id')
            .with_row_index('tmp_event_id')
            .rename({'tmp_event_id': 'event_id'})
        )

    event_types = events_df['event_type'].unique().to_list()
    event_types_idxmap = {event_type: idx for idx, event_type in enumerate(event_types, start=1)}

    print("Event types index map:")
    print(event_types_idxmap)

    print("Updating vocabulary sizes and offsets...")
    vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap),
        'dynamic_indices': len(code_to_index),
    }

    vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': len(event_types_idxmap),
    }

    print("Updating config with vocabulary sizes and offsets...")
    config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

    print(f"Number of unique mapped diagnosis codes: {df_dia['dynamic_indices'].n_unique()}")
    print(f"Number of unique mapped procedures codes: {df_prc['dynamic_indices'].n_unique()}")

    print("Checking for null or UNKNOWN_CODE values in dynamic_measurements_df")
    null_count = dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
    unknown_count = dynamic_measurements_df.filter(pl.col('dynamic_indices') == 'UNKNOWN_CODE').shape[0]
    print(f"Null values: {null_count}")
    print(f"UNKNOWN_CODE values: {unknown_count}")
    print(f"Sample of dynamic_measurements_df:\n{dynamic_measurements_df.head()}")

    print_memory_usage()

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

    # Create the Dataset object with eager DataFrames
    print("Creating Dataset object...")

    # Before creating the Dataset object
    subjects_df = subjects_df.collect() if isinstance(subjects_df, pl.LazyFrame) else subjects_df
    events_df = events_df.collect() if isinstance(events_df, pl.LazyFrame) else events_df
    dynamic_measurements_df = dynamic_measurements_df.collect() if isinstance(dynamic_measurements_df, pl.LazyFrame) else dynamic_measurements_df

    # Ensure subject_id and event_id are UInt32 in all dataframes
    subjects_df = subjects_df.with_columns([pl.col('subject_id')]) #not Unit32
    events_df = events_df.with_columns([
        pl.col('subject_id').cast(pl.Utf8), #not Unit32
        pl.col('event_id').cast(pl.UInt32)
    ])
    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('subject_id').cast(pl.Utf8), #not Unit32
        pl.col('event_id').cast(pl.UInt32)
    ])


    ESD = Dataset(
    config=config,
    subjects_df=subjects_df,
    events_df=events_df,
    dynamic_measurements_df=dynamic_measurements_df,
    code_mapping=code_to_index
)

    print("Dataset object created.")
    print_memory_usage()

    print("Splitting dataset...")
    ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
    print("Dataset split.")

    print("Preprocessing dataset...")
    ESD.preprocess()
    print("Dataset preprocessed.")

    print("Caching deep learning representation...")
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)
    print("Deep learning representation cached.")
    print_memory_usage()

    print("Add epsilon to concurrent timestamps")
    eps = np.finfo(float).eps

    # Handle null timestamp values by filling them with a default value
    default_timestamp = datetime(2000, 1, 1)  # Use Python's datetime

    if 'timestamp' not in ESD.events_df.columns:
        ESD.events_df = ESD.events_df.with_columns([
            pl.lit(default_timestamp).alias('timestamp')
        ])
    else:
        ESD.events_df = ESD.events_df.with_columns([
            pl.col('timestamp').fill_null(default_timestamp).cast(pl.Datetime).alias('timestamp')
        ])

    # Explicitly cast the event_id to a numeric type if it's not already
    ESD.events_df = ESD.events_df.with_columns([
        pl.col('event_id').cast(pl.Float64)
    ])

    ESD.events_df = ESD.events_df.with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
    ])

    if 'timestamp' not in ESD.dynamic_measurements_df.columns:
        ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
            pl.lit(default_timestamp).alias('timestamp')
        ])
    else:
        ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
            pl.col('timestamp').fill_null(default_timestamp).cast(pl.Datetime).alias('timestamp')
        ])

    # Explicitly cast the event_id to a numeric type if it's not already
    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
        pl.col('event_id').cast(pl.Float64)
    ])

    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
    ])

    # Update the vocabulary configuration
    ESD.vocabulary_config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    ESD.vocabulary_config.vocab_offsets_by_measurement = vocab_offsets_by_measurement
    ESD.vocabulary_config.event_types_idxmap = event_types_idxmap


    # Ensure 'UNKNOWN_CODE' is properly handled
    UNKNOWN_CODE = -1  # Define a numeric value to represent 'UNKNOWN_CODE'

    # Convert any 'UNKNOWN_CODE' string values to the numeric UNKNOWN_CODE before filtering
    if 'dynamic_indices' in ESD.dynamic_measurements_df.columns:
        ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
            pl.when(pl.col('dynamic_indices') == UNKNOWN_CODE).then(UNKNOWN_CODE).otherwise(pl.col('dynamic_indices')).cast(pl.Int64).alias('dynamic_indices')
        ])

    print("Updated Vocabulary sizes by measurement:")
    print(ESD.vocabulary_config.vocab_sizes_by_measurement)
    print("Updated Vocabulary offsets by measurement:")
    print(ESD.vocabulary_config.vocab_offsets_by_measurement)
    print("Updated Event types index map:")
    print(ESD.vocabulary_config.event_types_idxmap)

    print("Checking dynamic_indices in ESD.dynamic_measurements_df")
    null_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
    unknown_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices') == UNKNOWN_CODE).shape[0]
    print(f"Null values: {null_count}")
    print(f"UNKNOWN_CODE values: {unknown_count}")
    print(f"Sample of ESD.dynamic_measurements_df:\n{ESD.dynamic_measurements_df.head()}")

    print("Saving dataset...")
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
    data_dir = Path("data")
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






#     ESD = Dataset(
#         config=config,
#         subjects_df=subjects_df,
#         events_df=events_df,
#         dynamic_measurements_df=dynamic_measurements_df,
#         code_mapping=code_to_index
#     )

#     print("Dataset object created.")
#     print_memory_usage()

#     print("Splitting dataset...")
#     ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
#     print("Dataset split.")

#     print("Preprocessing dataset...")
#     ESD.preprocess()
#     print("Dataset preprocessed.")

#     print("Caching deep learning representation...")
#     ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)
#     print("Deep learning representation cached.")
#     print_memory_usage()

#     print("Add epsilon to concurrent timestamps")
#     eps = np.finfo(float).eps

#     # Handle null timestamp values by filling them with a default value
#     default_timestamp = datetime(2000, 1, 1)  # Use Python's datetime

#     if 'timestamp' not in ESD.events_df.columns:
#         ESD.events_df = ESD.events_df.with_columns([
#             pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
#         ])
#     else:
#         ESD.events_df = ESD.events_df.with_columns([
#             pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
#         ])

#     ESD.events_df = ESD.events_df.with_columns([
#         (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
#     ])

#     if 'timestamp' not in ESD.dynamic_measurements_df.columns:
#         ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
#             pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
#         ])
#     else:
#         ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
#             pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
#         ])

#     ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
#         (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
#     ])

#     # Update the vocabulary configuration
#     ESD.vocabulary_config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
#     ESD.vocabulary_config.vocab_offsets_by_measurement = vocab_offsets_by_measurement
#     ESD.vocabulary_config.event_types_idxmap = event_types_idxmap

#     print("Updated Vocabulary sizes by measurement:")
#     print(ESD.vocabulary_config.vocab_sizes_by_measurement)
#     print("Updated Vocabulary offsets by measurement:")
#     print(ESD.vocabulary_config.vocab_offsets_by_measurement)
#     print("Updated Event types index map:")
#     print(ESD.vocabulary_config.event_types_idxmap)

#     print("Checking dynamic_indices in ESD.dynamic_measurements_df")
#     null_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
#     unknown_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices') == 'UNKNOWN_CODE').shape[0]
#     print(f"Null values: {null_count}")
#     print(f"UNKNOWN_CODE values: {unknown_count}")
#     print(f"Sample of ESD.dynamic_measurements_df:\n{ESD.dynamic_measurements_df.head()}")

#     print("Saving dataset...")
#     print(type(ESD))
#     ESD.save(do_overwrite=do_overwrite)
#     print("Dataset saved.")

#     # Print final vocabulary information
#     print("Final Vocabulary sizes by measurement:", ESD.vocabulary_config.vocab_sizes_by_measurement)
#     print("Final Vocabulary offsets by measurement:", ESD.vocabulary_config.vocab_offsets_by_measurement)
#     print("Final Event types index map:", ESD.vocabulary_config.event_types_idxmap)

#     # Print the contents of ESD
#     print("Contents of ESD after saving:")
#     print("Config:", ESD.config)
#     print("Subjects DF:")
#     print(ESD.subjects_df.head())
#     print("Events DF:")
#     print(ESD.events_df.head())
#     print("Dynamic Measurements DF:")
#     print(ESD.dynamic_measurements_df.head())

#     print("Contents of Parquet files in data directory:")
#     data_dir = Path("data")
#     for parquet_file in data_dir.glob("*.parquet"):
#         print(f"File: {parquet_file}")
#         df = pl.read_parquet(parquet_file)
#         print("Columns:", df.columns)
#         print("Top 5 rows:")
#         print(df.head(5))
#         print()

#     print("Contents of Parquet files in task_dfs directory:")
#     task_dfs_dir = data_dir / "task_dfs"
#     for parquet_file in task_dfs_dir.glob("*.parquet"):
#         print(f"File: {parquet_file}")
#         df = pl.read_parquet(parquet_file)
#         print("Columns:", df.columns)
#         print("Top 5 rows:")
#         print(df.head(5))
#         print()

#     # Read and display the contents of the Parquet files
#     print("Contents of Parquet files in DL_reps directory:")
#     dl_reps_dir = ESD.config.save_dir / "DL_reps"
#     if dl_reps_dir.exists():
#         for split in ["train", "tuning", "held_out"]:
#             parquet_files = list(dl_reps_dir.glob(f"{split}*.parquet"))
#             if parquet_files:
#                 print(f"Parquet files for split '{split}':")
#                 for parquet_file in parquet_files:
#                     print(f"File: {parquet_file}")
#                     try:
#                         df = read_parquet_file(parquet_file)
#                         print(df.head())
#                     except Exception as e:
#                         print(f"Error reading Parquet file: {parquet_file}")
#                         print(f"Error message: {str(e)}")
#             else:
#                 print(f"No Parquet files found for split '{split}'.")
#     else:
#         print("DL_reps directory not found.")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process Event Stream Data.')
#     parser.add_argument('--use_labs', action='store_true', help='Include labs data in processing.')
#     parser.add_argument('--save_schema', action='store_true', help='Save dataset schema to file.')
#     parser.add_argument('--debug', action='store_true', help='Debug mode: import only 1% of observations')
#     args = parser.parse_args()

#     main(use_labs=args.use_labs, save_schema=args.save_schema, debug=args.debug)



# Sujay's working version: 

# import pandas as pd
# import torch
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.impute import SimpleImputer
# from torch.utils.data import TensorDataset
# import io
# from tqdm import tqdm
# import numpy as np
# import gc
# from math import ceil
# import json
# import re
# import scipy.sparse as sp
# from scipy.sparse import csr_matrix, hstack, vstack
# from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_sparse
# from joblib import Parallel, delayed
# import multiprocessing
# import concurrent.futures
# import argparse
# import dask
# import dask.dataframe as dd
# import dask.array as da
# from dask.diagnostics import ProgressBar
# from dask.distributed import Client, as_completed
# from typing import Any
# import polars as pl
# from polars.datatypes import DataType
# import pickle
# from pathlib import Path

# from EventStream.data.config import (
#     InputDFSchema,
#     MeasurementConfig,
# )
# from EventStream.data.dataset_config import DatasetConfig

# from EventStream.data.dataset_schema import DatasetSchema

# from EventStream.data.dataset_polars import Dataset

# from EventStream.data.types import (
#     DataModality,
#     InputDataType,
#     InputDFType,
#     TemporalityType,
# )
# from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals
# from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select
# from collections import defaultdict
# from EventStream.data.preprocessing.standard_scaler import StandardScaler

# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# import os
# import tempfile
# import shutil
# import pyarrow.parquet as pq

# #sujay

# def main(use_dask=False, use_labs=True):
#     outcomes_file_path = '/home/sbanerjee/diabetes_pred/data/DiabetesOutcomes.txt'
#     diagnoses_file_path = '/home/sbanerjee/diabetes_pred/data/Diagnoses_3yr.txt'
#     procedures_file_path = '/home/sbanerjee/diabetes_pred/data/Procedures_3yr.txt'

#     if use_labs:
#         labs_file_path = '/home/sbanerjee/diabetes_pred/data/Labs_3yr.txt'


#     df_outcomes = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select)
#     df_outcomes = df_outcomes if isinstance(df_outcomes, pl.DataFrame) else pl.from_pandas(df_outcomes)

#     print("Outcomes DataFrame columns:", df_outcomes.columns)

#     if df_outcomes.is_empty():
#         raise ValueError("Outcomes DataFrame is empty.")


#     df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select)# deleting min freq)

#     df_dia = df_dia if isinstance(df_dia, pl.DataFrame) else pl.from_pandas(df_dia)

#     if df_dia.is_empty():
#         raise ValueError("Diagnoses DataFrame is empty.")


#     df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select) # deleting min freq)

#     df_prc = df_prc if isinstance(df_prc, pl.DataFrame) else pl.from_pandas(df_prc)

#     if df_prc.is_empty():
#         raise ValueError("Procedures DataFrame is empty.")

#     if use_labs:

#         df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, chunk_size=200000) # deleting min freq)

#         df_labs = df_labs if isinstance(df_labs, pl.DataFrame) else pl.from_pandas(df_labs)

#         if df_labs.is_empty():
#             raise ValueError("Labs DataFrame is empty.")
        
#         print("Columns in df_labs:", df_labs.columns)
        
#     # Build measurement_configs and track input schemas
#     #changed EMPI to StudyID
#     #changed subject_id to StudyID
#     subject_id_col = 'StudyID'
#     measurements_by_temporality = {
#         TemporalityType.STATIC: {
#             'outcomes': {
#                 # DataModality.SINGLE_LABEL_CLASSIFICATION: ['A1cGreaterThan7'],
#                 DataModality.UNIVARIATE_REGRESSION: ['InitialA1c', 'SDI_score'],
#                 DataModality.SINGLE_LABEL_CLASSIFICATION: ['Female', 'Married', 'GovIns', 'English', 'AgeYears', 'Veteran']
#             }
#         },
#         TemporalityType.DYNAMIC: {
#             'diagnoses': {
#                 DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
#             },
#             'procedures': {
#                 DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
#             },
#         }
#     }
#     static_sources = defaultdict(dict)
#     dynamic_sources = defaultdict(dict)
#     measurement_configs = {}

#     for temporality, sources_by_modality in measurements_by_temporality.items():
#         schema_source = static_sources if temporality == TemporalityType.STATIC else dynamic_sources
#         for source_name, modalities_by_measurement in sources_by_modality.items():
#             for modality, measurements in modalities_by_measurement.items():
#                 if not measurements:
#                     continue
#                 data_schema = schema_source[source_name]
#                 for m in measurements: #adding conditional to deal with presence of CodeWithType
#                     if m == 'CodeWithType' and 'CodeWithType' not in data_schema:
#                             continue  # Skip if CodeWithType is not present

#                         measurement_config_kwargs = {
#                             "temporality": temporality,
#                             "modality": modality,
#                         }

#                     if isinstance(m, dict):
#                         m_dict = m
#                         measurement_config_kwargs["name"] = m_dict.pop("name")
#                         if m.get("values_column", None):
#                             values_column = m_dict.pop("values_column")
#                             m = [measurement_config_kwargs["name"], values_column]
#                         else:
#                             m = measurement_config_kwargs["name"]

#                         measurement_config_kwargs.update(m_dict)

#                     # Refactored code without match-case
#                     if isinstance(m, str) and modality == DataModality.UNIVARIATE_REGRESSION:
#                         add_to_container(m, InputDataType.FLOAT, data_schema)
#                     elif isinstance(m, list) and len(m) == 2 and isinstance(m[0], str) and isinstance(m[1], str) and modality == DataModality.MULTIVARIATE_REGRESSION:
#                         add_to_container(m[0], InputDataType.CATEGORICAL, data_schema)
#                         add_to_container(m[1], InputDataType.FLOAT, data_schema)
#                         measurement_config_kwargs["values_column"] = m[1]
#                         measurement_config_kwargs["name"] = m[0]
#                     elif isinstance(m, str) and modality == DataModality.SINGLE_LABEL_CLASSIFICATION:
#                         add_to_container(m, InputDataType.CATEGORICAL, data_schema)
#                     elif isinstance(m, str) and modality == DataModality.MULTI_LABEL_CLASSIFICATION:
#                         add_to_container(m, InputDataType.CATEGORICAL, data_schema)
#                     else:
#                         raise ValueError(f"{m}, {modality} invalid! Must be in {DataModality.values()}!")

#                     if m in measurement_configs:
#                         old = {k: v for k, v in measurement_configs[m].to_dict().items() if v is not None}
#                         if old != measurement_config_kwargs:
#                             raise ValueError(
#                                 f"{m} differs across input sources!\n{old}\nvs.\n{measurement_config_kwargs}"
#                             )
#                     else:
#                         measurement_configs[m] = MeasurementConfig(**measurement_config_kwargs)

#     # Add the columns to the 'col_schema' dictionary for the 'outcomes' input schema
#     #changed EMPI to StudyID
#     static_sources['outcomes']['StudyID'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['InitialA1c'] = InputDataType.FLOAT
#     static_sources['outcomes']['Female'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['Married'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['GovIns'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['English'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['AgeYears'] = InputDataType.CATEGORICAL
#     static_sources['outcomes']['SDI_score'] = InputDataType.FLOAT
#     static_sources['outcomes']['Veteran'] = InputDataType.CATEGORICAL

#     static_sources['diagnoses']['CodeWithType'] = InputDataType.CATEGORICAL
#     static_sources['procedures']['CodeWithType'] = InputDataType.CATEGORICAL
#     #changed EMPI to StudyID
#     static_sources['procedures']['StudyID'] = InputDataType.CATEGORICAL
#     static_sources['diagnoses']['StudyID'] = InputDataType.CATEGORICAL

#     # Build DatasetSchema
#     connection_uri = None
    
#     def build_schema(
#         col_schema: dict[str, InputDataType],
#         source_schema: dict[str, Any],
#         schema_name: str,
#         **extra_kwargs,
#     ) -> InputDFSchema:
#         input_schema_kwargs = {}

#         if "input_df" in source_schema:
#             if schema_name == 'outcomes':
#                 input_schema_kwargs["input_df"] = df_outcomes
#             elif schema_name == 'diagnoses':
#                 input_schema_kwargs["input_df"] = df_dia
#             elif schema_name == 'procedures':
#                 input_schema_kwargs["input_df"] = df_prc
#             elif schema_name == 'labs':
#                 input_schema_kwargs["input_df"] = df_labs
#             else:
#                 raise ValueError(f"Unknown schema name: {schema_name}")
#         else:
#             raise ValueError("Must specify an input dataframe!")

#         for param in (
#             "start_ts_col",
#             "end_ts_col",
#             "ts_col",
#             "event_type",
#             "start_ts_format",
#             "end_ts_format",
#             "ts_format",
#         ):
#             if param in source_schema:
#                 input_schema_kwargs[param] = source_schema[param]

#         if source_schema.get("start_ts_col", None):
#             input_schema_kwargs["type"] = InputDFType.RANGE
#         elif source_schema.get("ts_col", None):
#             input_schema_kwargs["type"] = InputDFType.EVENT
#         else:
#             input_schema_kwargs["type"] = InputDFType.STATIC

#         if input_schema_kwargs["type"] != InputDFType.STATIC and "event_type" not in input_schema_kwargs:
#             event_type = schema_name.upper()
#             input_schema_kwargs["event_type"] = event_type

#         cols_covered = []
#         any_schemas_present = False
#         for n, cols_n in (
#             ("start_data_schema", "start_columns"),
#             ("end_data_schema", "end_columns"),
#             ("data_schema", "columns"),
#         ):
#             if cols_n not in source_schema:
#                 continue
#             cols = source_schema[cols_n]
#             data_schema = {}

#             for col in cols.items():
#                 in_name, out_val = col
#                 if isinstance(out_val, tuple):
#                     out_name, out_type = out_val
#                 else:
#                     out_name, out_type = out_val, col_schema.get(out_val)

#                 if out_type is None:
#                     raise ValueError(f"Column {out_val} not found in col_schema: {col_schema}")

#                 cols_covered.append(out_name)
#                 add_to_container(in_name, (out_name, out_type), data_schema)

#             input_schema_kwargs[n] = data_schema
#             any_schemas_present = True

#         if not any_schemas_present and (len(col_schema) > len(cols_covered)):
#             input_schema_kwargs["data_schema"] = {}

#         for col, dt in col_schema.items():
#             if col in cols_covered:
#                 continue

#             if col == 'SDI_score':
#                 col_schema[col] = InputDataType.FLOAT
#             else:
#                 col_schema[col] = dt

#             for schema in ("start_data_schema", "end_data_schema", "data_schema"):
#                 if schema in input_schema_kwargs:
#                     input_schema_kwargs[schema][col] = dt

#         must_have = source_schema.get("must_have", None)
#         if must_have is None:
#             pass
#         elif isinstance(must_have, list):
#             input_schema_kwargs["must_have"] = must_have
#         elif isinstance(must_have, dict):
#             must_have_processed = []
#             for k, v in must_have.items():
#                 if v is True:
#                     must_have_processed.append(k)
#                 elif isinstance(v, list):
#                     must_have_processed.append((k, v))
#                 else:
#                     raise ValueError(f"{v} invalid for `must_have`")
#             input_schema_kwargs["must_have"] = must_have_processed
#         else:
#             raise ValueError("Unhandled `must_have` type")
       
#         return InputDFSchema(**input_schema_kwargs, **extra_kwargs)
    
#     #changed EMPI to StudyID
#     inputs = {
#         'outcomes': {
#             'input_df': outcomes_file_path,
#             'columns': {
#                 'StudyID': ('StudyID', InputDataType.CATEGORICAL),
#                 'InitialA1c': ('InitialA1c', InputDataType.FLOAT),
#                 'Female': ('Female', InputDataType.CATEGORICAL),
#                 'Married': ('Married', InputDataType.CATEGORICAL),
#                 'GovIns': ('GovIns', InputDataType.CATEGORICAL),
#                 'English': ('English', InputDataType.CATEGORICAL),
#                 'AgeYears': ('AgeYears', InputDataType.CATEGORICAL),
#                 'SDI_score': ('SDI_score', InputDataType.FLOAT),
#                 'Veteran': ('Veteran', InputDataType.CATEGORICAL)
#             }
#         },
#         'diagnoses': {
#             'input_df': diagnoses_file_path,
#             'columns': { #change from EMPI to StudyID
#                 'StudyID': ('StudyID', InputDataType.CATEGORICAL),
#                 'Date': ('Date', InputDataType.TIMESTAMP),
#                 'Code': ('Code', InputDataType.CATEGORICAL),
#                 'Code_Type': ('Code_Type', InputDataType.CATEGORICAL),
#                 'IndexDate': ('IndexDate', InputDataType.TIMESTAMP),
#                 'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)

#             }
#         },
#         'procedures': {
#             'input_df': procedures_file_path,
#             'columns': {
#                 'StudyID': ('StudyID', InputDataType.CATEGORICAL),
#                 'Date': ('Date', InputDataType.TIMESTAMP),
#                 'Code': ('Code', InputDataType.CATEGORICAL),
#                 'Code_Type': ('Code_Type', InputDataType.CATEGORICAL),
#                 'IndexDate': ('IndexDate', InputDataType.TIMESTAMP),
#                 'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
#             }
#         }, #fix all these
#         'labs': {
#             'input_df': labs_file_path,
#             'event_type': 'LAB',
#             'ts_col': 'Date',
#             'ts_format': '%Y-%m-%d %H:%M:%S',
#             'columns': {
#                 'StudyID': ('StudyID', InputDataType.CATEGORICAL),
#                 'Date': ('Date', InputDataType.TIMESTAMP),
#                 'Code': ('Code', InputDataType.CATEGORICAL),
#                 'Result': ('Result', InputDataType.CATEGORICAL),
#                 'Source': ('CodeWithType', InputDataType.CATEGORICAL),
#                 'IndexDate': ('IndexDate', InputDataType.TIMESTAMP)
#             }
#         } if use_labs else []
#     }

#     #changed dynamic_indices to CodeWithType
#     dynamic_sources['diagnoses']['CodeWithType'] = InputDataType.CATEGORICAL
#     dynamic_sources['procedures']['CodeWithType'] = InputDataType.CATEGORICAL

#     # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
#     #changed to StudyID
#     dynamic_sources['diagnoses']['StudyID'] = InputDataType.CATEGORICAL
#     dynamic_sources['procedures']['StudyID'] = InputDataType.CATEGORICAL

#     # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
#     dynamic_sources['diagnoses']['Date'] = InputDataType.TIMESTAMP
#     dynamic_sources['procedures']['Date'] = InputDataType.TIMESTAMP

#     if use_labs:
#         # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
#         dynamic_sources['labs']['StudyID'] = InputDataType.CATEGORICAL
#         #dynamic_sources['labs']['EMPI'] = InputDataType.CATEGORICAL



#         # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
#         dynamic_sources['labs']['Date'] = InputDataType.TIMESTAMP


#         # Add the 'Result' column to the 'col_schema' dictionary for the 'labs' input schema
#         dynamic_sources['labs']['Result'] = InputDataType.FLOAT

#         # Add the 'Code' column to the 'col_schema' dictionary for the 'labs' input schema
#         dynamic_sources['labs']['Code'] = InputDataType.CATEGORICAL

#         dynamic_sources['labs']['Source'] = InputDataType.CATEGORICAL

#         dynamic_sources['labs']['IndexDate'] = InputDataType.TIMESTAMP


        


#     # Build DatasetSchema
#     dynamic_input_schemas = []
#     events_dfs = []

#     if df_dia.is_empty():
#         raise ValueError("Diagnoses DataFrame is empty.")
#     else:
#         dynamic_input_schemas.append(
#             InputDFSchema(
#                 input_df=df_dia, #changed subject_id to study_id
#                 subject_id_col=None,  # Set to None for dynamic input schemas
#                 data_schema={
#                     'Date': InputDataType.TIMESTAMP
#                 },
#                 event_type='DIAGNOSIS',
#                 ts_col='Date',
#                 ts_format='%Y-%m-%d %H:%M:%S',
#                 type=InputDFType.EVENT
#             )
#         )
#         #StudyID
#         #deleting renaming of StudyID to subject_id
#         events_dfs.append(df_dia.select('StudyID', 'Date'))

#     #no rename
#     if not df_dia.is_empty():
#         events_dfs.append(df_dia.select('StudyID', 'Date'))

#     if not df_prc.is_empty():
#         events_dfs.append(df_prc.select('StudyID', 'Date'))
    
#     dynamic_input_schemas.extend([
#         InputDFSchema(
#             input_df=df_dia,
#             subject_id_col=None,
#             data_schema={'CodeWithType': InputDataType.CATEGORICAL, 'Date': InputDataType.TIMESTAMP},
#             event_type='DIAGNOSIS',
#             ts_col='Date',
#             ts_format='%Y-%m-%d %H:%M:%S',
#             type=InputDFType.EVENT
#         ),
#         InputDFSchema(
#             input_df=df_prc,
#             subject_id_col=None,
#             data_schema={'CodeWithType': InputDataType.CATEGORICAL, 'Date': InputDataType.TIMESTAMP},
#             event_type='PROCEDURE',
#             ts_col='Date',
#             ts_format='%Y-%m-%d %H:%M:%S',
#             type=InputDFType.EVENT
#         )
#     ])

#     #changed timestamp to Date
#     if use_labs:
#         dynamic_input_schemas.append(
#             InputDFSchema(
#                 input_df=df_labs,
#                 subject_id_col=None,
#                 data_schema={
#                     'StudyID': ('StudyID', InputDataType.CATEGORICAL),
#                     'Date': ('Date', InputDataType.TIMESTAMP),
#                     'Code': ('Code', InputDataType.CATEGORICAL),
#                     'Result': ('Result', InputDataType.CATEGORICAL),
#                     'Source': ('CodeWithType', InputDataType.CATEGORICAL),
#                     'IndexDate': ('IndexDate', InputDataType.TIMESTAMP)
#                 },
#                 event_type='LAB',
#                 ts_col='Date',
#                 ts_format='%Y-%m-%d %H:%M:%S',
#                 type=InputDFType.EVENT
#             )
#         )

#     # Collect the 'EMPI' column from the DataFrames
#     #changed to StudyID
#     df_dia_studyid = df_dia.lazy().select('StudyID').collect()['StudyID']
#     df_prc_studyid = df_prc.lazy().select('StudyID').collect()['StudyID']
#     if use_labs:
#         df_labs_studyid = df_labs.lazy().select('StudyID').collect()['StudyID']

#     # Process events and measurements data in a streaming fashion
#     events_df = pl.concat(events_dfs, how='diagonal')
#     events_df = events_df.with_columns(
#         #changed subject_id to StudyID
#         pl.col('StudyID').cast(pl.Utf8), #changed Uint32 to Utf8
#         pl.col('Date').map_elements(lambda s: pl.datetime(s, fmt='%Y-%m-%d %H:%M:%S') if isinstance(s, str) else s, return_dtype=pl.Datetime).alias('Date'),
#         #changed EMPI to StudyID
#         pl.when(pl.col('StudyID').is_in(df_dia_studyid) & pl.col('StudyID').is_in(df_prc_studyid))
#         .then(pl.lit('DIAGNOSIS').cast(pl.Categorical))
#         # .when(pl.col('subject_id').is_in(df_prc_empi))
#         # .then(pl.lit('PROCEDURE').cast(pl.Categorical))
#         .otherwise(pl.lit('PROCEDURE').cast(pl.Categorical))
#         .alias('event_type')
#     )

#     # Print the unique values and null count of the 'StudyID' column
#     #changed subject_id to StudyID
#     print("Unique values in StudyID column of events_df:", events_df['StudyID'].unique())
#     print("Number of null values in StudyID column of events_df:", events_df['StudyID'].null_count())
    
#     # Update the event_types_idxmap
#     event_types_idxmap = {
#         'DIAGNOSIS': 1,
#         'PROCEDURE': 2,
#         'LAB': 3
#     }

#     #changed dynamic_indices to CodeWithType
#     vocab_sizes_by_measurement = {
#         'event_type': len(event_types_idxmap) + 1,  # Add 1 for the unknown token
#         'CodeWithType': max(len(df_dia['CodeWithType'].unique()), len(df_prc['CodeWithType'].unique())) + 1,
#     }

#     vocab_offsets_by_measurement = {
#         'event_type': 0,
#         'CodeWithType': len(event_types_idxmap) + 1,
#     }

#     # Add the 'event_id' column to the 'events_df' DataFrame
#     events_df = events_df.with_row_index(name='event_id')

#     #changed subject_id to StudyID
#     df_outcomes = df_outcomes.with_columns(pl.col('StudyID').cast(pl.Utf8).alias('StudyID')) #changed Uint32 to Utf8
#     #changed to StudyID

#     # Print the unique values and null count of the 'StudyID' column
#     print("Unique values in StudyID column of df_outcomes:", df_outcomes['StudyID'].unique())
#     print("Number of null values in StudyID column of df_outcomes:", df_outcomes['StudyID'].null_count())

#     print("Shape of df_outcomes:", df_outcomes.shape)
#     print("Columns of df_outcomes:", df_outcomes.columns)

#     # Add the 'event_id' column to the 'dynamic_measurements_df' DataFrame
#     #changed EMPI to StudyID
#     if use_labs:
#         dynamic_measurements_df = df_labs.select('StudyID', 'Date', 'Code', 'Result', 'Source', 'IndexDate') #timestamp? changed to Date
#         #changed subject_id to StudyID
#         dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('StudyID').cast(pl.Utf8)) #changed Uint32 to Utf8
#         dynamic_measurements_df = (
#             dynamic_measurements_df
#             .group_by(['StudyID', 'Date'])
#             .agg(pl.col('Result').count().alias('tmp_event_id'))
#             .drop('tmp_event_id')
#             .with_row_index('tmp_event_id')
#             .rename({'tmp_event_id': 'event_id'})
#         )

#     if not use_labs:
#         #changed EMPI to StudyID, added conditional to deal with presence of required columns
#         dynamic_measurements_df = pl.concat([
#             df_dia.select('StudyID', 'Date', 'CodeWithType') if 'CodeWithType' in df_dia.columns else df_dia.select('StudyID', 'Date'),
#             df_prc.select('StudyID', 'Date', 'CodeWithType') if 'CodeWithType' in df_prc.columns else df_prc.select('StudyID', 'Date')
#         ], how='diagonal')
#         #changed EMPI to StudyID
#         #no rename: CodeWithType and Date
#         #dynamic_measurements_df = dynamic_measurements_df.rename({'Date': 'timestamp', 'CodeWithType': 'dynamic_indices'})
#         dynamic_measurements_df = dynamic_measurements_df.with_columns(
#             pl.col('StudyID').cast(pl.Utf8), #changed Uint32 to Utf8
#             pl.col('Date').cast(pl.Datetime),
#             pl.col('CodeWithType').cast(pl.Categorical)
#         )
#         dynamic_measurements_df = (
#             dynamic_measurements_df
#             .group_by(['StudyID', 'Date'])
#             .agg(pl.len().alias('tmp_event_id'))
#             .join(
#                 #changed subject_id to StudyID
#                 dynamic_measurements_df.select('StudyID', 'Date', 'CodeWithType'), 
#                 on=['StudyID', 'Date'],
#                 how='left'
#             )
#             .drop('tmp_event_id')
#             .with_row_index('tmp_event_id')
#             .rename({'tmp_event_id': 'event_id'})
#         )


#     print("Columns of dynamic_measurements_df:", dynamic_measurements_df.columns)
#     interval_dynamic_measurements_df = dynamic_measurements_df  # Initialize the variable

#     print("Data types of dynamic_measurements_df columns:")
#     for col in dynamic_measurements_df.columns:
#         print(f"{col}: {dynamic_measurements_df[col].dtype}")

#     #dynanic_indices to CodeWithType, timestamp to Date

#     #commenting out the following code to deal with presence of required columns
# #     dynamic_measurements_df = dynamic_measurements_df.with_columns(
# #     pl.col('StudyID').cast(pl.Utf8), #changed Uint32 to Utf8
# #     pl.col('Date').cast(pl.Datetime),
# #     pl.col('CodeWithType').cast(pl.Categorical)
# # )
#     required_columns = ['StudyID', 'Date']
#     if not use_labs:
#         required_columns.append('CodeWithType')
#     missing_columns = [col for col in required_columns if col not in dynamic_measurements_df.columns]
#     if missing_columns:
#         raise ValueError(f"Missing columns in dynamic_measurements_df: {missing_columns}")

#     dynamic_measurements_df = dynamic_measurements_df.with_columns(
#         pl.col('StudyID').cast(pl.Utf8),
#         pl.col('Date').cast(pl.Datetime)
#     )

#     if not use_labs:
#         dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('CodeWithType').cast(pl.Categorical))
    

#     # Print to ensure correct handling of join
#     print("Shape of dynamic_measurements_df before join:", dynamic_measurements_df.shape)
#     print("Sample rows of dynamic_measurements_df before join:")
#     print(dynamic_measurements_df.head(5))

#     print("Shape of events_df used for join:", events_df.select('Date', 'event_id').shape)
#     print("Sample rows of events_df used for join:")
#     print(events_df.select('Date', 'event_id').head(5))

#     print("Shape of dynamic_measurements_df after join:", dynamic_measurements_df.shape)
#     print("Sample rows of dynamic_measurements_df after join:")
#     print(dynamic_measurements_df.head(5))

#     print("Columns of events_df:", events_df.columns)
#     print("Unique values in event_type:", events_df['event_type'].unique())

#     print("Columns of dynamic_measurements_df:", dynamic_measurements_df.columns)
#     if not use_labs:
#         print("Unique values in CodeWithType:", dynamic_measurements_df['CodeWithType'].unique())

#     # Print the unique values and null count of the 'StudyID' column
#     print("Unique values in StudyID column of dynamic_measurements_df:", dynamic_measurements_df['StudyID'].unique())
#     print("Number of null values in StudyID column of dynamic_measurements_df:", dynamic_measurements_df['StudyID'].null_count())

#     print("Shape of dynamic_measurements_df before join:", dynamic_measurements_df.shape)
#     print("Sample rows of dynamic_measurements_df before join:")
#     print(dynamic_measurements_df.head(5))

#     #changed timestamp to Date
#     print("Shape of events_df used for join:", events_df.select('Date', 'event_id').shape)
#     print("Sample rows of events_df used for join:")
#     print(events_df.select('Date', 'event_id').head(5))

#     print("Shape of dynamic_measurements_df after join:", dynamic_measurements_df.shape)
#     print("Sample rows of dynamic_measurements_df after join:")
#     print(dynamic_measurements_df.head(5))

#     print("Columns of events_df:", events_df.columns)
#     print("Unique values in event_type:", events_df['event_type'].unique())

#     print("Columns of dynamic_measurements_df:", dynamic_measurements_df.columns)
#     #changed dynamic_indices to CodeWithType
#     print("Unique values in CodeWithType:", dynamic_measurements_df['CodeWithType'].unique())

#     dataset_schema = DatasetSchema(
#         static=InputDFSchema(
#             input_df=df_outcomes,  # Use df_outcomes directly
#             subject_id_col='StudyID',
#             data_schema={
#                 'InitialA1c': InputDataType.FLOAT,
#                 'Female': InputDataType.CATEGORICAL,
#                 'Married': InputDataType.CATEGORICAL,
#                 'GovIns': InputDataType.CATEGORICAL,
#                 'English': InputDataType.CATEGORICAL,
#                 'AgeYears': InputDataType.CATEGORICAL,
#                 'SDI_score': InputDataType.FLOAT,
#                 'Veteran': InputDataType.CATEGORICAL
#             },
#             type=InputDFType.STATIC
#         ),
#         dynamic=dynamic_input_schemas
#     )

#     # Build Config
#     split = (0.7, 0.2, 0.1)
#     seed = 42
#     do_overwrite = True
#     DL_chunk_size = 20000

#     #changed dynamic_indices to CodeWithType
#     config = DatasetConfig(
#         measurement_configs={
#             'CodeWithType': MeasurementConfig(
#                 temporality=TemporalityType.DYNAMIC,
#                 modality=DataModality.MULTI_LABEL_CLASSIFICATION,
#             ),
#         },
#         normalizer_config={'cls': 'standard_scaler'},
#         #save_dir=Path("data")  # Modify this line -- should this be /home/sbanerjee/diabetes_pred/data?
#         save_dir=Path("/home/sbanerjee/diabetes_pred/data")
#     )
#     config.event_types_idxmap = event_types_idxmap  # Assign event_types_idxmap to the config object

#     if config.save_dir is not None:
#         dataset_schema_dict = dataset_schema.to_dict()
#         with open(config.save_dir / "input_schema.json", "w") as f:
#             json.dump(dataset_schema_dict, f, default=json_serial)

#     print("Process events data")

#     # Determine the start and end timestamps
#     #changed timestamp to Date
#     start_timestamp = events_df['Date'].min()
#     end_timestamp = events_df['Date'].max()

#     # Convert the start and end timestamps to datetime objects
#     start_timestamp = datetime.fromisoformat(str(start_timestamp))
#     end_timestamp = datetime.fromisoformat(str(end_timestamp))

#     # Generate time intervals (e.g., weekly intervals)
#     time_intervals = generate_time_intervals(start_timestamp.date(), end_timestamp.date(), 7)

#     # Process the data in time intervals
#     processed_events_df = pl.DataFrame()
#     processed_dynamic_measurements_df = pl.DataFrame()

#     for start_date, end_date in time_intervals:
#         print(f"Processing interval {start_date} - {end_date}")

#         # Filter events_df and dynamic_measurements_df for the current time interval
#         #changed timestamp to Date
#         interval_events_df = events_df.filter(
#             (pl.col('Date') >= pl.datetime(start_date.year, start_date.month, start_date.day)) &
#             (pl.col('Date') < pl.datetime(end_date.year, end_date.month, end_date.day))
#         )
#         interval_dynamic_measurements_df = dynamic_measurements_df.filter(
#             (pl.col('Date') >= pl.datetime(start_date.year, start_date.month, start_date.day)) &
#             (pl.col('Date') < pl.datetime(end_date.year, end_date.month, end_date.day))
#         )

#         # Append the processed intervals to the final DataFrames
#         processed_events_df = pl.concat([processed_events_df, interval_events_df])
#         processed_dynamic_measurements_df = pl.concat([processed_dynamic_measurements_df, interval_dynamic_measurements_df])

#     # Drop duplicate event_id values from the processed_events_df
#     processed_events_df = processed_events_df.unique(subset=['event_id'])

#     # Update the events_df and dynamic_measurements_df with the processed data
#     events_df = processed_events_df
#     dynamic_measurements_df = processed_dynamic_measurements_df

#     print(f"Final shape of events_df: {events_df.shape}")
#     print(f"Final shape of dynamic_measurements_df: {dynamic_measurements_df.shape}")

#     print("Creating Dataset object...")
#     ESD = Dataset(
#         config=config,
#         subjects_df=df_outcomes,  # Use df_outcomes directly
#         events_df=events_df,  # Use events_df directly
#         dynamic_measurements_df=dynamic_measurements_df
#     )

#     print("Dataset object created.")

#     print("Splitting dataset...")
#     ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
#     print("Dataset split.")

#     print("Preprocessing dataset...")
#     ESD.preprocess()
#     print("Dataset preprocessed.")

#     print("Add epsilon to concurrent timestamps")
#     eps = np.finfo(float).eps

#     # Handle null timestamp values by filling them with a default value
#     default_timestamp = pl.datetime(2000, 1, 1)  # Choose an appropriate default value
#     #changed timestamp to Date
#     ESD.events_df = ESD.events_df.with_columns(
#         pl.col('Date').fill_null(default_timestamp).alias('Date')
#     )
#     ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(
#         pl.col('Date').fill_null(default_timestamp).alias('Date')
#     )

#     # Add epsilon to concurrent timestamps
#     eps = np.finfo(float).eps
#     ESD.events_df = ESD.events_df.with_columns(
#         (pl.col('Date').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('Date')
#     )
#     ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(
#         (pl.col('Date').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('Date')
#     )

#     # Assign the vocabulary sizes and offsets to the Dataset's vocabulary_config
#     ESD.vocabulary_config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
#     ESD.vocabulary_config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

#     print("Saving dataset...")
#     ESD.save(do_overwrite=do_overwrite)
#     print("Dataset saved.")

#     # Print the contents of ESD
#     print("Contents of ESD after saving:")
#     print("Config:", ESD.config)
#     print("Subjects DF:")
#     print(ESD.subjects_df.head())
#     print("Events DF:")
#     print(ESD.events_df.head())
#     print("Dynamic Measurements DF:")
#     print(ESD.dynamic_measurements_df.head())
#     print("Inferred Measurement Configs:", ESD.inferred_measurement_configs)

#     print("Caching deep learning representation...")
#     ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)
#     print("Deep learning representation cached.")

#     # Read and display the contents of the Parquet files
#     print("Contents of Parquet files in DL_reps directory:")
#     dl_reps_dir = ESD.config.save_dir / "DL_reps"
#     if dl_reps_dir.exists():
#         for split in ["train", "tuning", "held_out"]:
#             parquet_files = list(dl_reps_dir.glob(f"{split}*.parquet"))
#             if parquet_files:
#                 print(f"Parquet files for split '{split}':")
#                 for parquet_file in parquet_files:
#                     print(f"File: {parquet_file}")
#                     try:
#                         df = read_parquet_file(parquet_file)
#                         print(df.head())
#                     except Exception as e:
#                         print(f"Error reading Parquet file: {parquet_file}")
#                         print(f"Error message: {str(e)}")
#             else:
#                 print(f"No Parquet files found for split '{split}'.")
#     else:
#         print("DL_reps directory not found.")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
#     args = parser.parse_args()
#     main(use_dask=args.use_dask)