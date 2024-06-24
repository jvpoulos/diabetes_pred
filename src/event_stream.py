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
from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals, create_code_mapping, map_codes_to_indices, create_inverse_mapping
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

def main(use_labs=False, save_schema=False):
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'
    if use_labs:
        labs_file_path = 'data/Labs.txt'

    subjects_df = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select, chunk_size=10000)

    # Aggregate subjects_df to have one row per EMPI
    subjects_df = subjects_df.groupby('EMPI').agg([
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

    # Add the subject_id column
    subjects_df = subjects_df.with_columns(pl.col('EMPI').cast(pl.UInt32).alias('subject_id'))

    print("Shape of subjects_df after aggregation:", subjects_df.shape)
    print("Number of unique EMPIs:", subjects_df['EMPI'].n_unique())

    print("Processing diagnosis and procedure data...")
    df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select, min_frequency=ceil(subjects_df.height*0.01), chunk_size=90000)
    df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select, min_frequency=ceil(subjects_df.height*0.01), chunk_size=90000)

    print("Creating code mapping...")
    code_mapping = create_code_mapping(df_dia, df_prc)
    print(f"Total unique codes: {len(code_mapping)}")

    print("Creating inverse_mapping...")
    inverse_mapping = {idx: code for code, idx in code_mapping.items()}
    print(f"Total unique codes in inverse_mapping: {len(inverse_mapping)}")

    print("Mapping codes to indices...")
    df_dia = map_codes_to_indices(df_dia, code_mapping)
    df_prc = map_codes_to_indices(df_prc, code_mapping)
    
    if use_labs:
        df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, chunk_size=90000, min_frequency=ceil(subjects_df.height*0.01))

    # Build measurement_configs and track input schemas
    subject_id_col = 'EMPI'
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
    static_sources['outcomes']['EMPI'] = InputDataType.CATEGORICAL
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

    static_sources['procedures']['EMPI'] = InputDataType.CATEGORICAL
    static_sources['diagnoses']['EMPI'] = InputDataType.CATEGORICAL

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
                'EMPI': ('EMPI', InputDataType.CATEGORICAL),
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
                'EMPI': ('EMPI', InputDataType.CATEGORICAL),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
            }
        },
        'procedures': {
            'input_df': procedures_file_path,
            'columns': {
                'EMPI': ('EMPI', InputDataType.CATEGORICAL),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
            }
        },
        'labs': {
            'input_df': labs_file_path,
            'event_type': 'LAB',
            'ts_col': 'Date',
            'ts_format': '%Y-%m-%d %H:%M:%S',
            'columns': {
                'EMPI': ('EMPI', InputDataType.CATEGORICAL),
                'Code': ('Code', InputDataType.CATEGORICAL),
                'Result': ('Result', InputDataType.FLOAT),
                'Date': ('Date', InputDataType.TIMESTAMP)
            }
        } if use_labs else []
    }

    dynamic_sources['diagnoses']['dynamic_indices'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['dynamic_indices'] = InputDataType.CATEGORICAL

    # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
    dynamic_sources['diagnoses']['EMPI'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['EMPI'] = InputDataType.CATEGORICAL

    # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
    dynamic_sources['diagnoses']['Date'] = InputDataType.TIMESTAMP
    dynamic_sources['procedures']['Date'] = InputDataType.TIMESTAMP

    if use_labs:
        # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['EMPI'] = InputDataType.CATEGORICAL

        # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['Date'] = InputDataType.TIMESTAMP

        # Add the 'Result' column to the 'col_schema' dictionary for the 'labs' input schema
        dynamic_sources['labs']['Result'] = InputDataType.FLOAT

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
                    'Result': InputDataType.FLOAT,
                    'timestamp': InputDataType.TIMESTAMP
                },
                event_type='LAB',
                ts_col='timestamp',
                ts_format='%Y-%m-%d %H:%M:%S',
                type=InputDFType.EVENT
            )
        )
    
    if use_labs:
        df_labs_empi = df_labs['EMPI'].unique()

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

    print("Processing events and measurements data...")
    temp_dataset = Dataset(config)

    events_df_dia, dynamic_measurements_df_dia = temp_dataset._process_events_and_measurements_df(
        df_dia, "DIAGNOSIS", {"CodeWithType": ("CodeWithType", InputDataType.CATEGORICAL), "dynamic_indices": ("dynamic_indices", InputDataType.CATEGORICAL)}
    )
    events_df_prc, dynamic_measurements_df_prc = temp_dataset._process_events_and_measurements_df(
        df_prc, "PROCEDURE", {"CodeWithType": ("CodeWithType", InputDataType.CATEGORICAL), "dynamic_indices": ("dynamic_indices", InputDataType.CATEGORICAL)}
    )

    events_df = pl.concat([df_dia, df_prc], how='diagonal')
    events_df = events_df.with_columns([
        pl.col('EMPI').alias('subject_id').cast(pl.UInt32),
        pl.col('Date').alias('timestamp'),
        pl.when(pl.col('EMPI').is_in(df_dia['EMPI']))
        .then(pl.lit('DIAGNOSIS'))
        .otherwise(pl.lit('PROCEDURE'))
        .alias('event_type').cast(pl.Categorical)
    ])
    events_df = events_df.join(subjects_df.select(['EMPI', 'subject_id']), on='EMPI', how='inner')
    events_df = events_df.with_row_index('event_id')

    dynamic_measurements_df = pl.concat([dynamic_measurements_df_dia, dynamic_measurements_df_prc], how="diagonal")

    print("Columns in dynamic_measurements_df:", dynamic_measurements_df.columns)

    dynamic_measurements_df = pl.concat([
        df_dia.select('EMPI', 'CodeWithType', 'Date'),
        df_prc.select('EMPI', 'CodeWithType', 'Date')
    ])

    dynamic_measurements_df = dynamic_measurements_df.with_columns([
        pl.col('EMPI').alias('subject_id').cast(pl.UInt32),
        pl.col('Date').alias('timestamp'),
        pl.col('CodeWithType').alias('dynamic_indices'),  # Keep as string
        pl.lit(1).cast(pl.Int32).alias('dynamic_counts'),
        pl.when(pl.col('EMPI').is_in(df_dia['EMPI']))
        .then(pl.lit('DIAGNOSIS'))
        .otherwise(pl.lit('PROCEDURE'))
        .alias('dynamic_indices_event_type').cast(pl.Categorical),
        pl.lit(1).cast(pl.Int32).alias('dynamic_counts_event_type')
    ])

    dynamic_measurements_df = dynamic_measurements_df.join(subjects_df.select(['EMPI', 'subject_id']), on='EMPI', how='inner')

    dynamic_measurements_df = dynamic_measurements_df.group_by([
        'subject_id', 'timestamp', 'dynamic_indices', 'dynamic_indices_event_type'
    ]).agg([
        pl.col('dynamic_counts').sum(),
        pl.col('dynamic_counts_event_type').sum()
    ])

    dynamic_measurements_df = dynamic_measurements_df.with_row_index('event_id')

    subjects_df = subjects_df.with_columns(pl.col('EMPI').cast(pl.UInt32).alias('subject_id'))

    print("Shape of subjects_df:", subjects_df.shape)
    print("Columns of subjects_df:", subjects_df.columns)

    if use_labs:
        dynamic_measurements_df = df_labs.select('EMPI', 'Code', 'Result', 'Date').rename({'EMPI': 'subject_id', 'Date': 'timestamp'})
        dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('subject_id').cast(pl.UInt32))
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

    config.event_types_idxmap = event_types_idxmap  # Assign event_types_idxmap to the config object

    print("Event types index map:")
    print(event_types_idxmap)

    # Update the vocab_sizes_by_measurement and vocab_offsets_by_measurement
    vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap),
        'dynamic_indices': len(code_mapping),
    }

    vocab_offsets_by_measurement = {
        'event_type': 1,
        'dynamic_indices': len(event_types_idxmap),
    }

    print("Updated Vocabulary sizes by measurement:", vocab_sizes_by_measurement)
    print("Updated Vocabulary offsets by measurement:", vocab_offsets_by_measurement)

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

    ESD = Dataset(
        config=config,
        subjects_df=subjects_df,
        events_df=events_df,
        dynamic_measurements_df=dynamic_measurements_df,
        code_mapping=code_mapping
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
            pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
        ])
    else:
        ESD.events_df = ESD.events_df.with_columns([
            pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
        ])

    ESD.events_df = ESD.events_df.with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
    ])

    if 'timestamp' not in ESD.dynamic_measurements_df.columns:
        ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
            pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
        ])
    else:
        ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
            pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
        ])

    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns([
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp_with_epsilon')
    ])

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

    print("Checking dynamic_indices in ESD.dynamic_measurements_df")
    null_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices').is_null()).shape[0]
    unknown_count = ESD.dynamic_measurements_df.filter(pl.col('dynamic_indices') == 'UNKNOWN_CODE').shape[0]
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
    args = parser.parse_args()

    main(use_labs=args.use_labs, save_schema=args.save_schema)