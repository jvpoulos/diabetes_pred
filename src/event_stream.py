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
from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select
from collections import defaultdict
from EventStream.data.preprocessing.standard_scaler import StandardScaler

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import tempfile
import shutil
import pyarrow.parquet as pq

#sujay
<<<<<<< Updated upstream
def main(use_dask=False, use_labs=False):
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'
    if use_labs:
        labs_file_path = 'data/Labs.txt'
=======
def main(use_dask=False, use_labs=True):
    outcomes_file_path = '/home/sbanerjee/diabetes_pred/data/DiabetesOutcomes.txt'
    diagnoses_file_path = '/home/sbanerjee/diabetes_pred/data/Diagnoses_3yr.txt'
    procedures_file_path = '/home/sbanerjee/diabetes_pred/data/Procedures_3yr.txt'

    if use_labs:
        labs_file_path = '/home/sbanerjee/diabetes_pred/data/Labs_3yr.txt'
>>>>>>> Stashed changes

    df_outcomes = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_outcomes = df_outcomes if isinstance(df_outcomes, pl.DataFrame) else pl.from_pandas(df_outcomes)

    print("Outcomes DataFrame columns:", df_outcomes.columns)

    if df_outcomes.is_empty():
        raise ValueError("Outcomes DataFrame is empty.")

<<<<<<< Updated upstream
    df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select, min_frequency=ceil(38960*0.01))
=======
    df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select)# deleting min freq)
>>>>>>> Stashed changes
    df_dia = df_dia if isinstance(df_dia, pl.DataFrame) else pl.from_pandas(df_dia)

    if df_dia.is_empty():
        raise ValueError("Diagnoses DataFrame is empty.")

<<<<<<< Updated upstream
    df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select, min_frequency=ceil(38960*0.01))
=======
    df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select) # deleting min freq)
>>>>>>> Stashed changes
    df_prc = df_prc if isinstance(df_prc, pl.DataFrame) else pl.from_pandas(df_prc)

    if df_prc.is_empty():
        raise ValueError("Procedures DataFrame is empty.")

    if use_labs:
<<<<<<< Updated upstream
        df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, chunk_size=700000, min_frequency=ceil(38960*0.01))
=======
        df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, chunk_size=200000) # deleting min freq)
>>>>>>> Stashed changes
        df_labs = df_labs if isinstance(df_labs, pl.DataFrame) else pl.from_pandas(df_labs)

        if df_labs.is_empty():
            raise ValueError("Labs DataFrame is empty.")
        
    # Build measurement_configs and track input schemas
    #changed EMPI to StudyID
    subject_id_col = 'StudyID'
    measurements_by_temporality = {
        TemporalityType.STATIC: {
            'outcomes': {
                # DataModality.SINGLE_LABEL_CLASSIFICATION: ['A1cGreaterThan7'],
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

                    # Refactored code without match-case
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
    #changed EMPI to StudyID
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
    #changed EMPI to StudyID
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
                input_schema_kwargs["input_df"] = df_outcomes
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
    
    #changed EMPI to StudyID
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
            'columns': { #change from EMPI to StudyID
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'Date': ('Date', InputDataType.TIMESTAMP),
                'Code': ('Code', InputDataType.CATEGORICAL),
                'Code_Type': ('Code_Type', InputDataType.CATEGORICAL),
                'IndexDate': ('IndexDate', InputDataType.TIMESTAMP),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)

            }
        },
        'procedures': {
            'input_df': procedures_file_path,
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'Date': ('Date', InputDataType.TIMESTAMP),
                'Code': ('Code', InputDataType.CATEGORICAL),
                'Code_Type': ('Code_Type', InputDataType.CATEGORICAL),
                'IndexDate': ('IndexDate', InputDataType.TIMESTAMP),
                'CodeWithType': ('CodeWithType', InputDataType.CATEGORICAL)
            }
        }, #fix all these
        'labs': {
            'input_df': labs_file_path,
            'event_type': 'LAB',
            'ts_col': 'Date',
            'ts_format': '%Y-%m-%d %H:%M:%S',
            'columns': {
                'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                'Date': ('Date', InputDataType.TIMESTAMP),
                'Code': ('Code', InputDataType.CATEGORICAL),
                'Result': ('Result', InputDataType.CATEGORICAL),
                'Source': ('CodeWithType', InputDataType.CATEGORICAL),
                'IndexDate': ('IndexDate', InputDataType.TIMESTAMP)
            }
        } if use_labs else []
    }

    dynamic_sources['diagnoses']['dynamic_indices'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['dynamic_indices'] = InputDataType.CATEGORICAL

    # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
    #changed to StudyID
    dynamic_sources['diagnoses']['StudyID'] = InputDataType.CATEGORICAL
    dynamic_sources['procedures']['StudyID'] = InputDataType.CATEGORICAL

    # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
    dynamic_sources['diagnoses']['Date'] = InputDataType.TIMESTAMP
    dynamic_sources['procedures']['Date'] = InputDataType.TIMESTAMP

    if use_labs:
        # Add the 'EMPI' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['StudyID'] = InputDataType.CATEGORICAL
        #dynamic_sources['labs']['EMPI'] = InputDataType.CATEGORICAL



        # Add the 'Date' column to the 'col_schema' dictionary for all dynamic input schemas
        dynamic_sources['labs']['Date'] = InputDataType.TIMESTAMP


        # Add the 'Result' column to the 'col_schema' dictionary for the 'labs' input schema
        dynamic_sources['labs']['Result'] = InputDataType.FLOAT

        # Add the 'Code' column to the 'col_schema' dictionary for the 'labs' input schema
        dynamic_sources['labs']['Code'] = InputDataType.CATEGORICAL

        dynamic_sources['labs']['Source'] = InputDataType.CATEGORICAL

        dynamic_sources['labs']['IndexDate'] = InputDataType.TIMESTAMP


        


    # Build DatasetSchema
    dynamic_input_schemas = []
    events_dfs = []

    if df_dia.is_empty():
        raise ValueError("Diagnoses DataFrame is empty.")
    else:
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
<<<<<<< Updated upstream
        events_dfs.append(df_dia.select('EMPI', 'Date').rename({'EMPI': 'subject_id'}))

    if not df_dia.is_empty():
        events_dfs.append(df_dia.select('EMPI', 'Date').rename({'EMPI': 'subject_id', 'Date': 'timestamp'}))

    if not df_prc.is_empty():
        events_dfs.append(df_prc.select('EMPI', 'Date').rename({'EMPI': 'subject_id', 'Date': 'timestamp'}))
=======
        #StudyID
        events_dfs.append(df_dia.select('StudyID', 'Date').rename({'StudyID': 'subject_id'}))

    if not df_dia.is_empty():
        events_dfs.append(df_dia.select('StudyID', 'Date').rename({'StudyID': 'subject_id', 'Date': 'timestamp'}))

    if not df_prc.is_empty():
        events_dfs.append(df_prc.select('StudyID', 'Date').rename({'StudyID': 'subject_id', 'Date': 'timestamp'}))
>>>>>>> Stashed changes
    
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
                    'StudyID': ('StudyID', InputDataType.CATEGORICAL),
                    'Date': ('Date', InputDataType.TIMESTAMP),
                    'Code': ('Code', InputDataType.CATEGORICAL),
                    'Result': ('Result', InputDataType.CATEGORICAL),
                    'Source': ('CodeWithType', InputDataType.CATEGORICAL),
                    'IndexDate': ('IndexDate', InputDataType.TIMESTAMP)
                },
                event_type='LAB',
                ts_col='timestamp',
                ts_format='%Y-%m-%d %H:%M:%S',
                type=InputDFType.EVENT
            )
        )

    # Collect the 'EMPI' column from the DataFrames
<<<<<<< Updated upstream
    df_dia_empi = df_dia.lazy().select('EMPI').collect()['EMPI']
    df_prc_empi = df_prc.lazy().select('EMPI').collect()['EMPI']
    if use_labs:
        df_labs_empi = df_labs.lazy().select('EMPI').collect()['EMPI']
=======
    #changed to StudyID
    df_dia_studyid = df_dia.lazy().select('StudyID').collect()['StudyID']
    df_prc_studyid = df_prc.lazy().select('StudyID').collect()['StudyID']
    if use_labs:
        df_labs_studyid = df_labs.lazy().select('StudyID').collect()['StudyID']
>>>>>>> Stashed changes

    # Process events and measurements data in a streaming fashion
    events_df = pl.concat(events_dfs, how='diagonal')
    events_df = events_df.with_columns(
<<<<<<< Updated upstream
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('Date').map_elements(lambda s: pl.datetime(s, fmt='%Y-%m-%d %H:%M:%S') if isinstance(s, str) else s, return_dtype=pl.Datetime).alias('timestamp'),
        pl.when(pl.col('subject_id').is_in(df_dia_empi) & pl.col('subject_id').is_in(df_prc_empi))
=======
        pl.col('subject_id').cast(pl.Utf8), #changed Uint32 to Utf8
        pl.col('Date').map_elements(lambda s: pl.datetime(s, fmt='%Y-%m-%d %H:%M:%S') if isinstance(s, str) else s, return_dtype=pl.Datetime).alias('timestamp'),
        #changed EMPI to StudyID
        pl.when(pl.col('subject_id').is_in(df_dia_studyid) & pl.col('subject_id').is_in(df_prc_studyid))
>>>>>>> Stashed changes
        .then(pl.lit('DIAGNOSIS').cast(pl.Categorical))
        # .when(pl.col('subject_id').is_in(df_prc_empi))
        # .then(pl.lit('PROCEDURE').cast(pl.Categorical))
        .otherwise(pl.lit('PROCEDURE').cast(pl.Categorical))
        .alias('event_type')
    )

    # Print the unique values and null count of the 'subject_id' column
    print("Unique values in subject_id column of events_df:", events_df['subject_id'].unique())
    print("Number of null values in subject_id column of events_df:", events_df['subject_id'].null_count())
<<<<<<< Updated upstream

    # Update the event_types_idxmap
    event_types_idxmap = {
        'DIAGNOSIS': 1,
        'PROCEDURE': 2
=======
    
    # Update the event_types_idxmap
    event_types_idxmap = {
        'DIAGNOSIS': 1,
        'PROCEDURE': 2,
        'LAB': 3
>>>>>>> Stashed changes
    }

    vocab_sizes_by_measurement = {
        'event_type': len(event_types_idxmap) + 1,  # Add 1 for the unknown token
        'dynamic_indices': max(len(df_dia['CodeWithType'].unique()), len(df_prc['CodeWithType'].unique())) + 1,
    }

    vocab_offsets_by_measurement = {
        'event_type': 0,
        'dynamic_indices': len(event_types_idxmap) + 1,
    }

    # Add the 'event_id' column to the 'events_df' DataFrame
    events_df = events_df.with_row_index(name='event_id')

<<<<<<< Updated upstream
    df_outcomes = df_outcomes.with_columns(pl.col('EMPI').cast(pl.UInt32).alias('subject_id'))
=======
    df_outcomes = df_outcomes.with_columns(pl.col('StudyID').cast(pl.Utf8).alias('subject_id')) #changed Uint32 to Utf8
    #changed to StudyID
>>>>>>> Stashed changes

    # Print the unique values and null count of the 'subject_id' column
    print("Unique values in subject_id column of df_outcomes:", df_outcomes['subject_id'].unique())
    print("Number of null values in subject_id column of df_outcomes:", df_outcomes['subject_id'].null_count())

    print("Shape of df_outcomes:", df_outcomes.shape)
    print("Columns of df_outcomes:", df_outcomes.columns)

    # Add the 'event_id' column to the 'dynamic_measurements_df' DataFrame
<<<<<<< Updated upstream
=======
    #changed EMPI to StudyID
>>>>>>> Stashed changes
    if use_labs:
        dynamic_measurements_df = df_labs.select('StudyID', 'Date', 'Code', 'Result', 'Source', 'IndexDate').rename({'StudyID': 'subject_id', 'Date': 'timestamp'})
        dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col('subject_id').cast(pl.Utf8)) #changed Uint32 to Utf8
        dynamic_measurements_df = (
            dynamic_measurements_df
            .group_by(['subject_id', 'timestamp'])
            .agg(pl.col('Result').count().alias('tmp_event_id'))
            .drop('tmp_event_id')
            .with_row_index('tmp_event_id')
            .rename({'tmp_event_id': 'event_id'})
        )

    if not use_labs:
<<<<<<< Updated upstream
        dynamic_measurements_df = pl.concat([
            df_dia.select('EMPI', 'CodeWithType', 'Date'),
            df_prc.select('EMPI', 'CodeWithType', 'Date')
        ], how='diagonal')
        dynamic_measurements_df = dynamic_measurements_df.rename({'EMPI': 'subject_id', 'Date': 'timestamp', 'CodeWithType': 'dynamic_indices'})
        dynamic_measurements_df = dynamic_measurements_df.with_columns(
            pl.col('subject_id').cast(pl.UInt32),
            pl.col('timestamp').cast(pl.Datetime),
     #       pl.col('dynamic_indices').cast(pl.Categorical)
=======
        #changed EMPI to StudyID
        dynamic_measurements_df = pl.concat([
            df_dia.select('StudyID', 'CodeWithType', 'Date'),
            df_prc.select('StudyID', 'CodeWithType', 'Date')
        ], how='diagonal')
        #changed EMPI to StudyID
        dynamic_measurements_df = dynamic_measurements_df.rename({'StudyID': 'subject_id', 'Date': 'timestamp', 'CodeWithType': 'dynamic_indices'})
        dynamic_measurements_df = dynamic_measurements_df.with_columns(
            pl.col('subject_id').cast(pl.Utf8), #changed Uint32 to Utf8
            pl.col('timestamp').cast(pl.Datetime),
            pl.col('dynamic_indices').cast(pl.Categorical)
>>>>>>> Stashed changes
        )
        dynamic_measurements_df = (
            dynamic_measurements_df
            .group_by(['subject_id', 'timestamp'])
            .agg(pl.len().alias('tmp_event_id'))
            .join(
                dynamic_measurements_df.select('subject_id', 'timestamp', 'dynamic_indices'), 
                on=['subject_id', 'timestamp'],
                how='left'
            )
            .drop('tmp_event_id')
            .with_row_index('tmp_event_id')
            .rename({'tmp_event_id': 'event_id'})
        )


    print("Columns of dynamic_measurements_df:", dynamic_measurements_df.columns)
    interval_dynamic_measurements_df = dynamic_measurements_df  # Initialize the variable

    print("Data types of dynamic_measurements_df columns:")
    for col in dynamic_measurements_df.columns:
        print(f"{col}: {dynamic_measurements_df[col].dtype}")

    dynamic_measurements_df = dynamic_measurements_df.with_columns(
<<<<<<< Updated upstream
    pl.col('subject_id').cast(pl.UInt32),
=======
    pl.col('subject_id').cast(pl.Utf8), #changed Uint32 to Utf8
>>>>>>> Stashed changes
    pl.col('timestamp').cast(pl.Datetime),
    pl.col('dynamic_indices').cast(pl.Categorical)
)

    # Print the unique values and null count of the 'subject_id' column
    print("Unique values in subject_id column of dynamic_measurements_df:", dynamic_measurements_df['subject_id'].unique())
    print("Number of null values in subject_id column of dynamic_measurements_df:", dynamic_measurements_df['subject_id'].null_count())

    print("Shape of dynamic_measurements_df before join:", dynamic_measurements_df.shape)
    print("Sample rows of dynamic_measurements_df before join:")
    print(dynamic_measurements_df.head(5))

    print("Shape of events_df used for join:", events_df.select('timestamp', 'event_id').shape)
    print("Sample rows of events_df used for join:")
    print(events_df.select('timestamp', 'event_id').head(5))

    print("Shape of dynamic_measurements_df after join:", dynamic_measurements_df.shape)
    print("Sample rows of dynamic_measurements_df after join:")
    print(dynamic_measurements_df.head(5))

    print("Columns of events_df:", events_df.columns)
    print("Unique values in event_type:", events_df['event_type'].unique())

    print("Columns of dynamic_measurements_df:", dynamic_measurements_df.columns)
    print("Unique values in dynamic_indices:", dynamic_measurements_df['dynamic_indices'].unique())

    dataset_schema = DatasetSchema(
        static=InputDFSchema(
            input_df=df_outcomes,  # Use df_outcomes directly
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
<<<<<<< Updated upstream
        save_dir=Path("data")  # Modify this line
=======
        #save_dir=Path("data")  # Modify this line -- should this be /home/sbanerjee/diabetes_pred/data?
        save_dir=Path("/home/sbanerjee/diabetes_pred/data")
>>>>>>> Stashed changes
    )
    config.event_types_idxmap = event_types_idxmap  # Assign event_types_idxmap to the config object

    if config.save_dir is not None:
        dataset_schema_dict = dataset_schema.to_dict()
        with open(config.save_dir / "input_schema.json", "w") as f:
            json.dump(dataset_schema_dict, f, default=json_serial)

    print("Process events data")

    # Determine the start and end timestamps
    start_timestamp = events_df['timestamp'].min()
    end_timestamp = events_df['timestamp'].max()

    # Convert the start and end timestamps to datetime objects
    start_timestamp = datetime.fromisoformat(str(start_timestamp))
    end_timestamp = datetime.fromisoformat(str(end_timestamp))

    # Generate time intervals (e.g., weekly intervals)
    time_intervals = generate_time_intervals(start_timestamp.date(), end_timestamp.date(), 7)

    # Process the data in time intervals
    processed_events_df = pl.DataFrame()
    processed_dynamic_measurements_df = pl.DataFrame()

    for start_date, end_date in time_intervals:
        print(f"Processing interval {start_date} - {end_date}")

        # Filter events_df and dynamic_measurements_df for the current time interval
        interval_events_df = events_df.filter(
            (pl.col('timestamp') >= pl.datetime(start_date.year, start_date.month, start_date.day)) &
            (pl.col('timestamp') < pl.datetime(end_date.year, end_date.month, end_date.day))
        )
        interval_dynamic_measurements_df = dynamic_measurements_df.filter(
            (pl.col('timestamp') >= pl.datetime(start_date.year, start_date.month, start_date.day)) &
            (pl.col('timestamp') < pl.datetime(end_date.year, end_date.month, end_date.day))
        )

        # Append the processed intervals to the final DataFrames
        processed_events_df = pl.concat([processed_events_df, interval_events_df])
        processed_dynamic_measurements_df = pl.concat([processed_dynamic_measurements_df, interval_dynamic_measurements_df])
<<<<<<< Updated upstream

    # Drop duplicate event_id values from the processed_events_df
    processed_events_df = processed_events_df.unique(subset=['event_id'])

    # Update the events_df and dynamic_measurements_df with the processed data
    events_df = processed_events_df
    dynamic_measurements_df = processed_dynamic_measurements_df

=======

    # Drop duplicate event_id values from the processed_events_df
    processed_events_df = processed_events_df.unique(subset=['event_id'])

    # Update the events_df and dynamic_measurements_df with the processed data
    events_df = processed_events_df
    dynamic_measurements_df = processed_dynamic_measurements_df

>>>>>>> Stashed changes
    print(f"Final shape of events_df: {events_df.shape}")
    print(f"Final shape of dynamic_measurements_df: {dynamic_measurements_df.shape}")

    print("Creating Dataset object...")
    ESD = Dataset(
        config=config,
        subjects_df=df_outcomes,  # Use df_outcomes directly
        events_df=events_df,  # Use events_df directly
        dynamic_measurements_df=dynamic_measurements_df
    )

    print("Dataset object created.")

    print("Splitting dataset...")
    ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
    print("Dataset split.")

    print("Preprocessing dataset...")
    ESD.preprocess()
    print("Dataset preprocessed.")

    print("Add epsilon to concurrent timestamps")
    eps = np.finfo(float).eps

    # Handle null timestamp values by filling them with a default value
    default_timestamp = pl.datetime(2000, 1, 1)  # Choose an appropriate default value
    ESD.events_df = ESD.events_df.with_columns(
        pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
    )
    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(
        pl.col('timestamp').fill_null(default_timestamp).alias('timestamp')
    )

    # Add epsilon to concurrent timestamps
    eps = np.finfo(float).eps
    ESD.events_df = ESD.events_df.with_columns(
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp')
    )
    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(
        (pl.col('timestamp').cast(pl.Float64) + pl.col('event_id') * eps).cast(pl.Datetime).alias('timestamp')
    )

    # Assign the vocabulary sizes and offsets to the Dataset's vocabulary_config
    ESD.vocabulary_config.vocab_sizes_by_measurement = vocab_sizes_by_measurement
    ESD.vocabulary_config.vocab_offsets_by_measurement = vocab_offsets_by_measurement

    print("Saving dataset...")
    ESD.save(do_overwrite=do_overwrite)
    print("Dataset saved.")

    # Print the contents of ESD
    print("Contents of ESD after saving:")
    print("Config:", ESD.config)
    print("Subjects DF:")
    print(ESD.subjects_df.head())
    print("Events DF:")
    print(ESD.events_df.head())
    print("Dynamic Measurements DF:")
    print(ESD.dynamic_measurements_df.head())
    print("Inferred Measurement Configs:", ESD.inferred_measurement_configs)

    print("Caching deep learning representation...")
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)
    print("Deep learning representation cached.")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
    args = parser.parse_args()
    main(use_dask=args.use_dask)