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
from data_utils import read_file, preprocess_dataframe
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select
from collections import defaultdict

from datetime import datetime

def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Type {type(obj)} not serializable')

def add_to_container(key: str, val: Any, cont: dict[str, Any]):
    if key in cont:
        if cont[key] == val:
            print(f"WARNING: {key} is specified twice with value {val}.")
        else:
            raise ValueError(f"{key} is specified twice ({val} v. {cont[key]})")
    else:
        cont[key] = val

def main(use_dask=False, use_labs=False):
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'
    if use_labs:
        labs_file_path = 'data/Labs.txt'

    df_outcomes = preprocess_dataframe('Outcomes', outcomes_file_path, outcomes_columns, outcomes_columns_select, use_threshold=False)
    df_outcomes = df_outcomes if isinstance(df_outcomes, pl.DataFrame) else pl.from_pandas(df_outcomes)

    if df_outcomes.is_empty():
        raise ValueError("Outcomes DataFrame is empty.")

    df_dia = preprocess_dataframe('Diagnoses', diagnoses_file_path, dia_columns, dia_columns_select, use_threshold=False)
    df_dia = df_dia if isinstance(df_dia, pl.DataFrame) else pl.from_pandas(df_dia)

    if df_dia.is_empty():
        raise ValueError("Diagnoses DataFrame is empty.")

    df_prc = preprocess_dataframe('Procedures', procedures_file_path, prc_columns, prc_columns_select, use_threshold=False)
    df_prc = df_prc if isinstance(df_prc, pl.DataFrame) else pl.from_pandas(df_prc)

    if df_prc.is_empty():
        raise ValueError("Procedures DataFrame is empty.")

    if use_labs:
        df_labs = preprocess_dataframe('Labs', labs_file_path, labs_columns, labs_columns_select, chunk_size=700000, use_threshold=False)
        df_labs = df_labs if isinstance(df_labs, pl.DataFrame) else pl.from_pandas(df_labs)

        if df_labs.is_empty():
            raise ValueError("Labs DataFrame is empty.")
        
    # Build measurement_configs and track input schemas
    subject_id_col = 'EMPI'
    measurements_by_temporality = {
        TemporalityType.STATIC: {
            'outcomes': {
                DataModality.SINGLE_LABEL_CLASSIFICATION: ['A1cGreaterThan7'],
            }
        },
        TemporalityType.DYNAMIC: {
            'diagnoses': {
                DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
            },
            'procedures': {
                DataModality.MULTI_LABEL_CLASSIFICATION: ['CodeWithType'],
            },
            'labs': {
                DataModality.UNIVARIATE_REGRESSION: ['Result'] if use_labs else []
            }
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
    static_sources['outcomes']['EMPI'] = InputDataType.CATEGORICAL
    static_sources['outcomes']['InitialA1c'] = InputDataType.FLOAT
    static_sources['outcomes']['A1cGreaterThan7'] = InputDataType.CATEGORICAL
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

    inputs = {
        'outcomes': {
            'input_df': outcomes_file_path,
            'columns': {
                'EMPI': ('EMPI', InputDataType.CATEGORICAL),
                'InitialA1c': ('InitialA1c', InputDataType.FLOAT),
                'A1cGreaterThan7': ('A1cGreaterThan7', InputDataType.BOOLEAN),
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
        events_dfs.append(df_dia.select('EMPI', 'Date').rename({'EMPI': 'subject_id'}))

    if not df_dia.is_empty():
        events_dfs.append(df_dia.select('EMPI', 'Date').rename({'EMPI': 'subject_id', 'Date': 'timestamp'}))

    if not df_prc.is_empty():
        events_dfs.append(df_prc.select('EMPI', 'Date').rename({'EMPI': 'subject_id', 'Date': 'timestamp'}))

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
        if df_labs.is_empty():
            raise ValueError("Labs DataFrame is empty.")
        else:
            dynamic_input_schemas.append(
                InputDFSchema(
                    input_df=df_labs,
                    subject_id_col=None,  # Set to None for dynamic input schemas
                    data_schema={
                        'Code': InputDataType.CATEGORICAL,
                        'Result': InputDataType.FLOAT,
                        'Date': InputDataType.TIMESTAMP
                    },
                    event_type='LAB',
                    ts_col='Date',
                    ts_format='%Y-%m-%d %H:%M:%S',
                    type=InputDFType.EVENT
                )
            )


    if not df_dia.is_empty():
        df_dia = df_dia.lazy().with_columns(pl.col('Date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('Date'))

    if not df_prc.is_empty():
        df_prc = df_prc.lazy().with_columns(pl.col('Date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('Date'))

    if use_labs:
        if not df_labs.is_empty():
            df_labs = df_labs.lazy().with_columns(
                pl.col('Date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%.f').alias('timestamp')
            )

    # Collect the 'EMPI' column from the lazy DataFrames
    df_dia_empi = df_dia.select('EMPI').collect()['EMPI']
    df_prc_empi = df_prc.select('EMPI').collect()['EMPI']
    if use_labs:
        df_labs_empi = df_labs.select('EMPI').collect()['EMPI']

    # Process events and measurements data in a streaming fashion
    events_df = pl.concat(events_dfs, how='diagonal').lazy()
    events_df = events_df.with_columns(
        pl.col('subject_id').cast(pl.UInt32),
        pl.col('Date').alias('timestamp'),  # Use the 'Date' column as 'timestamp'
        pl.when(pl.col('subject_id').is_in(df_dia_empi))
        .then(pl.lit('DIAGNOSIS').cast(pl.Categorical))
        .when(pl.col('subject_id').is_in(df_prc_empi))
        .then(pl.lit('PROCEDURE').cast(pl.Categorical))
        # .when(pl.col('subject_id').is_in(df_labs_empi)) 
        # .then(pl.lit('LAB').cast(pl.Categorical))
        .otherwise(pl.lit('UNKNOWN').cast(pl.Categorical))
        .alias('event_type')
    )

    # Add the 'event_id' column to the 'events_df' DataFrame
    events_df = events_df.with_row_index(name='event_id')
    event_types_idxmap = {et: i for i, et in enumerate(events_df.select('event_type').collect()['event_type'].unique().to_list(), start=1)}

    df_outcomes = df_outcomes.lazy().with_columns(pl.col('EMPI').cast(pl.UInt32).alias('subject_id'))

    # Add the 'event_id' column to the 'dynamic_measurements_df' DataFrame
    if use_labs:
        dynamic_measurements_df = df_labs.join(events_df.select('timestamp', 'event_id'), on='timestamp', how='left').collect()

    if not use_labs:
        dynamic_measurements_df = pl.concat([
            df_dia.select('EMPI', 'CodeWithType', 'Date'),
            df_prc.select('EMPI', 'CodeWithType', 'Date')
        ], how='diagonal')
        dynamic_measurements_df = dynamic_measurements_df.rename({'EMPI': 'subject_id', 'Date': 'timestamp'})
        dynamic_measurements_df = dynamic_measurements_df.join(events_df.select('timestamp', 'event_id'), on='timestamp', how='left')

    dataset_schema = DatasetSchema(
        static=InputDFSchema(
            input_df=df_outcomes.collect(),  # Convert LazyFrame to DataFrame
            subject_id_col='subject_id',
            data_schema={
                'InitialA1c': InputDataType.FLOAT,
                'A1cGreaterThan7': InputDataType.CATEGORICAL,
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
        measurement_configs=measurement_configs,
        save_dir="./data"
    )
    config.event_types_idxmap = event_types_idxmap  # Assign event_types_idxmap to the config object

    if config.save_dir is not None:
        dataset_schema_dict = dataset_schema.to_dict()
        with open(config.save_dir / "input_schema.json", "w") as f:
            json.dump(dataset_schema_dict, f, default=json_serial)

    # Check if subjects_df is populated correctly
    if df_outcomes.collect().is_empty():
        raise ValueError("Outcomes DataFrame is empty.")
    else:
        print(f"Outcomes DataFrame shape: {df_outcomes.collect().shape}")
        print(f"Outcomes DataFrame columns: {df_outcomes.collect().columns}")
        
    # Check if events_df is populated correctly
    if events_df.collect().is_empty():
        raise ValueError("Events DataFrame is empty.")
    else:
        print(f"Events DataFrame shape: {events_df.collect().shape}")
        print(f"Events DataFrame columns: {events_df.collect().columns}")
        
    # Check if dynamic_measurements_df is populated 
    if use_labs:
        if dynamic_measurements_df.is_empty():
            raise ValueError("Dynamic Measurements DataFrame is empty.")
        else:
            print(f"Dynamic Measurements DataFrame shape: {dynamic_measurements_df.shape}")
            print(f"Dynamic Measurements DataFrame columns: {dynamic_measurements_df.columns}")
        
    # Check if subjects_df contains the "subject_id" column
    if "subject_id" not in df_outcomes.collect().columns:
        raise ValueError("subjects_df does not contain the 'subject_id' column.")

    # Check if events_df contains the "subject_id" column
    if "subject_id" not in events_df.collect().columns:
        raise ValueError("events_df does not contain the 'subject_id' column.")
        
    # Check if the subject IDs in events_df match the subject IDs in subjects_df
    if not set(events_df.collect()["subject_id"]).issubset(set(df_outcomes.collect()["subject_id"])):
        raise ValueError("Subject IDs in events_df do not match the subject IDs in subjects_df.")

    print("Creating Dataset object...")
    ESD = Dataset(
        config=config,
        subjects_df=df_outcomes.collect(),  # Convert LazyFrame to DataFrame
        events_df=events_df.collect(),  # Convert LazyFrame to DataFrame
        dynamic_measurements_df=dynamic_measurements_df if use_labs else None
    )

    print("Dataset object created.")

    print("Splitting dataset...")

    ESD.split(split_fracs=split, split_names=["train", "tuning", "held_out"], seed=seed)
    print("Dataset split.")
    print("Preprocessing dataset...")
    ESD.preprocess()
    print("Dataset preprocessed.")

    print("Saving dataset...")
    ESD.save(do_overwrite=do_overwrite)
    print("Dataset saved.")

    print("Caching deep learning representation...")
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)
    print("Deep learning representation cached.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
    args = parser.parse_args()
    main(use_dask=args.use_dask)