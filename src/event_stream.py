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

from EventStream.data.config import (
    DatasetConfig,
    DatasetSchema,
    InputDFSchema,
    MeasurementConfig,
)
from EventStream.data.dataset_polars import Dataset
from EventStream.data.types import (
    DataModality,
    InputDataType,
    InputDFType,
    TemporalityType,
)
from data_utils import read_file, dask_df_to_tensor, preprocess_data
from rdpr_dict import outcomes_columns, dia_columns, prc_columns, outcomes_columns_select, dia_columns_select, prc_columns_select
from scripts.build_dataset import add_to_container

def main(use_dask=False):
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'

    df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select)
    df_prc = read_file(procedures_file_path, prc_columns, prc_columns_select)

    df_dia, df_prc, df_outcomes = preprocess_data(df_dia, df_prc, df_outcomes)

    # Build measurement_configs and track input schemas
    subject_id_col = 'EMPI'
    measurements_by_temporality = {
        TemporalityType.STATIC: {
            DataModality.SINGLE_LABEL_CLASSIFICATION: {
                'outcomes': ['A1cGreaterThan7']
            },
        },
        TemporalityType.DYNAMIC: {
            DataModality.SINGLE_LABEL_CLASSIFICATION: {
                'outcomes': ['A1cGreaterThan7']
            }
        }
    }

    static_sources = defaultdict(dict)
    dynamic_sources = defaultdict(dict)
    measurement_configs = {}

    for temporality, measurements_by_modality in measurements_by_temporality.items():
        schema_source = static_sources if temporality == TemporalityType.STATIC else dynamic_sources
        for modality, measurements_by_source in measurements_by_modality.items():
            if not measurements_by_source:
                continue
            for source_name, measurements in measurements_by_source.items():
                data_schema = schema_source[source_name]

                if type(measurements) is str:
                    measurements = [measurements]
                for m in measurements:
                    measurement_config_kwargs = {
                        "temporality": temporality,
                        "modality": modality,
                    }
                    
                    if type(m) is dict:
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
                    elif isinstance(m, list) and isinstance(m[0], str) and isinstance(m[1], str) and modality == DataModality.MULTIVARIATE_REGRESSION:
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
            input_schema_kwargs["input_df"] = source_schema["input_df"]
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

            if type(cols) is dict:
                cols = [list(t) for t in cols.items()]

            for col in cols:
                if isinstance(col, list) and len(col) == 2 and isinstance(col[0], str) and isinstance(col[1], str) and col[1] in col_schema:
                    in_name, out_name = col
                    schema_key = in_name
                    schema_val = (out_name, col_schema[out_name])
                    # Continue processing for this case
                elif isinstance(col, str) and col in col_schema:
                    col_name = col
                    schema_key = col_name
                    schema_val = (col_name, col_schema[col_name])
                    # Continue processing for this case
                else:
                    raise ValueError(f"{col} unprocessable! Col schema: {col_schema}")

                cols_covered.append(schema_val[0])
                add_to_container(schema_key, schema_val, data_schema)
            input_schema_kwargs[n] = data_schema
            any_schemas_present = True

        if not any_schemas_present and (len(col_schema) > len(cols_covered)):
            input_schema_kwargs["data_schema"] = {}

        for col, dt in col_schema.items():
            if col in cols_covered:
                continue

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
            'input_df': df_outcomes
        },
        'diagnoses': {
            'input_df': df_dia,
            'event_type': 'DIAGNOSIS',
            'ts_col': 'Date',
            'ts_format': '%Y-%m-%d %H:%M:%S'
        },
        'procedures': {
            'input_df': df_prc,
            'event_type': 'PROCEDURE',
            'ts_col': 'Date',
            'ts_format': '%Y-%m-%d %H:%M:%S'
        }
    }

    dataset_schema = DatasetSchema(
        static=build_schema(
            col_schema=static_sources['outcomes'],
            source_schema=inputs['outcomes'],
            subject_id_col=subject_id_col,
            schema_name='outcomes',
        ),
        dynamic=[
            build_schema(
                col_schema=dynamic_sources.get(dynamic_key, {}),
                source_schema=source_schema,
                schema_name=dynamic_key,
            )
            for dynamic_key, source_schema in inputs.items() if dynamic_key != 'outcomes'
        ],
    )

    # Build Config
    split = (0.7, 0.2, 0.1)
    seed = 42
    do_overwrite = False
    DL_chunk_size = 20000

    config = DatasetConfig(measurement_configs=measurement_configs, save_dir="./data")

    if config.save_dir is not None:
        dataset_schema.to_json_file(config.save_dir / "input_schema.json", do_overwrite=do_overwrite)

    ESD = Dataset(config=config, input_schema=dataset_schema)
    ESD.split(split, seed=seed)
    ESD.preprocess()
    ESD.save(do_overwrite=do_overwrite)
    ESD.cache_deep_learning_representation(DL_chunk_size, do_overwrite=do_overwrite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
    args = parser.parse_args()
    main(use_dask=args.use_dask)