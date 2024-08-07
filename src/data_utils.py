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
from typing import Dict, Set, List, Any, Sequence, Union
DF_T = Union[pl.DataFrame, pl.LazyFrame]
import dill
import logging
import ast
import psutil
import chardet
import io
import chardet
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from EventStream.data.dataset_base import DatasetBase
from EventStream.data.measurement_config import MeasurementConfig
from EventStream.data.types import DataModality, TemporalityType, NumericDataModalitySubtype
from EventStream.data.vocabulary import Vocabulary
from EventStream.data.preprocessing.standard_scaler import StandardScaler

def try_convert_to_float(x, val_type):
    if val_type == 'Numeric':
        try:
            return float(x)
        except (ValueError, TypeError):
            return None
    return x  # Return as-is for non-numeric types

class ConcreteDataset(DatasetBase):
    PREPROCESSORS = {
        "standard_scaler": StandardScaler,
        # Add other preprocessors here if needed
    }
    def __init__(self, config, subjects_df, events_df, dynamic_measurements_df, code_mapping, **kwargs):
        super().__init__(config, subjects_df, events_df, dynamic_measurements_df, **kwargs)
        self.code_mapping = code_mapping or self._create_code_mapping()
        self.inverse_mapping = {v: k for k, v in self.code_mapping.items()}

    def _get_flat_static_rep(self, feature_columns: list[str], **kwargs) -> pl.LazyFrame:
        static_features = [c for c in feature_columns if c.startswith("static/")]
        return self._normalize_flat_rep_df_cols(
            self._summarize_static_measurements(static_features, **kwargs).collect().lazy(),
            static_features,
            set_count_0_to_null=False,
        )

    def _get_flat_ts_rep(self, feature_columns: list[str], **kwargs) -> pl.LazyFrame:
        return self._normalize_flat_rep_df_cols(
            self._summarize_time_dependent_measurements(feature_columns, **kwargs)
            .join(
                self._summarize_dynamic_measurements(feature_columns, **kwargs),
                on="event_id",
                how="inner",
            )
            .drop("event_id")
            .sort(by=["subject_id", "timestamp"])
            .collect()
            .lazy(),
            [c for c in feature_columns if not c.startswith("static/")],
        )

    def _summarize_over_window(self, df: pl.DataFrame, window_size: str) -> pl.LazyFrame:
        if window_size == "FULL":
            return df.group_by("subject_id").agg([
                pl.all().exclude("subject_id").cumsum().alias(pl.col("^.*$").name)
            ]).explode(pl.all().exclude("subject_id"))
        else:
            return df.group_by_dynamic(
                "timestamp",
                every=window_size,
                by="subject_id",
                closed="left"
            ).agg([
                pl.all().exclude(["subject_id", "timestamp"]).sum().alias(pl.col("^.*$").name)
            ])

    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        try:
            source_df, keys_col_name, vals_col_name, inliers_col_name, _ = self._prep_numerical_source(
                measure, config, source_df
            )

            keys_col = pl.col(keys_col_name)
            vals_col = pl.col(vals_col_name)

            cols_to_drop_at_end = [col for col in config.measurement_metadata if col != measure and col in source_df.columns]

            bound_cols = {
                col: pl.col(col)
                for col in [
                    "drop_upper_bound",
                    "drop_upper_bound_inclusive",
                    "drop_lower_bound",
                    "drop_lower_bound_inclusive",
                    "censor_lower_bound",
                    "censor_upper_bound",
                ]
                if col in source_df.columns
            }

            if bound_cols:
                vals_col = self.drop_or_censor(vals_col, **bound_cols)

            if 'value_type' in source_df.columns:
                value_type = pl.col("value_type")
            else:
                value_type = pl.lit(NumericDataModalitySubtype.FLOAT)  # Default to FLOAT if value_type is not present

            # For dynamic_values, we don't need to apply the categorical transformations
            if measure != 'dynamic_values':
                # Safely cast vals_col to string for concatenation
                safe_vals_col = pl.when(vals_col.is_null()).then(pl.lit("")).otherwise(vals_col.cast(pl.Utf8))

                keys_col = (
                    pl.when(value_type == NumericDataModalitySubtype.DROPPED)
                    .then(keys_col)
                    .when(value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER)
                    .then(keys_col + "__EQ_" + safe_vals_col)
                    .when(value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT)
                    .then(keys_col + "__EQ_" + safe_vals_col)
                    .otherwise(keys_col)
                    .alias(f"{keys_col_name}_transformed")
                )

                vals_col = (
                    pl.when(value_type.is_in([NumericDataModalitySubtype.DROPPED, NumericDataModalitySubtype.CATEGORICAL_INTEGER, NumericDataModalitySubtype.CATEGORICAL_FLOAT]))
                    .then(pl.lit(None))
                    .otherwise(vals_col)
                    .alias(f"{vals_col_name}_transformed")
                )

                source_df = source_df.with_columns([keys_col, vals_col])

            # Apply outlier detection
            if self.config.outlier_detector_config is not None:
                M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)
                inliers_col = ~M.predict_from_polars(vals_col, pl.col("outlier_model")).alias(inliers_col_name)
                source_df = source_df.with_columns(inliers_col)

            # Apply normalization
            if self.config.normalizer_config is not None:
                M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)
                if "normalizer" in source_df.columns:
                    normalized_vals_col = M.predict_from_polars(vals_col, pl.col("normalizer"))
                    source_df = source_df.with_columns(normalized_vals_col.alias(f"{vals_col_name}_normalized"))
                else:
                    print(f"Warning: 'normalizer' column not found for measure {measure}. Skipping normalization.")

            result_df = source_df.drop(cols_to_drop_at_end)
            
            # Rename the transformed columns back to their original names
            if measure != 'dynamic_values':
                result_df = result_df.rename({
                    f"{keys_col_name}_transformed": keys_col_name,
                    f"{vals_col_name}_transformed": vals_col_name
                })
            
            return result_df

        except Exception as e:
            print(f"Error in _transform_numerical_measurement for measure {measure}: {str(e)}")
            print(f"Source DataFrame schema: {source_df.schema}")
            raise

    def _normalize_flat_rep_df_cols(
        self, flat_df: pl.DataFrame, feature_columns: list[str] | None = None, set_count_0_to_null: bool = True
    ) -> pl.DataFrame:
        if feature_columns is None:
            feature_columns = [x for x in flat_df.columns if x not in ("subject_id", "timestamp")]
            cols_to_add = set()
            cols_to_retype = set(feature_columns)
        else:
            cols_to_add = set(feature_columns) - set(flat_df.columns)
            cols_to_retype = set(feature_columns).intersection(set(flat_df.columns))

        cols_to_add = [(c, self._get_flat_col_dtype(c)) for c in cols_to_add]
        cols_to_retype = [(c, self._get_flat_col_dtype(c)) for c in cols_to_retype]

        if "timestamp" in flat_df.columns:
            key_cols = ["subject_id", "timestamp"]
        else:
            key_cols = ["subject_id"]

        flat_df = flat_df.with_columns(
            *[pl.lit(None, dtype=dt).alias(c) for c, dt in cols_to_add],
            *[pl.col(c).cast(dt).alias(c) for c, dt in cols_to_retype],
        ).select(*key_cols, *feature_columns)

        if not set_count_0_to_null:
            return flat_df

        flat_df = flat_df.with_columns(
            pl.when(pl.col("^.*count$") != 0).then(pl.col("^.*count$")).otherwise(None)
        )
        return flat_df

    def _summarize_static_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.subjects_df
        else:
            df = self.subjects_df.filter(pl.col("subject_id").is_in(list(include_only_subjects)))

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "static":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]

            if cfg.modality == DataModality.UNIVARIATE_REGRESSION and cfg.vocabulary is None:
                if allowed_vocab != {m}:
                    raise ValueError(
                        f"Encountered a measure {m} with no vocab but a pre-set feature vocab of "
                        f"{allowed_vocab}"
                    )
                out_dfs[m] = (
                    df.lazy()
                    .filter(pl.col(m).is_not_null())
                    .select("subject_id", pl.col(m).alias(f"static/{m}/{m}/value").cast(pl.Float32))
                )
                continue
            elif cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                raise ValueError(f"{cfg.modality} is not supported for {cfg.temporality} measures.")

            ID_cols = ["subject_id"]
            pivoted_df = (
                df.select(*ID_cols, m)
                .filter(pl.col(m).is_in(list(allowed_vocab)))
                .with_columns(pl.lit(True).alias("__indicator"))
                .pivot(
                    index=ID_cols,
                    columns=m,
                    values="__indicator",
                    aggregate_function=None,
                )
            )

            remap_cols = [c for c in pivoted_df.columns if c not in ID_cols]
            out_dfs[m] = pivoted_df.lazy().select(
                *ID_cols, *[pl.col(c).alias(f"static/{m}/{c}/present").cast(pl.Boolean) for c in remap_cols]
            )

        return pl.concat(list(out_dfs.values()), how="align")

    def _summarize_dynamic_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.dynamic_measurements_df
        else:
            df = self.dynamic_measurements_df.join(
                self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects))).select(
                    "event_id"
                ),
                on="event_id",
                how="inner",
            )

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "dynamic":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]

            if m == 'dynamic_indices':
                out_dfs[m] = (
                    df.lazy()
                    .select("event_id", "dynamic_indices", "dynamic_values")
                    .filter(pl.col("dynamic_indices").is_not_null())
                    .group_by("event_id")
                    .agg(
                        pl.col("dynamic_indices").alias(f"dynamic/{m}/indices"),
                        pl.col("dynamic_values").alias(f"dynamic/{m}/values")
                    )
                )
                continue

            total_observations = int(
                math.ceil(
                    cfg.observation_rate_per_case
                    * cfg.observation_rate_over_cases
                    * sum(self.n_events_per_subject.values())
                )
            )

            count_type = self.get_smallest_valid_uint_type(total_observations)

            if cfg.modality == DataModality.UNIVARIATE_REGRESSION and cfg.vocabulary is None:
                prefix = f"dynamic/{m}/{m}"

                key_col = pl.col(m)
                val_col = pl.col(m).drop_nans().cast(pl.Float32)

                out_dfs[m] = (
                    df.lazy()
                    .select("measurement_id", "event_id", m)
                    .filter(pl.col(m).is_not_null())
                    .group_by("event_id")
                    .agg(
                        pl.col(m).is_not_null().sum().cast(count_type).alias(f"{prefix}/count"),
                        (
                            (pl.col(m).is_not_nan() & pl.col(m).is_not_null())
                            .sum()
                            .cast(count_type)
                            .alias(f"{prefix}/has_values_count")
                        ),
                        val_col.sum().alias(f"{prefix}/sum"),
                        (val_col**2).sum().alias(f"{prefix}/sum_sqd"),
                        val_col.min().alias(f"{prefix}/min"),
                        val_col.max().alias(f"{prefix}/max"),
                    )
                )
                continue
            elif cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                column_cols = [m, m]
                values_cols = [m, cfg.values_column]
                key_prefix = f"{m}_{m}_"
                val_prefix = f"{cfg.values_column}_{m}_"

                key_col = pl.col(f"^{key_prefix}")
                val_col = pl.col(f"^{val_prefix}").drop_nans().cast(pl.Float32)

                aggs = [
                    key_col.is_not_null()
                    .sum()
                    .cast(count_type)
                    .alias(lambda c: f"dynamic/{m}/{c.replace(key_prefix, '')}/count"),
                    (
                        (pl.col(f"^{val_prefix}").is_not_null() & pl.col(f"^{val_prefix}").is_not_nan())
                        .sum()
                        .alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/has_values_count")
                    ),
                    val_col.sum().alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum"),
                    (val_col**2)
                    .sum()
                    .alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/sum_sqd"),
                    val_col.min().alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/min"),
                    val_col.max().alias(lambda c: f"dynamic/{m}/{c.replace(val_prefix, '')}/max"),
                ]
            else:
                column_cols = [m]
                values_cols = [m]
                aggs = [
                    pl.all()
                    .is_not_null()
                    .sum()
                    .cast(count_type)
                    .alias(lambda c: f"dynamic/{m}/{c}/count")
                ]

            ID_cols = ["measurement_id", "event_id"]
            out_dfs[m] = (
                df.select(*ID_cols, *set(column_cols + values_cols))
                .filter(pl.col(m).is_in(allowed_vocab))
                .pivot(
                    index=ID_cols,
                    columns=column_cols,
                    values=values_cols,
                    aggregate_function=None,
                )
                .lazy()
                .drop("measurement_id")
                .group_by("event_id")
                .agg(*aggs)
            )

        return pl.concat(list(out_dfs.values()), how="align")

    def _summarize_time_dependent_measurements(
        self,
        feature_columns: list[str],
        include_only_subjects: set[int] | None = None,
    ) -> pl.LazyFrame:
        if include_only_subjects is None:
            df = self.events_df
        else:
            df = self.events_df.filter(pl.col("subject_id").is_in(list(include_only_subjects)))

        valid_measures = {}
        for feat_col in feature_columns:
            temp, meas, feat, _ = self._parse_flat_feature_column(feat_col)

            if temp != "functional_time_dependent":
                continue

            if meas not in valid_measures:
                valid_measures[meas] = set()
            valid_measures[meas].add(feat)

        out_dfs = {}
        for m, allowed_vocab in valid_measures.items():
            cfg = self.measurement_configs[m]
            if cfg.modality == DataModality.UNIVARIATE_REGRESSION and cfg.vocabulary is None:
                out_dfs[m] = (
                    df.lazy()
                    .filter(pl.col(m).is_not_null())
                    .select(
                        "event_id",
                        "subject_id",
                        "timestamp",
                        pl.col(m).cast(pl.Float32).alias(f"functional_time_dependent/{m}/{m}/value"),
                    )
                )
                continue
            elif cfg.modality == DataModality.MULTIVARIATE_REGRESSION:
                raise ValueError(f"{cfg.modality} is not supported for {cfg.temporality} measures.")

            ID_cols = ["event_id", "subject_id", "timestamp"]
            pivoted_df = (
                df.select(*ID_cols, m)
                .filter(pl.col(m).is_in(allowed_vocab))
                .with_columns(pl.lit(True).alias("__indicator"))
                .pivot(
                    index=ID_cols,
                    columns=m,
                    values="__indicator",
                    aggregate_function=None,
                )
            )

            remap_cols = [c for c in pivoted_df.columns if c not in ID_cols]
            out_dfs[m] = pivoted_df.lazy().select(
                *ID_cols,
                *[
                    pl.col(c).cast(pl.Boolean).alias(f"functional_time_dependent/{m}/{c}/present")
                    for c in remap_cols
                ],
            )

        return pl.concat(list(out_dfs.values()), how="align")

    @staticmethod
    def get_smallest_valid_uint_type(num: int) -> pl.DataType:
        """
        Determines the smallest unsigned integer type that can represent the given number.

        Args:
            num (int): The number to determine the type for.

        Returns:
            pl.DataType: The smallest Polars unsigned integer type that can represent the number.
        """
        if num <= 255:
            return pl.UInt8
        elif num <= 65535:
            return pl.UInt16
        elif num <= 4294967295:
            return pl.UInt32
        else:
            return pl.UInt64

    @staticmethod
    def drop_or_censor(
        col: pl.Expr,
        drop_lower_bound: pl.Expr | None = None,
        drop_lower_bound_inclusive: pl.Expr | None = None,
        drop_upper_bound: pl.Expr | None = None,
        drop_upper_bound_inclusive: pl.Expr | None = None,
        censor_lower_bound: pl.Expr | None = None,
        censor_upper_bound: pl.Expr | None = None,
        **ignored_kwargs,
    ) -> pl.Expr:
        """Appropriately either drops (returns None) or censors (returns the censor value) the value
        based on the provided bounds.

        Args:
            col: The column to apply the drop or censor operations on.
            drop_lower_bound: A lower bound such that if `col` is either below or at or below this level,
                `None` will be returned. If `None`, no lower bound will be applied.
            drop_lower_bound_inclusive: If `True`, returns `None` if ``col <= drop_lower_bound``.
                Else, returns `None` if ``col < drop_lower_bound``.
            drop_upper_bound: An upper bound such that if `col` is either above or at or above this level,
                `None` will be returned. If `None`, no upper bound will be applied.
            drop_upper_bound_inclusive: If `True`, returns `None` if ``col >= drop_upper_bound``.
                Else, returns `None` if ``col > drop_upper_bound``.
            censor_lower_bound: A lower bound such that if `col` is below this level but above
                `drop_lower_bound`, `censor_lower_bound` will be returned. If `None`, no lower censoring will be applied.
            censor_upper_bound: An upper bound such that if `col` is above this level but below
                `drop_upper_bound`, `censor_upper_bound` will be returned. If `None`, no upper censoring will be applied.
        """
        conditions = []

        if drop_lower_bound is not None:
            conditions.append(
                (
                    (col < drop_lower_bound) | ((col == drop_lower_bound) & drop_lower_bound_inclusive),
                    pl.lit(None),
                )
            )

        if drop_upper_bound is not None:
            conditions.append(
                (
                    (col > drop_upper_bound) | ((col == drop_upper_bound) & drop_upper_bound_inclusive),
                    pl.lit(None),
                )
            )

        if censor_lower_bound is not None:
            conditions.append((col < censor_lower_bound, censor_lower_bound))
        if censor_upper_bound is not None:
            conditions.append((col > censor_upper_bound, censor_upper_bound))

        if not conditions:
            return col

        expr = pl.when(conditions[0][0]).then(conditions[0][1])
        for cond, val in conditions[1:]:
            expr = expr.when(cond).then(val)
        return expr.otherwise(col)

    @classmethod
    def _inc_df_col(cls, df: pl.DataFrame, col: str, inc_by: int) -> pl.DataFrame:
        return df.with_columns(pl.col(col) + inc_by)

    @classmethod
    def _read_df(cls, fp: Path, **kwargs) -> pl.DataFrame:
        return pl.read_parquet(fp)

    def _sort_events(self):
        self.events_df = self.events_df.sort("subject_id", "timestamp", descending=False)

    @classmethod
    def _write_df(cls, df: pl.DataFrame, fp: Path, **kwargs):
        do_overwrite = kwargs.get("do_overwrite", False)

        if not do_overwrite and fp.is_file():
            raise FileExistsError(f"{fp} exists and do_overwrite is {do_overwrite}!")

        fp.parent.mkdir(exist_ok=True, parents=True)

        df.write_parquet(fp)

    @classmethod
    def _rename_cols(cls, df: pl.DataFrame, to_rename: dict[str, str]) -> pl.DataFrame:
        return df.rename(to_rename)

    @classmethod
    def _filter_col_inclusion(cls, df: pl.DataFrame, col_inclusion_targets: dict[str, bool | Sequence[Any]]) -> pl.DataFrame:
        filter_exprs = []
        for col, incl_targets in col_inclusion_targets.items():
            match incl_targets:
                case True:
                    filter_exprs.append(pl.col(col).is_not_null())
                case False:
                    filter_exprs.append(pl.col(col).is_null())
                case _:
                    filter_exprs.append(pl.col(col).is_in(list(incl_targets)))

        return df.filter(pl.all_horizontal(filter_exprs))

    @classmethod
    def _concat_dfs(cls, dfs: list[pl.DataFrame]) -> pl.DataFrame:
        return pl.concat(dfs, how="diagonal")

    def _prep_numerical_source(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> tuple[DF_T, str, str, str, pl.DataFrame]:
        try:
            metadata = config.measurement_metadata
            metadata_schema = self.get_metadata_schema(config)

            if config.modality == DataModality.UNIVARIATE_REGRESSION:
                key_col = "const_key"
                val_col = measure
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items()}}
                )
                source_df = source_df.with_columns(pl.lit(measure).cast(pl.Categorical).alias(key_col))
            elif config.modality == DataModality.MULTIVARIATE_REGRESSION:
                key_col = measure
                val_col = config.values_column
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items() if c != key_col}}
                )
            else:
                raise ValueError(f"Called _prep_numerical_source on {config.modality} measure {measure}!")

            # Handle empty outlier_model and normalizer
            for col in ['outlier_model', 'normalizer']:
                if col in metadata_as_polars.columns and len(metadata_as_polars.drop_nulls(col)) == 0:
                    metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias(col))

            # Add val_col if not present
            if val_col not in metadata_as_polars.columns:
                metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias(val_col))

            # Cast columns with proper error handling
            cast_exprs = []
            for col_name, dtype in {key_col: pl.Categorical, val_col: pl.Float64, **metadata_schema}.items():
                if col_name in source_df.columns:
                    try:
                        if col_name == 'dynamic_values':
                            # For dynamic_values, ensure it's treated as Float64
                            cast_exprs.append(pl.col(col_name).cast(pl.Float64).alias(col_name))
                        else:
                            cast_exprs.append(pl.col(col_name).cast(dtype).alias(col_name))
                    except pl.exceptions.ComputeError as e:
                        print(f"Error casting column {col_name} to {dtype}: {str(e)}")
                        print(f"Column contents: {source_df[col_name].head()}")
                        # Use a fallback type if casting fails
                        cast_exprs.append(pl.col(col_name).cast(pl.Object).alias(col_name))

            if cast_exprs:
                source_df = source_df.with_columns(cast_exprs)

            # Rename the values column to 'value' in metadata_as_polars
            if val_col in metadata_as_polars.columns:
                metadata_as_polars = metadata_as_polars.rename({val_col: "value"})

            # Add the key column to metadata_as_polars if it's not present
            if key_col not in metadata_as_polars.columns:
                metadata_as_polars = metadata_as_polars.with_columns(pl.lit(measure).alias(key_col))

            # Ensure the key column in metadata_as_polars has the same dtype as in source_df
            key_col_dtype = source_df[key_col].dtype
            metadata_as_polars = metadata_as_polars.with_columns(
                pl.when(pl.col(key_col) == "dynamic_values")
                .then(pl.lit(0))
                .otherwise(pl.col(key_col))
                .cast(key_col_dtype)
                .alias(key_col)
            )

            # Join with proper error handling
            try:
                source_df = source_df.join(metadata_as_polars, on=key_col, how="left")
            except Exception as e:
                print(f"Error joining source_df with metadata_as_polars: {str(e)}")
                print(f"source_df columns: {source_df.columns}")
                print(f"metadata_as_polars columns: {metadata_as_polars.columns}")
                raise

            return source_df, key_col, val_col, f"{measure}_is_inlier", metadata_as_polars

        except Exception as e:
            print(f"Error in _prep_numerical_source for measure {measure}: {str(e)}")
            print(f"Source DataFrame schema: {source_df.schema}")
            raise

    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: pl.DataFrame) -> Vocabulary:
        if measure == 'dynamic_indices':
            unique_codes = source_df['dynamic_indices'].drop_nulls().unique().sort()
            vocab_elements = unique_codes.to_list()
            el_counts = source_df.group_by('dynamic_indices').count().sort('dynamic_indices')['count'].to_list()
            
            if len(vocab_elements) == 0:
                return Vocabulary(vocabulary=["UNK"], obs_frequencies=[1])
            
            return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
        
        elif config.modality == DataModality.MULTI_LABEL_CLASSIFICATION:
            observations = source_df.get_column(measure).cast(pl.Utf8)
            observations = observations.map_elements(lambda s: s.split("|") if s is not None else [], return_dtype=pl.List(pl.Utf8))
            observations = observations.explode()
        else:
            observations = source_df.get_column(measure)

        observations = observations.drop_nulls()
        N = len(observations)
        if N == 0:
            return None

        try:
            value_counts = observations.value_counts().sort(by="count", descending=True)
            vocab_elements = value_counts[measure].to_list()
            el_counts = value_counts["count"].to_list()
            return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
        except AssertionError as e:
            raise AssertionError(f"Failed to build vocabulary for {measure}") from e

    def _add_time_dependent_measurements(self):
        exprs = []
        join_cols = set()
        for col, cfg in self.config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                continue
            fn = cfg.functor
            join_cols.update(fn.link_static_cols)
            exprs.append(fn.pl_expr().alias(col))

        join_cols = list(join_cols)

        if join_cols:
            self.events_df = (
                self.events_df.join(self.subjects_df.select("subject_id", *join_cols), on="subject_id")
                .with_columns(exprs)
                .drop(join_cols)
            )
        else:
            self.events_df = self.events_df.with_columns(exprs)

    def _agg_by_time(self):
        # Ensure event_id is Int64 in both DataFrames
        self.events_df = self.events_df.with_columns(pl.col("event_id").cast(pl.Int64))
        self.dynamic_measurements_df = self.dynamic_measurements_df.with_columns(pl.col("event_id").cast(pl.Int64))

        event_id_dt = pl.Int64  # Use Int64 consistently

        if self.config.agg_by_time_scale is None:
            grouped = self.events_df.filter(pl.col("timestamp").is_not_null()).group_by(["subject_id", "timestamp"], maintain_order=True)
        else:
            self.events_df = self.events_df.filter(pl.col("timestamp").is_not_null())
            grouped = self.events_df.sort(["subject_id", "timestamp"], descending=False).group_by_dynamic(
                index_column="timestamp",
                every=self.config.agg_by_time_scale,
                period="1h",  # This should match your agg_by_time_scale
                closed="left",
                by="subject_id",
                include_boundaries=False,
                start_by="datapoint",
            )
        grouped = (
            grouped.agg(
                pl.col("event_type").unique().sort(),
                pl.col("event_id").unique().alias("old_event_id"),
            )
            .sort("subject_id", "timestamp", descending=False)
            .with_row_count("event_id")
            .with_columns(
                pl.col("event_id").cast(event_id_dt),
                pl.col("event_type")
                .list.eval(pl.col("").cast(pl.Utf8))
                .list.join("&")
                .cast(pl.Categorical)
                .alias("event_type"),
            )
        )
        new_to_old_set = grouped[["event_id", "old_event_id"]].explode("old_event_id")
        self.events_df = grouped.drop("old_event_id")
        self.dynamic_measurements_df = (
            self.dynamic_measurements_df.rename({"event_id": "old_event_id"})
            .join(new_to_old_set, on="old_event_id", how="left")
            .drop("old_event_id")
        )

    def _load_input_df(self, df, columns, subject_id_col=None, subject_ids_map=None, subject_id_dtype=None, filter_on=None, subject_id_source_col=None):
        if isinstance(df, (str, Path)):
            df = pl.read_csv(df, separator='|')
        elif isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        
        if subject_id_col is not None:
            df = df.with_columns(pl.col(subject_id_col).cast(subject_id_dtype))
            if subject_ids_map is not None:
                df = df.with_columns(pl.col(subject_id_col).replace(subject_ids_map))
        
        if filter_on:
            df = self._filter_col_inclusion(df, filter_on)
        
        return df

    def _process_events_and_measurements_df(self, df, event_type, columns_schema, code_to_index, subject_id_mapping):
        cols_select_exprs = [
            pl.col("Date").alias("timestamp"),
            pl.col("StudyID"),
            pl.lit(event_type).cast(pl.Categorical).alias("event_type")
        ]

        for col in columns_schema:
            if col in df.columns and col != "StudyID" and col != "Date":
                cols_select_exprs.append(pl.col(col))

        df = (
            df.filter(pl.col("Date").is_not_null() & pl.col("StudyID").is_not_null())
            .select(cols_select_exprs)
            .unique()
            .with_row_count("event_id")
        )

        df = df.with_columns([
            pl.col('StudyID').cast(pl.Utf8).replace(subject_id_mapping).alias('subject_id').cast(pl.UInt32)
        ])

        events_df = df.select("event_id", "subject_id", "timestamp", "event_type")
        
        dynamic_cols = ["event_id", "subject_id", "timestamp"]
        if event_type in ['DIAGNOSIS', 'PROCEDURE']:
            dynamic_cols.append(pl.col("CodeWithType").replace(code_to_index).alias("dynamic_indices"))
            dynamic_cols.append(pl.lit(None).cast(pl.Float64).alias("dynamic_values"))
        elif event_type == 'LAB':
            dynamic_cols.append(pl.col("Code").replace(code_to_index).alias("dynamic_indices"))
            dynamic_cols.append(pl.col("Result").cast(pl.Float64).alias("dynamic_values"))
        
        dynamic_measurements_df = df.select(dynamic_cols)

        return events_df, dynamic_measurements_df

    def _resolve_ts_col(self, df, ts_col, out_name="timestamp"):
        if isinstance(ts_col, list):
            ts_expr = pl.min(ts_col)
            ts_to_drop = [c for c in ts_col if c != out_name]
        else:
            ts_expr = pl.col(ts_col)
            ts_to_drop = [ts_col] if ts_col != out_name else []

        return df.with_columns(ts_expr.alias(out_name)).drop(ts_to_drop)

    def _split_range_events_df(self, df):
        df = df.filter(pl.col("start_time") <= pl.col("end_time"))

        eq_df = df.filter(pl.col("start_time") == pl.col("end_time"))
        ne_df = df.filter(pl.col("start_time") != pl.col("end_time"))

        st_col, end_col = pl.col("start_time").alias("timestamp"), pl.col("end_time").alias("timestamp")
        drop_cols = ["start_time", "end_time"]
        return (
            eq_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(end_col).drop(drop_cols),
        )

    def _update_subject_event_properties(self):
        if self.events_df is not None:
            all_event_types = set()
            for event_type in self.events_df.get_column("event_type").unique():
                all_event_types.update(event_type.split('&'))
            self.event_types = sorted(list(all_event_types))

            self.event_types_idxmap = {event_type: idx for idx, event_type in enumerate(self.event_types, start=1)}

            n_events_pd = self.events_df.get_column("subject_id").value_counts(sort=False).to_pandas()
            n_events_pd.columns = ['subject_id', 'counts']
            self.n_events_per_subject = dict(zip(n_events_pd['subject_id'], n_events_pd['counts']))
            self.subject_ids = set(self.n_events_per_subject.keys())

        if self.subjects_df is not None:
            subjects_with_no_events = (
                set(self.subjects_df.get_column("subject_id").to_list()) - self.subject_ids
            )
            for sid in subjects_with_no_events:
                self.n_events_per_subject[sid] = 0
            self.subject_ids.update(subjects_with_no_events)

    def _validate_initial_dfs(self, subjects_df, events_df, dynamic_measurements_df):
        subjects_df, subjects_id_type = self._validate_initial_df(
            subjects_df, "subject_id", TemporalityType.STATIC
        )
        events_df, event_id_type = self._validate_initial_df(
            events_df,
            "event_id",
            TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            {"subject_id": subjects_id_type} if subjects_df is not None else None,
        )
        if events_df is not None:
            if "event_type" not in events_df.columns:
                raise ValueError("Missing event_type column!")
            events_df = events_df.with_columns(pl.col("event_type").cast(pl.Categorical))

            if "timestamp" not in events_df.columns or events_df.select("timestamp").dtypes[0] != pl.Datetime:
                raise ValueError("Malformed timestamp column!")

        if dynamic_measurements_df is not None:
            linked_ids = {}
            if events_df is not None:
                linked_ids["event_id"] = event_id_type

            dynamic_measurements_df, dynamic_measurement_id_types = self._validate_initial_df(
                dynamic_measurements_df, "measurement_id", TemporalityType.DYNAMIC, linked_ids
            )

        return subjects_df, events_df, dynamic_measurements_df

    def _validate_initial_df(self, source_df, id_col_name, valid_temporality_type, linked_id_cols=None):
        if source_df is None:
            return None, None

        if linked_id_cols:
            for id_col, id_col_dt in linked_id_cols.items():
                if id_col not in source_df.columns:
                    raise ValueError(f"Missing mandatory linkage col {id_col}")
                source_df = source_df.with_columns(pl.col(id_col).cast(id_col_dt))

        if id_col_name not in source_df.columns:
            print(f"Creating {id_col_name} column as it doesn't exist")
            source_df = source_df.with_row_count(name=id_col_name)
        else:
            if source_df[id_col_name].n_unique() != len(source_df):
                print(f"Warning: {id_col_name} is not unique. Creating a new unique {id_col_name}")
                source_df = source_df.with_row_count(name=f"unique_{id_col_name}")
                source_df = source_df.drop(id_col_name).rename({f"unique_{id_col_name}": id_col_name})

        id_col, id_col_dt = self._validate_id_col(source_df.select(id_col_name))

        if id_col_name != id_col.name:
            source_df = source_df.rename({id_col.name: id_col_name})

        for col, cfg in self.config.measurement_configs.items():
            match cfg.modality:
                case DataModality.DROPPED:
                    continue
                case DataModality.UNIVARIATE_REGRESSION:
                    cat_col, val_col = None, col
                case DataModality.MULTIVARIATE_REGRESSION:
                    cat_col, val_col = col, cfg.values_column
                case _:
                    cat_col, val_col = col, None

            if cat_col is not None and cat_col in source_df.columns:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {cat_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(cat_col).cast(pl.Utf8).cast(pl.Categorical))

            if val_col is not None and val_col in source_df.columns:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {val_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(val_col).cast(pl.Float64))

        return source_df, id_col_dt

    def _validate_id_col(self, id_col):
        col_name = id_col.columns[0]
        unique_count = id_col[col_name].n_unique()
        total_count = id_col.shape[0]
        
        if unique_count != total_count:
            print(f"Warning: ID column {col_name} is not unique. Consider regenerating this column.")

        dtype = id_col.dtypes[0]
        print(f"Current data type of {col_name}: {dtype}")

        if dtype in (pl.Float32, pl.Float64):
            check_expr = (pl.col(col_name) == pl.col(col_name).round(0)) & (pl.col(col_name) >= 0)
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            check_expr = pl.col(col_name) >= 0
        elif dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            check_expr = pl.lit(True)
        else:
            raise ValueError(f"ID column {col_name} is not a numeric type!")

        is_valid = id_col.select(check_expr.all()).item()

        if not is_valid:
            raise ValueError(f"ID column {col_name} contains negative or non-integer values!")

        max_val = id_col[col_name].max()
        dt = self.get_smallest_valid_uint_type(max_val)

        return id_col.select(pl.col(col_name).cast(dt))[col_name], dt

    def _transform_multi_label_classification(self, measure, config, source_df):
        print(f"Transforming multi-label classification measurement: {measure}")
        if measure == 'dynamic_indices':
            # For dynamic_indices, we preserve the original values
            source_df = source_df.with_columns([
                pl.col('dynamic_indices').cast(pl.UInt32),
                pl.col('dynamic_values').cast(pl.Utf8)
            ])
            
            return source_df
        
        # Original code for other measures (if any)
        # Convert the column to string type
        source_df = source_df.with_columns(pl.col(measure).cast(str))

        # Split the column into a list
        source_df = source_df.with_columns(
            pl.col(measure).map_elements(lambda x: [x] if x else [], return_dtype=pl.List(pl.Utf8)).alias(f"{measure}_list")
        )

        # Explode the list column and count the occurrences
        transformed_df = source_df.with_columns(
            pl.col(f"{measure}_list").explode().alias(measure)
        ).with_columns(
            pl.col(f"{measure}_list").map_elements(len, return_dtype=pl.UInt32).alias(f"{measure}_counts")
        )

        return transformed_df

    def _get_preprocessing_model(self, model_config: dict[str, Any], for_fit: bool = False) -> Any:
        if "cls" not in model_config:
            raise KeyError("Missing mandatory preprocessor class configuration parameter `'cls'`.")
        if model_config["cls"] not in self.PREPROCESSORS:
            raise KeyError(
                f"Invalid preprocessor model class {model_config['cls']}! {self.__class__.__name__} options are "
                f"{', '.join(self.PREPROCESSORS.keys())}"
            )

        model_cls = self.PREPROCESSORS[model_config["cls"]]
        if for_fit:
            return model_cls(**{k: v for k, v in model_config.items() if k != "cls"})
        else:
            return model_cls

    def get_metadata_schema(self, config: MeasurementConfig) -> dict[str, pl.DataType]:
        schema = {
            "value_type": pl.Categorical,
        }

        if self.config.outlier_detector_config is not None:
            M = self._get_preprocessing_model(self.config.outlier_detector_config, for_fit=False)
            schema["outlier_model"] = pl.Struct(M.params_schema())
        if self.config.normalizer_config is not None:
            M = self._get_preprocessing_model(self.config.normalizer_config, for_fit=False)
            schema["normalizer"] = pl.Struct(M.params_schema())

        metadata = config.measurement_metadata
        if metadata is None:
            return schema

        for col in (
            "drop_upper_bound",
            "drop_lower_bound",
            "censor_upper_bound",
            "censor_lower_bound",
            "drop_upper_bound_inclusive",
            "drop_lower_bound_inclusive",
        ):
            if col in metadata:
                schema[col] = pl.Float64 if 'bound' in col else pl.Boolean

        # Handle 'dynamic_values' explicitly
        if 'dynamic_values' in metadata:
            schema['dynamic_values'] = pl.Object  # Use a flexible type for dynamic_values

        # Remove any None values from the schema
        schema = {k: v for k, v in schema.items() if v is not None}

        print(f"Metadata schema: {schema}")  # Add this line for debugging

        return schema

    def _convert_dynamic_indices_to_indices(self):
        print("Entering _convert_dynamic_indices_to_indices")
        print("Columns in dynamic_measurements_df:", self.dynamic_measurements_df.columns)
        
        code_column = 'dynamic_indices'
        print(f"Using {code_column} as code column")
        
        # Check the data type of the dynamic_indices column
        dynamic_indices_dtype = self.dynamic_measurements_df[code_column].dtype
        print(f"Data type of {code_column}: {dynamic_indices_dtype}")

        # Print some sample values before conversion
        print("Sample of dynamic_indices before conversion:")
        print(self.dynamic_measurements_df.select(code_column).head())

        # Keep dynamic_indices as a string
        self.dynamic_measurements_df = self.dynamic_measurements_df.with_columns([
            pl.col(code_column).cast(pl.Utf8).alias(code_column)
        ])

        print("Exiting _convert_dynamic_indices_to_indices")
        print("Data types in dynamic_measurements_df:")
        for col in self.dynamic_measurements_df.columns:
            print(f"{col}: {self.dynamic_measurements_df[col].dtype}")

        # Print a sample of the dynamic_indices column after conversion
        print("Sample of dynamic_indices column after conversion:")
        print(self.dynamic_measurements_df.select(code_column).head())

        # Count of unique values after conversion
        unique_count = self.dynamic_measurements_df[code_column].n_unique()
        print(f"Number of unique values in dynamic_indices after conversion: {unique_count}")

        # Distribution of dynamic_indices
        value_counts = self.dynamic_measurements_df[code_column].value_counts().sort("count", descending=True)
        print("Top 10 most common dynamic_indices values:")
        print(value_counts.head(10))

    def _create_code_mapping(self):
        all_codes = set()
        if self.dynamic_measurements_df is not None and 'dynamic_indices' in self.dynamic_measurements_df.columns:
            all_codes.update(self.dynamic_measurements_df['dynamic_indices'].cast(pl.Utf8).unique().to_list())
        all_codes = {str(code) for code in all_codes if code is not None}
        sorted_codes = sorted(all_codes)
        code_mapping = {code: idx for idx, code in enumerate(sorted_codes, start=1)}
        code_mapping['UNKNOWN'] = len(code_mapping) + 1
        return code_mapping

    def _denormalize(self, events_df: pl.DataFrame, col: str) -> pl.DataFrame:
        if self.config.normalizer_config is None:
            return events_df
        elif self.config.normalizer_config["cls"] != "standard_scaler":
            raise ValueError(f"De-normalizing from {self.config.normalizer_config} not yet supported!")
        
        config = self.measurement_configs[col]
        if config.modality != DataModality.UNIVARIATE_REGRESSION:
            raise ValueError(f"De-normalizing {config.modality} is not currently supported.")
        
        normalizer_params = config.measurement_metadata.normalizer
        return events_df.with_columns(
            ((pl.col(col) * normalizer_params["std_"]) + normalizer_params["mean_"]).alias(col)
        )

    def _fit_measurement_metadata(self, measure: str, config: MeasurementConfig, source_df: pl.DataFrame) -> pd.DataFrame:
        # Implementation depends on your specific requirements
        # This is a placeholder implementation
        metadata = pd.DataFrame()
        return metadata

    def _total_possible_and_observed(self, measure: str, config: MeasurementConfig, source_df: pl.DataFrame) -> tuple[int, int, int]:
        if config.temporality == TemporalityType.DYNAMIC:
            num_possible = source_df.select(pl.col("event_id").n_unique()).item()
            num_non_null = source_df.select(
                pl.col("event_id").filter(pl.col(measure).is_not_null()).n_unique()
            ).item()
            num_total = source_df.select(pl.col(measure).is_not_null().sum()).item()
        else:
            num_possible = len(source_df)
            num_non_null = len(source_df.drop_nulls(measure))
            num_total = num_non_null
        return num_possible, num_non_null, num_total

    def _transform_categorical_measurement(self, measure: str, config: MeasurementConfig, source_df: pl.DataFrame) -> pl.DataFrame:
        print(f"Transforming categorical measurement: {measure}")
        if measure not in source_df.columns:
            print(f"Warning: Measure {measure} not found in the source DataFrame.")
            return source_df
        if config.vocabulary is None:
            print(f"Warning: Vocabulary is None for measure {measure}. Skipping transformation.")
            return source_df

        vocab_el_col = pl.col(measure)
        
        if measure == 'dynamic_values':
            # For dynamic_values, we keep it as Float64 and don't transform
            transform_expr = [
                vocab_el_col.cast(pl.Float64).alias(measure)
            ]
        elif measure == 'dynamic_indices':
            # For dynamic_indices, we keep it as a string (Utf8)
            transform_expr = [
                vocab_el_col.cast(pl.Utf8).alias(measure)
            ]
        else:
            # Convert vocabulary to strings to ensure compatibility
            vocab_as_strings = [str(v) for v in config.vocabulary.vocabulary]
            vocab_lit = pl.Series(vocab_as_strings).cast(pl.Categorical)

            transform_expr = [
                pl.when(vocab_el_col.is_null())
                .then(vocab_el_col)  # Preserve null values
                .when(~vocab_el_col.cast(pl.Utf8).is_in(vocab_lit))
                .then(vocab_el_col)  # Preserve values not in vocabulary
                .otherwise(vocab_el_col)
                .cast(pl.Categorical)
                .alias(measure)
            ]

        return source_df.with_columns(transform_expr)

    def _update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: list[str]):
        old_df = getattr(self, attr)

        # Only update columns that exist in both old_df and df
        cols_to_update = [col for col in cols_to_update if col in old_df.columns and col in df.columns]

        # Remove duplicates from cols_to_update
        cols_to_update = list(dict.fromkeys(cols_to_update))

        # Create a new dataframe with only the columns to be updated
        new_df = df.select([id_col] + cols_to_update)

        # Update the old dataframe with the new values
        updated_df = old_df.join(new_df, on=id_col, how="left")

        # Replace the old columns with the new ones
        for col in cols_to_update:
            updated_df = updated_df.with_columns(pl.col(f"{col}_right").alias(col))
        
        # Drop the temporary right columns
        updated_df = updated_df.drop([f"{col}_right" for col in cols_to_update])

        setattr(self, attr, updated_df)

    def build_DL_cached_representation(self, subject_ids: list[int] | None = None, do_sort_outputs: bool = False) -> pl.DataFrame:
        print("Starting build_DL_cached_representation")
        subject_measures, event_measures, dynamic_measures = [], [], ["dynamic_indices"]
        for m in self.unified_measurements_vocab[1:]:
            temporality = self.measurement_configs[m].temporality
            match temporality:
                case TemporalityType.STATIC:
                    subject_measures.append(m)
                case TemporalityType.FUNCTIONAL_TIME_DEPENDENT:
                    event_measures.append(m)
                case TemporalityType.DYNAMIC:
                    dynamic_measures.append(m)
                case _:
                    raise ValueError(f"Unknown temporality type {temporality} for {m}")

        if subject_ids:
            subjects_df = self._filter_col_inclusion(self.subjects_df, {"subject_id": subject_ids})
            events_df = self._filter_col_inclusion(self.events_df, {"subject_id": subject_ids})
            dynamic_measurements_df = self._filter_col_inclusion(self.dynamic_measurements_df, {"subject_id": subject_ids})
        else:
            subjects_df = self.subjects_df
            events_df = self.events_df
            dynamic_measurements_df = self.dynamic_measurements_df

        static_data = subjects_df.select(
            "subject_id",
            *[pl.col(m) for m in subject_measures],
            "InitialA1c", "Female", "Married", "GovIns", 
            "English", "AgeYears", "SDI_score", "Veteran"
        )

        subject_id_dtype = pl.UInt32
        static_data = static_data.with_columns(pl.col("subject_id").cast(subject_id_dtype))
        events_df = events_df.with_columns(pl.col("subject_id").cast(subject_id_dtype))
        dynamic_measurements_df = dynamic_measurements_df.with_columns(pl.col("subject_id").cast(subject_id_dtype))

        dynamic_data = dynamic_measurements_df.select(
            "event_id",
            "dynamic_indices",
            "dynamic_values"
        )

        dynamic_data = dynamic_data.with_columns([
            pl.col("dynamic_indices").cast(pl.UInt32),
            pl.col("dynamic_values").cast(pl.Float64)
        ])

        # Handle dynamic_values (keep as Float64)
        dynamic_data = dynamic_data.with_columns(pl.col("dynamic_values").cast(pl.Float64))

        event_data = events_df.select(
            "subject_id",
            "timestamp",
            "event_id",
            "event_type",
            *event_measures
        ).join(
            dynamic_data,
            on="event_id",
            how="left"
        )

        for col in ["dynamic_indices", "dynamic_values"]:
            if col not in event_data.columns:
                event_data = event_data.with_columns(pl.lit(None).alias(col))

        event_data = event_data.with_columns([
            pl.col("dynamic_indices").cast(pl.UInt32),
            pl.col("dynamic_values").cast(pl.Utf8)
        ])

        if do_sort_outputs:
            event_data = event_data.sort("event_id")

        out = static_data.join(event_data, on="subject_id", how="inner")

        if do_sort_outputs:
            out = out.sort("subject_id")

        return out

def process_events_and_measurements_df(
    df: pl.DataFrame,
    event_type: str,
    columns_schema: list,
    code_to_index: dict,
    subject_id_mapping: dict
):
    print(f"Processing {event_type} data...")
    print(f"Input columns: {df.columns}")

    cols_select_exprs = [
        pl.col("Date").alias("timestamp"),
        pl.col("StudyID"),
        pl.lit(event_type).cast(pl.Utf8).alias("event_type")
    ]

    for col in columns_schema:
        if col in df.columns and col != "StudyID" and col != "Date":
            cols_select_exprs.append(pl.col(col))

    df = (
        df.filter(pl.col("Date").is_not_null() & pl.col("StudyID").is_not_null())
        .select(cols_select_exprs)
        .unique()
        .with_row_count("event_id")
    )

    # Add subject_id column
    df = df.with_columns([
        pl.col('StudyID').cast(pl.Utf8).replace(subject_id_mapping).alias('subject_id').cast(pl.UInt32)
    ])

    events_df = df.select("event_id", "subject_id", "timestamp", "event_type")
    
    dynamic_cols = ["event_id", "subject_id", "timestamp"]
    if event_type in ['DIAGNOSIS', 'PROCEDURE']:
        dynamic_cols.append(pl.col("CodeWithType").replace(code_to_index).alias("dynamic_indices"))
        dynamic_cols.append(pl.lit(None).cast(pl.Float64).alias("Result"))
        dynamic_cols.append(pl.lit(None).cast(pl.Float64).alias("dynamic_values"))
    elif event_type == 'LAB':
        dynamic_cols.append(pl.col("Code").replace(code_to_index).alias("dynamic_indices"))
        if "Result" in df.columns:
            dynamic_cols.append(pl.col("Result").cast(pl.Float64))
            dynamic_cols.append(pl.col("Result").cast(pl.Float64).alias("dynamic_values"))
        else:
            dynamic_cols.append(pl.lit(None).cast(pl.Float64).alias("Result"))
            dynamic_cols.append(pl.lit(None).cast(pl.Float64).alias("dynamic_values"))
    
    dynamic_measurements_df = df.select(dynamic_cols)

    print(f"Output columns for {event_type}: {dynamic_measurements_df.columns}")
    print(f"Sample output for {event_type}:")
    print(dynamic_measurements_df.head())

    return events_df, dynamic_measurements_df
        
def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

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
    if 'CodeWithType' in df.columns:
        return df.with_columns([
            pl.col('CodeWithType').map_dict(code_to_index, default=0).alias('dynamic_indices')
        ]).filter(pl.col('dynamic_indices') != 0)
    elif 'Code' in df.columns:  # For labs data
        return df.with_columns([
            pl.col('Code').map_dict(code_to_index, default=0).alias('dynamic_indices')
        ]).filter(pl.col('dynamic_indices') != 0)
    else:
        raise ValueError("Neither 'CodeWithType' nor 'Code' column found in the dataframe")

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

def create_code_mapping(df_dia, df_prc, df_labs=None):
    """
    Create a mapping from codes to indices for diagnoses, procedures, and optionally labs.
    
    Args:
    df_dia: DataFrame containing diagnosis codes
    df_prc: DataFrame containing procedure codes
    df_labs: Optional DataFrame containing lab codes
    
    Returns:
    A dictionary mapping codes to indices
    """
    all_codes = set(df_dia['CodeWithType'].unique()) | set(df_prc['CodeWithType'].unique())
    
    if df_labs is not None:
        all_codes |= set(df_labs['Code'].unique())
    
    # Remove any None or empty string values
    all_codes = {code for code in all_codes if code and str(code).strip()}
    
    sorted_codes = sorted(all_codes)
    code_to_index = {str(code): idx for idx, code in enumerate(sorted_codes, start=1)}
    
    print(f"Total unique codes: {len(code_to_index)}")
    print(f"Sample of code_to_index: {dict(list(code_to_index.items())[:5])}")
    
    return code_to_index
    
class CustomPytorchDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, dl_reps_dir, subjects_df, df_dia, df_prc, df_labs=None, task_df=None, device=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initializing CustomPytorchDataset for split: {split}")

        self.split = split
        self.dl_reps_dir = Path(dl_reps_dir)
        self.subjects_df = subjects_df
        self.task_df = task_df
        self.df_dia = df_dia
        self.df_prc = df_prc
        self.df_labs = df_labs
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
        """Create a mapping from codes to indices for diagnoses, procedures, and labs."""
        all_codes = set(self.df_dia['CodeWithType'].unique()) | set(self.df_prc['CodeWithType'].unique())
        
        if self.df_labs is not None:
            all_codes |= set(self.df_labs['Code'].unique())
        
        # Remove any None or empty string values
        all_codes = {code for code in all_codes if code and str(code).strip()}
        
        sorted_codes = sorted(all_codes)
        self.code_to_index = {str(code): idx for idx, code in enumerate(sorted_codes, start=1)}
        
        self.logger.info(f"Total unique codes: {len(self.code_to_index)}")
        self.logger.info(f"Sample of code_to_index: {dict(list(self.code_to_index.items())[:5])}")
        
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

        self.logger.info(f"After filtering, {self.cached_data.shape[0]} rows remain")
        self.logger.info(f"Sample of dynamic_indices after filtering: {self.cached_data['dynamic_indices'].head(5)}")

    def __len__(self):
        return self.cached_data.shape[0]

    def __getitem__(self, idx):
        try:
            row = self.cached_data.row(idx)
            subject_id = row[self.cached_data.columns.index('subject_id')]
            raw_dynamic_indices = row[self.cached_data.columns.index('dynamic_indices')]
            dynamic_indices = self.process_dynamic_indices(raw_dynamic_indices)
            
            # Ensure dynamic_indices is not empty
            if dynamic_indices.numel() == 0:
                dynamic_indices = torch.tensor([0], dtype=torch.long)  # Use 0 for padding
            
            item = {
                'subject_id': subject_id,
                'dynamic_indices': dynamic_indices,
                'labels': torch.tensor(row[self.cached_data.columns.index('label')], dtype=torch.float32) if self.has_task else None,
            }
            
            # Handle dynamic_values
            if 'dynamic_values' in self.cached_data.columns:
                raw_dynamic_values = row[self.cached_data.columns.index('dynamic_values')]
                item['dynamic_values'] = self.process_dynamic_values(raw_dynamic_values)
            else:
                item['dynamic_values'] = torch.tensor([0.0], dtype=torch.float32)  # Default value
            
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

    def process_dynamic_values(self, raw_dynamic_values):
        if raw_dynamic_values is None:
            return torch.tensor([0.0], dtype=torch.float32)  # Default value for missing dynamic_values
        
        if isinstance(raw_dynamic_values, str):
            try:
                values = ast.literal_eval(raw_dynamic_values)
                if isinstance(values, (list, tuple)):
                    return torch.tensor(values, dtype=torch.float32)
                else:
                    return torch.tensor([float(values)], dtype=torch.float32)
            except:
                self.logger.warning(f"Failed to parse string dynamic_values: {raw_dynamic_values}, using default value")
                return torch.tensor([0.0], dtype=torch.float32)
        
        if isinstance(raw_dynamic_values, (float, int)):
            return torch.tensor([float(raw_dynamic_values)], dtype=torch.float32)
        
        if isinstance(raw_dynamic_values, (list, tuple)):
            return torch.tensor(raw_dynamic_values, dtype=torch.float32)
        
        self.logger.error(f"Unexpected type for dynamic_values: {type(raw_dynamic_values)}")
        return torch.tensor([0.0], dtype=torch.float32)

    def process_dynamic_indices(self, indices):
        if indices is None:
            self.logger.warning(f"Received None for dynamic_indices, using default value")
            return torch.tensor([0], dtype=torch.long)
        
        if isinstance(indices, (int, np.integer)):
            return torch.tensor([indices], dtype=torch.long)
        
        if isinstance(indices, str):
            try:
                indices = ast.literal_eval(indices)
            except:
                self.logger.warning(f"Failed to parse string dynamic_indices: {indices}, using default value")
                return torch.tensor([0], dtype=torch.long)
        
        if isinstance(indices, list):
            return torch.tensor(indices, dtype=torch.long)
        
        self.logger.error(f"Unexpected type for dynamic_indices: {type(indices)}")
        return torch.tensor([0], dtype=torch.long)

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
        dynamic_measurements_df.group_by(["subject_id", "month"])
        .agg(pl.count("measurement_id").alias("num_measurements"))
        .group_by("month")
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
    monthly_counts = df.group_by('month').agg(pl.count(measure).alias('count'))
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

def preprocess_dataframe(df_name, file_path, columns, selected_columns, min_frequency=None, debug=False):
    # Detect file encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read(100000)  # Read the first 100000 bytes
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']

    print(f"Detected encoding: {encoding} (confidence: {confidence})")

    # If confidence is low or encoding is ASCII, use a fallback encoding
    if confidence < 0.9 or encoding == 'ascii':
        encoding = 'iso-8859-1'
        print(f"Using fallback encoding: {encoding}")

    # If in debug mode, only process 0.05% of the data
    if debug:
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            total_rows = sum(1 for _ in file)
        nrows = int(total_rows * 0.0005)  # 0.05% of the data
        print(f"Debug mode: Processing {nrows} rows out of {total_rows}")
    else:
        nrows = None

    # Use Polars to read the CSV file
    try:
        df = pl.read_csv(file_path, separator='|', columns=selected_columns, n_rows=nrows, encoding=encoding)
    except UnicodeDecodeError:
        print(f"Error with {encoding}, trying with 'iso-8859-1'")
        df = pl.read_csv(file_path, separator='|', columns=selected_columns, n_rows=nrows, encoding='iso-8859-1')

    # Process the data based on df_name
    if df_name == 'Outcomes':
        df = df.with_columns(pl.col('A1cGreaterThan7').cast(pl.Float32))

    if df_name in ['Diagnoses', 'Procedures', 'Labs']:
        df = df.with_columns(pl.col('Date').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False).cast(pl.Datetime('us')))
        
    if df_name in ['Diagnoses', 'Procedures']:
        df = df.drop_nulls(subset=['Date', 'CodeWithType'])
        if min_frequency is not None:
            code_counts = df.group_by('CodeWithType').agg(pl.count('CodeWithType').alias('count'))
            if isinstance(min_frequency, int):
                frequent_codes = code_counts.filter(pl.col('count') >= min_frequency)['CodeWithType']
            else:
                min_count_threshold = min_frequency * df.height
                frequent_codes = code_counts.filter(pl.col('count') >= min_count_threshold)['CodeWithType']
            df = df.filter(pl.col('CodeWithType').is_in(frequent_codes))

    elif df_name == 'Labs':
        df = df.drop_nulls(subset=['Date', 'Code', 'Result'])
        if min_frequency is not None:
            code_counts = df.group_by('Code').agg(pl.count('Code').alias('count'))
            if isinstance(min_frequency, int):
                frequent_codes = code_counts.filter(pl.col('count') >= min_frequency)['Code']
            else:
                min_count_threshold = min_frequency * df.height
                frequent_codes = code_counts.filter(pl.col('count') >= min_count_threshold)['Code']
            df = df.filter(pl.col('Code').is_in(frequent_codes))

    # Optimize memory usage
    df = df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int64),
        pl.col(pl.UInt32).cast(pl.Int64),
        pl.col(pl.Utf8).cast(pl.Utf8)
    ])

    return df

def optimize_labs_data(df_labs):
    # Add event_id column
    df_labs = df_labs.with_row_count("event_id")
    
    # Keep 'Date' as datetime, 'Result' and 'Code' as string
    df_labs = df_labs.with_columns([
        pl.col('Date').cast(pl.Datetime),
        pl.col('Result').cast(pl.Utf8),
        pl.col('Code').cast(pl.Utf8)
    ])
    
    # Categorize 'Source' column if it exists
    if 'Source' in df_labs.columns:
        df_labs = df_labs.with_columns(pl.col('Source').cast(pl.Categorical))
    
    return df_labs
    
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
        encoded_df = dd.concat([ddf[['StudyID', 'Code', 'Result']], encoded_features], axis=1)

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
            encoded_chunk = pd.concat([chunk[['StudyID', 'Code', 'Result']], pd.DataFrame(encoded_features.toarray())], axis=1)

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