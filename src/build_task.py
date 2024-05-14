import os
from pathlib import Path

import polars as pl

from EventStream.data.dataset_polars import Dataset
from EventStream.data.dataset_config import DatasetConfig

DATA_DIR = Path("data")

# Load the Dataset object
try:
    ESD = Dataset.load(DATA_DIR)
except AttributeError:
    # If the AttributeError occurs during loading, create a new Dataset object
    config = DatasetConfig(save_dir=DATA_DIR)
    ESD = Dataset(config=config)

# Create the task_dfs directory inside the data directory
TASK_DF_DIR = DATA_DIR / "task_dfs"
TASK_DF_DIR.mkdir(exist_ok=True, parents=True)

# Create a single-label binary classification task for A1cGreaterThan7
a1c_greater_than_7 = (
    ESD.subjects_df.lazy()
    .select(
        pl.col("subject_id"),
        pl.col("A1cGreaterThan7").cast(pl.Boolean).fill_null(False).alias("label"),
    )
    .with_columns(pl.lit(None, dtype=pl.Datetime).alias("start_time"))
    .with_columns(pl.lit(None, dtype=pl.Datetime).alias("end_time"))
)

a1c_greater_than_7.collect().write_parquet(TASK_DF_DIR / "a1c_greater_than_7.parquet")