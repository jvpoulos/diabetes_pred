import os
from pathlib import Path

import polars as pl

from EventStream.data.dataset_polars import Dataset

DATA_DIR = Path("data")

# Create the task_dfs directory inside the data directory
TASK_DF_DIR = DATA_DIR / "task_dfs"
TASK_DF_DIR.mkdir(exist_ok=True, parents=True)

ESD = Dataset.load(DATA_DIR)

a1c_greater_than_7 = (
    ESD.subjects_df.lazy()
    .select(
        pl.col("subject_id"),
        pl.col("A1cGreaterThan7").cast(pl.Boolean),
    )
)

a1c_greater_than_7.collect().write_parquet(TASK_DF_DIR / "a1c_greater_than_7.parquet")