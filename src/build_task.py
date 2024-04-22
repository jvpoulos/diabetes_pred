import os
from pathlib import Path

import polars as pl

from EventStream.data.dataset_polars import Dataset

COHORT_NAME = "MIMIC_IV/ESD_06-13-23_150GB_10cpu-1"
PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
DATA_DIR = PROJECT_DIR / "data" / COHORT_NAME
assert DATA_DIR.is_dir()

TASK_DF_DIR = DATA_DIR / "task_dfs"
TASK_DF_DIR.mkdir(exist_ok=True, parents=False)

ESD = Dataset.load(DATA_DIR)

a1c_greater_than_7 = (
    ESD.subjects_df.lazy()
    .select(
        pl.col("subject_id").alias("EMPI"),
        pl.col("A1cGreaterThan7").cast(pl.Boolean).alias("A1cGreaterThan7"),
        pl.exclude(["subject_id", "EMPI"])
    )
)

a1c_greater_than_7.collect().write_parquet(TASK_DF_DIR / "a1c_greater_than_7.parquet")