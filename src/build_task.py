import os
import sys
from pathlib import Path
import pickle
import polars as pl
import argparse

from EventStream.data.dataset_polars import Dataset
from EventStream.tasks.profile import summarize_binary_task
from EventStream.data.dataset_config import DatasetConfig
from data_utils import read_file, preprocess_dataframe, json_serial, add_to_container, read_parquet_file, generate_time_intervals, create_code_mapping, map_codes_to_indices, create_inverse_mapping, inspect_pickle_file, load_with_dill
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select, prc_columns_select, labs_columns_select

def main(ESD, use_labs=False):
    if use_labs:
        DATA_DIR = Path("data/labs")
    else:
        DATA_DIR = Path("data")

    # Create the task_dfs directory inside the data directory
    TASK_DF_DIR = DATA_DIR / "task_dfs"
    TASK_DF_DIR.mkdir(exist_ok=True, parents=True)

    # Finetuning objective: Predict A1cGreaterThan7 using single-label classification
    a1c_greater_than_7 = (
        ESD.subjects_df.lazy()
        .select(
            pl.col("subject_id"),
            pl.col("A1cGreaterThan7").cast(pl.Boolean).fill_null(False).alias("label"),
        )
        .with_columns(pl.lit(None, dtype=pl.Datetime).alias("start_time"))
        .with_columns(pl.lit(None, dtype=pl.Datetime).alias("end_time"))
    )

    # Collect the lazy DataFrame
    a1c_greater_than_7_df = a1c_greater_than_7.collect()

    # Filter the task DataFrame based on the split subject IDs
    train_df = a1c_greater_than_7_df.filter(pl.col("subject_id").is_in(ESD.split_subjects["train"]))
    val_df = a1c_greater_than_7_df.filter(pl.col("subject_id").is_in(ESD.split_subjects["tuning"]))
    test_df = a1c_greater_than_7_df.filter(pl.col("subject_id").is_in(ESD.split_subjects["held_out"]))

    # Save the split dataframes to parquet files
    train_df.write_parquet(TASK_DF_DIR / "a1c_greater_than_7_train.parquet")
    val_df.write_parquet(TASK_DF_DIR / "a1c_greater_than_7_val.parquet")
    test_df.write_parquet(TASK_DF_DIR / "a1c_greater_than_7_test.parquet")

    # Summarize

    result = summarize_binary_task(train_df)
    print("Summary of binary task:")
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build task dataframes.')
    parser.add_argument('--use_labs', action='store_true', help='Include labs data in processing.')
    args = parser.parse_args()

    main(use_labs=args.use_labs)