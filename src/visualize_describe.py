import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_utils import print_covariate_metadata, save_plot, create_scatter_plot, save_plot_measurements_per_subject, save_plot_line, save_plot_heatmap, temporal_dist_pd, save_scatter_plot, plot_avg_measurements_per_patient_per_month, inspect_pickle_file, load_with_dill, split_event_type, temporal_dist_pd_monthly
import pandas as pd
from datetime import datetime
import argparse
import gc

from EventStream.data.dataset_polars import Dataset
from EventStream.data.dataset_config import DatasetConfig

def main(ESD, use_labs=False):
    if use_labs:
        os.makedirs("data_summaries/labs", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries/labs")
    else:
        os.makedirs("data_summaries", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries")

    print("Unique event types in events_df:")
    print(ESD.events_df['event_type'].unique())

    static_covariates = ['InitialA1c_discretized', 'Female', 'Married', 'GovIns', 'English', 'AgeYears_discretized', 'SDI_score_discretized', 'Veteran']

    V = Visualizer(
        plot_by_time=False,
        age_col=None,
        dob_col='dob',
        static_covariates=static_covariates,
        plot_by_age=False,
        n_age_buckets=50,
        time_unit='1w',
        min_sub_to_plot_age_dist=10
    )

    with open(DATA_SUMMARIES_DIR / "dataset_description.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        ESD.describe(do_print_measurement_summaries=True, viz_config=V)
        sys.stdout = original_stdout

    print("Data type of event_id column:", ESD.dynamic_measurements_df['event_id'].dtype)

    print("Summary statistics:")
    pd.options.display.max_columns = ESD.subjects_df.shape[1]
    print(ESD.subjects_df.describe())
    print(ESD.events_df.describe())
    print(ESD.dynamic_measurements_df.describe())

    ESD.events_df = ESD.events_df.with_columns(pl.col("subject_id").cast(pl.UInt32))

    ESD.events_df = ESD.events_df.with_columns([
        pl.col('event_type').map_elements(split_event_type, return_dtype=pl.Utf8).alias('event_type')
    ])

    chunk_size = 10000
    events_per_subject_at_age = (
        ESD.events_df.lazy()
        .join(ESD.subjects_df.lazy().select(['subject_id', 'AgeYears_discretized', 'Female']), on='subject_id')
        .group_by(['AgeYears_discretized', 'Female'])
        .agg(pl.count('event_id').alias('num_events'))
        .collect()
    )

    subjects_with_events_by_age_gender = (
        ESD.subjects_df.lazy()
        .group_by(['AgeYears_discretized', 'Female'])
        .agg(pl.count('subject_id').alias('count'))
        .collect()
    )

    events_per_subject_at_age_pd = events_per_subject_at_age.to_pandas()
    subjects_with_events_pd = subjects_with_events_by_age_gender.to_pandas()
    merged_data = pd.merge(events_per_subject_at_age_pd, subjects_with_events_pd, on=['AgeYears_discretized', 'Female'])
    merged_data['events_per_subject'] = merged_data['num_events'] / merged_data['count']

    subjects_with_events_pd['percentage'] = subjects_with_events_pd['count'] / subjects_with_events_pd['count'].sum() * 100

    subjects_with_events_pd['Female'] = subjects_with_events_pd['Female'].astype(int)
    merged_data['Female'] = merged_data['Female'].astype(int)

    print("Merged data info:")
    print(merged_data.info())
    print("\nMerged data head:")
    print(merged_data.head())

    if merged_data.empty:
        print("Warning: merged_data is empty. Skipping plot creation.")
    else:
        null_counts = merged_data.isnull().sum()
        if null_counts.any():
            print("Warning: merged_data contains null values:")
            print(null_counts)
            merged_data = merged_data.dropna()
            print("Rows with null values removed. New shape:", merged_data.shape)

    save_plot(subjects_with_events_pd, 'AgeYears_discretized', 'percentage', 'Female',
              '% of Patients with an Event by Age Group and Gender', 'Percentage', None,
              DATA_SUMMARIES_DIR / "subjects_with_events_by_age_gender.png")

    save_plot(merged_data, 'AgeYears_discretized', 'events_per_subject', 'Female',
              'Events per Subject by Age Group and Gender', 'Events per Subject', None,
              DATA_SUMMARIES_DIR / "events_per_subject_at_age_by_gender.png")

    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(pl.col('subject_id').cast(pl.UInt32))

    subjects_with_measurements_list = []
    for i in range(0, ESD.subjects_df.shape[0], chunk_size):
        chunk = ESD.subjects_df.slice(i, chunk_size)
        chunk_with_measurements = chunk.join(ESD.dynamic_measurements_df.select('subject_id'), on='subject_id', how='inner').unique()
        subjects_with_measurements_list.append(chunk_with_measurements)
        del chunk, chunk_with_measurements
        gc.collect()

    subjects_with_measurements = pl.concat(subjects_with_measurements_list)
    del subjects_with_measurements_list
    gc.collect()

    total_count_subjects_with_measurements = subjects_with_measurements.select('subject_id').unique().count()

    subjects_with_measurements_by_age_gender = subjects_with_measurements.group_by(['AgeYears_discretized', 'Female']).agg(pl.count('subject_id').alias('count'))
    subjects_with_measurements_by_age_gender = subjects_with_measurements_by_age_gender.with_columns(pl.lit(total_count_subjects_with_measurements).alias('total_count'))
    subjects_with_measurements_by_age_gender = subjects_with_measurements_by_age_gender.with_columns((pl.col('count') / pl.col('total_count') * 100).alias('percentage'))

    measurements_per_subject_at_age = (
        ESD.dynamic_measurements_df.lazy()
        .join(ESD.subjects_df.lazy().select(['subject_id', 'AgeYears_discretized', 'Female']), on='subject_id')
        .group_by(['AgeYears_discretized', 'Female'])
        .agg(pl.count('measurement_id').alias('num_measurements'))
        .collect()
    )

    subjects_with_measurements = (
        ESD.subjects_df.lazy()
        .join(ESD.dynamic_measurements_df.lazy().select('subject_id').unique(), on='subject_id')
        .group_by(['AgeYears_discretized', 'Female'])
        .agg(pl.count('subject_id').alias('count'))
        .collect()
    )

    measurements_per_subject_at_age_pd = measurements_per_subject_at_age.to_pandas()
    subjects_with_measurements_pd = subjects_with_measurements.to_pandas()

    merged_data = pd.merge(measurements_per_subject_at_age_pd, subjects_with_measurements_pd, on=['AgeYears_discretized', 'Female'], how='outer')
    merged_data['measurements_per_subject'] = merged_data['num_measurements'] / merged_data['count']

    print("Shape of merged_data:", merged_data.shape)
    print("Sample of merged_data:")
    print(merged_data.head())

    merged_data = merged_data.sort_values('AgeYears_discretized')

    merged_data = merged_data.dropna(subset=['AgeYears_discretized'])

    if not merged_data.empty:
        fig = px.scatter(merged_data, x='AgeYears_discretized', y='measurements_per_subject', color='Female',
                         title='Average number of measurements per patient, grouped by age and gender',
                         labels={'measurements_per_subject': 'Avg. measurements per patient', 'AgeYears_discretized': 'Age Group'},
                         color_discrete_map={0: "blue", 1: "orange"})
        fig.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Average measurements per patient",
            yaxis_range=[0, merged_data['measurements_per_subject'].max() * 1.1],
            legend_title_text='Gender'
        )
        fig.update_traces(marker=dict(size=8))

        fig.for_each_trace(lambda t: t.update(name='Female' if t.name == '1' else 'Male'))

        fig.update_layout(legend=dict(
            itemsizing='constant',
            title_text='Gender',
            title_font_family='Arial',
            font=dict(family='Arial', size=12, color='black'),
            bordercolor='Black',
            borderwidth=1
        ))
        fig.write_image(str(DATA_SUMMARIES_DIR / "measurements_per_subject_at_age_by_gender.png"))
    else:
        print("Warning: merged_data is empty. Skipping plot creation.")

    measurements_per_subject_pd = (
        ESD.dynamic_measurements_df.lazy()
        .group_by('subject_id')
        .agg(pl.count('measurement_id').alias('num_measurements'))
        .collect()
        .to_pandas()
    )

    measurements_per_subject_pd = measurements_per_subject_pd.merge(
        ESD.subjects_df.lazy().select('subject_id', 'Female', 'AgeYears_discretized').collect().to_pandas(), 
        on='subject_id', 
        how='left'
    )

    print("Unique values in Female column:", measurements_per_subject_pd['Female'].unique())
    print("Columns in measurements_per_subject_pd:", measurements_per_subject_pd.columns)
    print("Sample of measurements_per_subject_pd:")
    print(measurements_per_subject_pd.head())

    create_scatter_plot(merged_data, DATA_SUMMARIES_DIR)

    save_plot_measurements_per_subject(measurements_per_subject_pd, 'num_measurements', 'Female',
                                       'Distribution of measurements per patient', 'Gender',
                                       DATA_SUMMARIES_DIR / "measurements_per_subject_distribution.png")

    print("Shape of measurements_per_subject_at_age:", measurements_per_subject_at_age.shape)
    print("Shape of subjects_with_measurements_by_age_gender:", subjects_with_measurements_by_age_gender.shape)
    print("Shape of merged_data:", merged_data.shape)
    print("Sample of merged_data:")
    print(merged_data.head())

    print("Shape of measurements_per_subject_pd:", measurements_per_subject_pd.shape)
    print("Sample of measurements_per_subject_pd:")
    print(measurements_per_subject_pd.head())

    temporal_distribution_measurements = temporal_dist_pd(ESD.dynamic_measurements_df, 'measurement_id')

    if isinstance(temporal_distribution_measurements, pd.DataFrame):
        df_to_plot = temporal_distribution_measurements
    else:
        df_to_plot = temporal_distribution_measurements.to_pandas()

    save_plot_line(df_to_plot, 'day', 'count', 
                   DATA_SUMMARIES_DIR / 'temporal_distribution_measurements.png', 
                   'Day', 'Count of Measurements', 'Temporal distribution of measurements (daily)', 
                   x_range=(mdates.date2num(datetime(1990, 1, 1)), mdates.date2num(datetime(2022, 12, 31))))

    plot_avg_measurements_per_patient_per_month(ESD.dynamic_measurements_df, filepath=DATA_SUMMARIES_DIR / 'avg_measurements_per_patient_per_month.png')

    measurements_per_patient_per_month_stats = (
        ESD.dynamic_measurements_df.group_by(["subject_id", pl.col("timestamp").dt.strftime("%Y-%m").alias("month")])
        .agg(pl.count("measurement_id").alias("num_measurements"))
        .group_by("subject_id")
        .agg(
            pl.mean("num_measurements").alias("mean"),
            pl.median("num_measurements").alias("median"),
            pl.std("num_measurements").alias("std"),
            pl.min("num_measurements").alias("min"),
            pl.max("num_measurements").alias("max"),
        )
    )

    print("Descriptive statistics for measurements per patient per month:")

    measurement_summary = ESD.dynamic_measurements_df.select(
        pl.col('dynamic_indices').mean().alias('mean_index'),
        pl.col('dynamic_indices').median().alias('median_index'),
        pl.col('dynamic_indices').std().alias('std_index'),
        pl.col('dynamic_indices').min().alias('min_index'),
        pl.col('dynamic_indices').max().alias('max_index'),
        pl.col('dynamic_values').mean().alias('mean_value'),
        pl.col('dynamic_values').median().alias('median_value'),
        pl.col('dynamic_values').std().alias('std_value'),
        pl.col('dynamic_values').min().alias('min_value'),
        pl.col('dynamic_values').max().alias('max_value')
    )

    print("Summary statistics for measurements across all patients:")
    print(measurement_summary)

    total_measurements = ESD.dynamic_measurements_df.shape[0]
    unique_patients = ESD.dynamic_measurements_df['subject_id'].n_unique()

    print(f"Total number of measurements: {total_measurements}")
    print(f"Number of unique patients: {unique_patients}")
    print(f"Average number of measurements per patient: {total_measurements / unique_patients:.2f}")

    print(measurements_per_patient_per_month_stats)

    # summarize measurements across patients instead of at the patient-level:
    # Additional data validation
    print("\nAdditional Data Validation:")
    print("Number of unique subjects in dynamic_measurements_df:", ESD.dynamic_measurements_df['subject_id'].n_unique())
    print("Number of unique subjects in subjects_df:", ESD.subjects_df['subject_id'].n_unique())
    print("Number of unique events in events_df:", ESD.events_df['event_id'].n_unique())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize and describe data.')
    parser.add_argument('--use_labs', action='store_true', help='Include labs data in processing.')
    args = parser.parse_args()

    main(use_labs=args.use_labs)