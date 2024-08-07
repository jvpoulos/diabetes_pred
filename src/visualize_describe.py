import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys
import polars as pl
import plotly.express as px
from data_utils import print_covariate_metadata, save_plot, save_plot_measurements_per_subject, save_plot_line, save_plot_heatmap, temporal_dist_pd, save_scatter_plot, plot_avg_measurements_per_patient_per_month, inspect_pickle_file, load_with_dill, temporal_dist_pd_monthly
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import argparse

from EventStream.data.dataset_polars import Dataset
from EventStream.data.dataset_config import DatasetConfig

def main(ESD, use_labs=False):
    if use_labs:
        # Create a data_summaries folder if it doesn't exist
        os.makedirs("data_summaries/labs", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries/labs")
    else:
        # Create a data_summaries folder if it doesn't exist
        os.makedirs("data_summaries", exist_ok=True)
        DATA_SUMMARIES_DIR = Path("data_summaries")

    print("Unique event types in events_df:")
    print(ESD.events_df['event_type'].unique())

    static_covariates = ['InitialA1c', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']

    # Create a Visualizer object
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

    # Describe the dataset and save the text output to a file
    with open(DATA_SUMMARIES_DIR / "dataset_description.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        ESD.describe(do_print_measurement_summaries=True, viz_config=V)
        sys.stdout = original_stdout

    print("Data type of event_id column:", ESD.dynamic_measurements_df['event_id'].dtype)

    # Print summary statistics for all variables
    print("Summary statistics:")
    pd.options.display.max_columns = ESD.subjects_df.shape[1]
    print(ESD.subjects_df.describe())
    print(ESD.events_df.describe())
    print(ESD.dynamic_measurements_df.describe())

    # Data preparation
    # Cast subject_id in events_df to UInt32
    ESD.events_df = ESD.events_df.with_columns(pl.col("subject_id").cast(pl.UInt32))

    subjects_with_events = ESD.subjects_df.join(ESD.events_df.select('subject_id'), on='subject_id', how='inner').unique()

    # Calculate the total count of subjects with events
    total_count_subjects_with_events = subjects_with_events.select('subject_id').unique().count()

    # Calculate the number of subjects with events by age and gender
    subjects_with_events_by_age_gender = subjects_with_events.group_by(['AgeYears', 'Female']).agg(pl.count('subject_id').alias('count'))

    # Join with the total count of subjects with events
    subjects_with_events_by_age_gender = subjects_with_events_by_age_gender.with_columns(pl.lit(total_count_subjects_with_events).alias('total_count'))

    # Calculate the percentage relative to the total count of subjects with events
    subjects_with_events_by_age_gender = subjects_with_events_by_age_gender.with_columns((pl.col('count') / pl.col('total_count') * 100).alias('percentage'))

    # Join events with subjects and calculate events per subject at each age, by gender
    events_per_subject_at_age = ESD.events_df.join(ESD.subjects_df, on='subject_id').group_by(['AgeYears', 'Female']).agg(pl.count('event_id').alias('num_events'))

    # Convert to a DataFrame and calculate events per subject by merging with count data
    events_per_subject_at_age_pd = events_per_subject_at_age.to_pandas()
    subjects_with_events_pd = subjects_with_events_by_age_gender.to_pandas()

    # Merge based on common columns to get event counts per subject
    merged_data = pd.merge(events_per_subject_at_age_pd, subjects_with_events_pd[['AgeYears', 'Female', 'count']], on=['AgeYears', 'Female'])
    merged_data['events_per_subject'] = merged_data['num_events'] / merged_data['count']

    # Convert back to Polars DataFrame if needed
    merged_data_pl = pl.DataFrame(merged_data)

    # Create and save the plots
    save_plot(subjects_with_events_pd, 'AgeYears', 'percentage', 'Female', 
              '% of Patients with an Event by Age and Gender', 'Percentage', (13, 80),
              DATA_SUMMARIES_DIR / "subjects_with_events_by_age_gender.png")

    save_plot(merged_data, 'AgeYears', 'events_per_subject', 'Female',
              'Events per Subject at Age, by Gender', 'Events per Subject', (13, 80),
              DATA_SUMMARIES_DIR / "events_per_subject_at_age_by_gender.png")

    # Calculate the total count of subjects with measurements
    # Cast the subject_id column in ESD.dynamic_measurements_df to UInt32
    ESD.dynamic_measurements_df = ESD.dynamic_measurements_df.with_columns(
        pl.col('subject_id').cast(pl.UInt32)
    )

    # Now the join operation should work
    subjects_with_measurements = ESD.subjects_df.join(
        ESD.dynamic_measurements_df.select('subject_id'), 
        on='subject_id', 
        how='inner'
    ).unique()
    total_count_subjects_with_measurements = subjects_with_measurements.select('subject_id').unique().count()

    # Calculate the number of subjects with measurements by age and gender
    subjects_with_measurements_by_age_gender = subjects_with_measurements.group_by(['AgeYears', 'Female']).agg(pl.count('subject_id').alias('count'))
    subjects_with_measurements_by_age_gender = subjects_with_measurements_by_age_gender.with_columns(pl.lit(total_count_subjects_with_measurements).alias('total_count'))
    subjects_with_measurements_by_age_gender = subjects_with_measurements_by_age_gender.with_columns((pl.col('count') / pl.col('total_count') * 100).alias('percentage'))

    # Join measurements with subjects and calculate measurements per subject at each age, by gender
    measurements_per_subject_at_age = ESD.dynamic_measurements_df.join(ESD.subjects_df, on='subject_id').group_by(['AgeYears', 'Female']).agg(pl.count('measurement_id').alias('num_measurements'))
    measurements_per_subject_at_age_pd = measurements_per_subject_at_age.to_pandas()
    subjects_with_measurements_pd = subjects_with_measurements_by_age_gender.to_pandas()
    merged_data = pd.merge(measurements_per_subject_at_age_pd, subjects_with_measurements_pd[['AgeYears', 'Female', 'count']], on=['AgeYears', 'Female'])
    merged_data['measurements_per_subject'] = merged_data['num_measurements'] / merged_data['count']

    # Create and save the plots
    save_plot(subjects_with_measurements_pd, 'AgeYears', 'percentage', 'Female', 
              'Share of patients with measurements by age and gender', 'Percentage', (13, 80),
              DATA_SUMMARIES_DIR / "subjects_with_measurements_by_age_gender.png")

    save_plot(merged_data, 'AgeYears', 'measurements_per_subject', 'Female',
              'Average number of measurements per patient, grouped by age and gender', 'Avg. measurements per patient', (13, 80),
              DATA_SUMMARIES_DIR / "measurements_per_subject_at_age_by_gender.png")

    # Distribution of measurements per subject:

    # Calculate the number of measurements per subject
    measurements_per_subject = ESD.dynamic_measurements_df.group_by('subject_id').agg(
        pl.count('measurement_id').alias('num_measurements')
    )

    # Convert subject_id to UInt32
    measurements_per_subject = measurements_per_subject.with_columns(
        pl.col("subject_id").cast(pl.UInt32)
    )
    subjects_df = ESD.subjects_df.with_columns(pl.col("subject_id").cast(pl.UInt32))

    # Join with the subjects DataFrame to get the 'Female' column
    measurements_per_subject = measurements_per_subject.join(
        subjects_df.select(["subject_id", "Female"]), on="subject_id", how="left"
    )

    # Convert to pandas DataFrame
    measurements_per_subject_pd = measurements_per_subject.to_pandas()

    # Save the plot using the custom function
    save_plot_measurements_per_subject(measurements_per_subject_pd, 'num_measurements', 'Female', 'Distribution of measurements per patient', 'Number of Measurements', DATA_SUMMARIES_DIR / "measurements_per_subject_distribution.png")

    print("Shape of subjects_with_measurements:", subjects_with_measurements.shape)
    print("Shape of subjects_with_events:", subjects_with_events.shape)

    print("Columns in subjects_with_measurements:", subjects_with_measurements.columns)
    print("Columns in subjects_with_events:", subjects_with_events.columns)

    # Temporal distribution of events and measurements:
    # The x-axis of both plots will be binned by day, showing the daily counts instead of individual timestamps.

    # Temporal distribution of measurements:
    temporal_distribution_measurements = temporal_dist_pd(ESD.dynamic_measurements_df, 'measurement_id')
    save_plot_line(temporal_distribution_measurements, 'day', 'count', 'temporal_distribution_measurements.png', 'Day', 'Count of Measurements', 'Temporal distribution of measurements (daily)', x_range=(mdates.date2num(datetime(1990, 1, 1)), mdates.date2num(datetime(2022, 12, 31))))

    # Temporal distribution of measurements (monthly)
    temporal_distribution_measurements = temporal_dist_pd_monthly(ESD.dynamic_measurements_df, 'measurement_id')
    temporal_distribution_measurements_pd = temporal_distribution_measurements.to_pandas()
    temporal_distribution_measurements_pd['month'] = pd.to_datetime(temporal_distribution_measurements_pd['month'])

    save_plot_line(
        temporal_distribution_measurements_pd, 
        'month', 
        'count', 
        'temporal_distribution_measurements_monthly.png', 
        'Month', 
        'Count of Measurements', 
        'Temporal Distribution of Measurements (monthly)', 
        x_range=(mdates.date2num(datetime(1990, 1, 1)), mdates.date2num(datetime(2022, 12, 31)))
    )

    # Plot average measurements per patient per month
    plot_avg_measurements_per_patient_per_month(ESD.dynamic_measurements_df)

    # Calculate descriptive statistics for measurements per patient per month
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

    # Calculate summary statistics across all patients
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

    # Calculate the total number of measurements and unique patients
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