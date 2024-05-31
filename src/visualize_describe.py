import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys
import polars as pl
import plotly.express as px
from data_utils import print_covariate_metadata, save_plot, save_plot_measurements_per_subject, save_plot_line, save_plot_heatmap, temporal_dist_pd, save_scatter_plot
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# Load the dataset
config_file = Path("./data/config.json")
try:
    ESD = Dataset.load(Path("./data"), do_pickle_config=False)
except AttributeError:
    # If the AttributeError occurs during loading, create a new Dataset object
    from EventStream.data.dataset_config import DatasetConfig
    config = DatasetConfig(save_dir=Path("./data"))
    ESD = Dataset(config=config)

# Split the dataset into train, validation, and test sets
ESD.split(split_fracs=[0.7, 0.2, 0.1])

# Preprocess the dataset
ESD.preprocess()

# Checks
assert 'A1cGreaterThan7' in ESD.subjects_df.columns, "A1cGreaterThan7 column is missing in subjects_df"
assert pl.datatypes.Categorical(ESD.subjects_df['A1cGreaterThan7']), "A1cGreaterThan7 should be a categorical column"

assert 'CodeWithType' in ESD.dynamic_measurements_df.columns, "CodeWithType column is missing in dynamic_measurements_df"

# Inspect dataframes

print(ESD.subjects_df.head())
print(ESD.events_df.head())
print(ESD.dynamic_measurements_df.head())

print("Unique event types in events_df:")
print(ESD.events_df['event_type'].unique())

print("Unique CodeWithType values in dynamic_measurements_df:")
print(ESD.dynamic_measurements_df['CodeWithType'].unique())

print("Unique values in A1cGreaterThan7:")
print(ESD.subjects_df['A1cGreaterThan7'].unique())

# vocabulary indices
print("Unified Vocabulary:")
for word, index in ESD.unified_vocabulary_idxmap.items():
    print(f"{index}: {word}")

static_covariates = ['InitialA1c', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']

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

# Create a data_summaries folder if it doesn't exist
os.makedirs("data_summaries", exist_ok=True)

# Describe the dataset and save the text output to a file
with open("data_summaries/dataset_description.txt", "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    ESD.describe(do_print_measurement_summaries=True, viz_config=V)
    sys.stdout = original_stdout

print("Unique values in A1cGreaterThan7:")
print(ESD.subjects_df['A1cGreaterThan7'].unique())

assert 'event_id' in ESD.dynamic_measurements_df.columns, "event_id column is missing in dynamic_measurements_df"
assert ESD.dynamic_measurements_df['event_id'].is_numeric(), "event_id should be a numeric column"
print("Data type of event_id column:", ESD.dynamic_measurements_df['event_id'].dtype)

# Print summary statistics for all variables
print("Summary statistics:")
print(ESD.subjects_df.describe())
print(ESD.events_df.describe())
print(ESD.dynamic_measurements_df.describe())

print("Frequency distribution of CodeWithType:")
print(ESD.dynamic_measurements_df['CodeWithType'].value_counts())

# Data preparation
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
          "data_summaries/subjects_with_events_by_age_gender.png")

save_plot(merged_data, 'AgeYears', 'events_per_subject', 'Female',
          'Events per Subject at Age, by Gender', 'Events per Subject', (13, 80),
          "data_summaries/events_per_subject_at_age_by_gender.png")

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
          "data_summaries/subjects_with_measurements_by_age_gender.png")

save_plot(merged_data, 'AgeYears', 'measurements_per_subject', 'Female',
          'Average number of measurements per patient, grouped by age and gender', 'Avg. measurements per patient', (13, 80),
          "data_summaries/measurements_per_subject_at_age_by_gender.png")

# Distribution of measurements per subject:

# Calculate the number of measurements per subject
measurements_per_subject = ESD.dynamic_measurements_df.groupby('subject_id').agg(
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
save_plot_measurements_per_subject(measurements_per_subject_pd, 'num_measurements', 'Female', 'Distribution of measurements per patient', 'Number of Measurements', "data_summaries/measurements_per_subject_distribution.png")

print("Shape of subjects_with_measurements:", subjects_with_measurements.shape)
print("Shape of subjects_with_events:", subjects_with_events.shape)

print("Columns in subjects_with_measurements:", subjects_with_measurements.columns)
print("Columns in subjects_with_events:", subjects_with_events.columns)
# Temporal distribution of events and measurements:
# The x-axis of both plots will be binned by day, showing the daily counts instead of individual timestamps.

# Temporal distribution of measurements:
temporal_distribution_measurements = temporal_dist_pd(ESD.dynamic_measurements_df, 'measurement_id')
save_plot_line(temporal_distribution_measurements, 'day', 'count', 'temporal_distribution_measurements.png', 'Day', 'Count of Measurements', 'Temporal distribution of measurements (daily)', x_range=(mdates.date2num(datetime(1990, 1, 1)), mdates.date2num(datetime(2022, 12, 31))))

# Correlation between static covariates and outcome variables:

# Assuming 'A1cGreaterThan7' is the outcome variable
static_covariates = ['InitialA1c', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran']

# Convert the subjects DataFrame to pandas
subjects_df_pd = ESD.subjects_df.to_pandas()

# Convert 'A1cGreaterThan7' to numeric values (1 for 'true', 0 for 'false')
subjects_df_pd['A1cGreaterThan7'] = subjects_df_pd['A1cGreaterThan7'].map({'true': 1, 'false': 0})

# Calculate the correlation matrix
corr_matrix = subjects_df_pd[static_covariates + ['A1cGreaterThan7']].corr()

# Define the desired labels for the covariates
covariate_labels = ['Baseline HbA1c', 'Female', 'Married', 'Medicare/Medicaid', 'English lang.', 'Age', 'SDI score', 'Veteran', 'HbA1c ≥ 7%']

# Create a dictionary to map the original column names to the desired labels
label_map = dict(zip(static_covariates + ['A1cGreaterThan7'], covariate_labels))

# Rename the columns and index of the correlation matrix using the label map
corr_matrix_labeled = corr_matrix.rename(columns=label_map, index=label_map)

# Save the plot with the updated labels
save_plot_heatmap(corr_matrix_labeled, corr_matrix_labeled.columns, corr_matrix_labeled.columns, 'Correlation matrix', "data_summaries/correlation_matrix.png")

# Create a scatter plot of measurements per subject vs InitialA1c
measurements_per_subject = ESD.dynamic_measurements_df.group_by('subject_id').agg(pl.count('measurement_id').alias('num_measurements'))
measurements_and_initial_a1c = measurements_per_subject.join(ESD.subjects_df[['subject_id', 'InitialA1c']], on='subject_id', how='left')

measurements_and_initial_a1c = ESD._denormalize(measurements_and_initial_a1c, 'InitialA1c')
measurements_and_initial_a1c_pd = measurements_and_initial_a1c.to_pandas()

save_scatter_plot(measurements_and_initial_a1c_pd, 'InitialA1c', 'num_measurements', 'Measurements per patient vs. baseline HbA1c (%)', 'Number of Measurements', "data_summaries/measurements_per_subject_vs_initial_a1c.png", x_label='Baseline HbA1c (%)', x_scale=(0, 20))