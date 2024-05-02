import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys
import polars as pl
import plotly.express as px
from data_utils import print_covariate_metadata, save_plot
import pandas as pd

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

# Inspect dataframes

print(ESD.subjects_df.head(3))
print(ESD.events_df.head(3))
print(ESD.dynamic_measurements_df.head(3))

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

# Print summary statistics for all variables
print("Summary statistics:")
print(ESD.subjects_df.describe())
print(ESD.events_df.describe())
print(ESD.dynamic_measurements_df.describe())

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