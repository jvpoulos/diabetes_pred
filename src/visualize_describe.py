import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys
import polars as pl
import plotly.express as px

# Load the dataset
config_file = Path("./data/config.json")
ESD = Dataset.load(Path("./data"), do_pickle_config=False)

# Split the dataset into train, validation, and test sets
ESD.split(split_fracs=[0.7, 0.2, 0.1])

# Preprocess the dataset
ESD.preprocess()

# Create a Visualizer object
V = Visualizer(
    plot_by_time=False,
    age_col=None,
    dob_col='dob',
    static_covariates=['InitialA1c', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English', 'AgeYears', 'SDI_score', 'Veteran'],
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

# Plot % of subjects with an event by age and gender
subjects_with_events = ESD.subjects_df.join(ESD.events_df.select('subject_id'), on='subject_id', how='inner').unique()
subjects_with_events_by_age_gender = subjects_with_events.groupby(['AgeYears', 'Female']).agg(pl.count('subject_id').alias('count')).with_columns((pl.col('count') / pl.col('count').sum()).alias('percentage'))
fig = px.bar(subjects_with_events_by_age_gender.to_pandas(), x='AgeYears', y='percentage', color='Female', barmode='group', title='% of Subjects with an Event by Age and Gender')
fig.write_image("data_summaries/subjects_with_events_by_age_gender.pdf", format="pdf", width=600, height=350, scale=2)

# Plot total number of events per subject, by gender
events_per_subject = ESD.events_df.join(ESD.subjects_df, on='subject_id').groupby(['subject_id', 'Female']).agg(pl.count('event_id').alias('num_events'))
fig = px.box(events_per_subject.to_pandas(), x='Female', y='num_events', title='Total Number of Events per Subject, by Gender')
fig.write_image("data_summaries/events_per_subject_by_gender.pdf", format="pdf", width=600, height=350, scale=2)

# Plot events per subject at age, by gender 
events_per_subject_at_age = ESD.events_df.join(ESD.subjects_df, on='subject_id').groupby(['AgeYears', 'Female']).agg(pl.count('event_id').alias('num_events'))
fig = px.scatter(events_per_subject_at_age.to_pandas(), x='AgeYears', y='num_events', color='Female', title='Events per Subject at Age, by Gender')
fig.write_image("data_summaries/events_per_subject_at_age_by_gender.pdf", format="pdf", width=600, height=350, scale=2)