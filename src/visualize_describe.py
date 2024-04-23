import os
from EventStream.data.dataset_polars import Dataset
from EventStream.data.visualize import Visualizer
from IPython.display import Image
from pathlib import Path
import sys

# Load the dataset
config_file = Path("./data/config.json")  # Convert to Path object
ESD = Dataset.load(Path("./data"))  # Convert to Path object

# Create a Visualizer object
V = Visualizer(
    plot_by_time=False,
    age_col=None,  # Set to None if there is no age column in events_df
    dob_col='dob',
    static_covariates=['AgeYears', 'Female', 'GovIns', 'English', 'SDI_score'],
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

# Visualize the dataset and save the figures to image files
figs = ESD.visualize(viz_config=V)
for i, fig in enumerate(figs):
    fig.write_image(f"data_summaries/fig_{i}.pdf", format="pdf", width=600, height=350, scale=2)