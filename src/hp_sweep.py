import os
import hydra
import wandb
from omegaconf import DictConfig
#from EventStream.data.dataset import Dataset
from EventStream.models.from_scratch_supervised import FromScratchSupervised
from EventStream.utils.train_utils import train_from_scratch_supervised
from EventStream.data.dataset_polars import Dataset
from build_task import TASK_DF_DIR
from EventStream.models.from_scratch_supervised import FromScratchSupervised
from EventStream.transformer.labelers import A1cGreaterThan7Labeler

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="from_scratch_supervised_hyperparameter_sweep_base",
)
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(config=cfg, project="diabetes_pred")
    
    # Load the preprocessed dataset
    data_dir = "data"
    dataset = Dataset.load(data_dir)

    # Load the task-specific DataFrame
    task_df = pl.scan_parquet(TASK_DF_DIR / "a1c_greater_than_7.parquet")

    # Get the hyperparameters from wandb config
    hyperparameters = wandb.config

    labeler = A1cGreaterThan7Labeler(dataset.config)

    model = FromScratchSupervised(
        dataset.deep_learning_sample_input,
        dataset.measurement_configs,
        dataset.vocabs,
        hyperparameters,
        labeler=labeler
    )

    # Train the model
    train_from_scratch_supervised(
        model,
        dataset,
        hyperparameters,
        experiment_name="A1cGreaterThan7_prediction",
        run_name=wandb.run.name,
    )

if __name__ == "__main__":
    main()