import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from pathlib import Path
import json
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.dataset_config import DatasetConfig
from EventStream.data.dataset_polars import Dataset
import polars as pl

@hydra.main(version_base=None, config_path=".", config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    data_config_path = cfg.pop("data_config_path", None)
    optimization_config_path = cfg.pop("optimization_config_path", None)

    if data_config_path:
        data_config_fp = Path(data_config_path)
        print(f"Loading data_config from {data_config_fp}")
        reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
        cfg["data_config"] = reloaded_data_config
        cfg["config"]["problem_type"] = cfg["config"]["problem_type"]  # Add this line
        cfg = FinetuneConfig(**cfg, data_config_path=data_config_path, optimization_config_path=optimization_config_path)
    else:
        cfg = FinetuneConfig(**cfg)

    print("Loaded FinetuneConfig:")
    print(cfg)

    # Load the Dataset object
    try:
        ESD = Dataset.load(Path("data"), do_pickle_config=False)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error loading dataset: {str(e)}")
        print("Creating a new Dataset object.")
        config_file = Path("data/config.json")
        config = DatasetConfig.from_json_file(config_file)
        ESD = Dataset(config=config)

        # Split the dataset into train, validation, and test sets
        ESD.split(split_fracs=[0.7, 0.2, 0.1])

        # Preprocess the dataset
        ESD.preprocess()

        ESD.save(do_overwrite=True)

    if isinstance(ESD, dict):
        print("Loaded object is a dictionary. Creating a new Dataset object.")
        config_file = Path("data/config.json")
        config = DatasetConfig.from_json_file(config_file)
        ESD = Dataset(config=config)

        # Split the dataset into train, validation, and test sets
        ESD.split(split_fracs=[0.7, 0.2, 0.1])

        # Preprocess the dataset
        ESD.preprocess()

        ESD.save(do_overwrite=True)

    # Load the task DataFrame
    task_df_path = Path("data/task_dfs/a1c_greater_than_7.parquet")
    task_df = pl.read_parquet(task_df_path)

    # Remove 'A1cGreaterThan7' from the input data
    ESD.subjects_df = ESD.subjects_df.drop('A1cGreaterThan7', 'EMPI')

    # Prepare data for fine-tuning
    cfg.data_config.task_df = task_df
    cfg.data_config.problem_type = cfg.config.problem_type

    try:
        tuning_metrics, held_out_metrics = train(cfg)
    except Exception as e:
        print(f"Error during training: {e}")
    else:
        print("Training completed successfully.")
        print(f"Tuning metrics: {tuning_metrics}")
        print(f"Held-out metrics: {held_out_metrics}")

if __name__ == "__main__":
    main()