import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig, train
from pathlib import Path
import json
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.dataset_polars import Dataset

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
        cfg["data_config"] = reloaded_data_config  # Assign the loaded data_config to cfg dictionary
        cfg = FinetuneConfig(**cfg, data_config_path=data_config_path, optimization_config_path=optimization_config_path)
    else:
        cfg = FinetuneConfig(**cfg)

    print("Loaded FinetuneConfig:")
    print(cfg)

    # Load the Dataset object
    dataset_path = Path("data/serialized_dataset.pkl")
    ESD = Dataset.load(dataset_path)

    # Load the task DataFrame
    task_df_name = cfg.task_df_name
    task_df_path = Path("data/task_dfs") / f"{task_df_name}.parquet"
    task_df = pl.read_parquet(task_df_path)

    # Remove 'A1cGreaterThan7' from the input data
    ESD.subjects_df = ESD.subjects_df.drop('A1cGreaterThan7')

    # Prepare data for fine-tuning
    cfg.data_config.task_df = task_df
    cfg.data_config.problem_type = cfg.config.problem_type

    try:
        tuning_metrics, held_out_metrics = train(cfg, pretrained=False)
    except Exception as e:
        print(f"Error during training: {e}")
    else:
        print("Training completed successfully.")
        print(f"Tuning metrics: {tuning_metrics}")
        print(f"Held-out metrics: {held_out_metrics}")

if __name__ == "__main__":
    main()