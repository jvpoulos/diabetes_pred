import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from pathlib import Path
import json
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.dataset_config import DatasetConfig
from EventStream.data.dataset_polars import Dataset
from EventStream.data.pytorch_dataset import PytorchDataset
from pytorch_lightning.loggers import WandbLogger
import polars as pl
import torch

from data_utils import CustomPytorchDataset

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
        cfg["config"]["problem_type"] = cfg["config"]["problem_type"]
        cfg = FinetuneConfig(**cfg, data_config_path=data_config_path, optimization_config_path=optimization_config_path)
    else:
        cfg = FinetuneConfig(**cfg)

    cfg.data_config.dl_reps_dir = Path("data/DL_reps")
    print("Loaded FinetuneConfig:")
    print(cfg)

    DATA_DIR = Path("data")

    # Load the subjects DataFrame
    subjects_df = pl.read_parquet(DATA_DIR / "subjects_df.parquet")
    subjects_df = subjects_df.rename({"subject_id": "subject_id_right"})

    # Load the split task DataFrames
    train_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_test.parquet")

    print(f"dl_reps_dir: {cfg.data_config.dl_reps_dir}")
    # Create the dataset instances with the loaded task DataFrames and subjects DataFrame
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=train_df, device=device)
    tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=val_df, device=device)
    held_out_pyd = CustomPytorchDataset(cfg.data_config, split="held_out", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=test_df, device=device)

    # Create the WandbLogger instance
    wandb_logger_kwargs = {k: v for k, v in cfg.wandb_logger_kwargs.items() if k not in ['do_log_graph', 'team']}
    wandb_logger = WandbLogger(
        **wandb_logger_kwargs,
        save_dir=cfg.save_dir,
    )

    # Pass the dataset instances and WandbLogger instance to the train function
    try:
        _, tuning_metrics, held_out_metrics = train(cfg, train_pyd, tuning_pyd, held_out_pyd, wandb_logger=wandb_logger)
    except ValueError as e:
        if "Train dataset is empty" in str(e):
            print("Error: Train dataset is empty. Please ensure the dataset is properly loaded and preprocessed.")
        else:
            print(f"Error during training: {e}")
    else:
        print("Training completed successfully.")
        if tuning_metrics is not None:
            print(f"Tuning metrics: {tuning_metrics}")
        if held_out_metrics is not None:
            print(f"Held-out metrics: {held_out_metrics}")

if __name__ == "__main__":
    main()