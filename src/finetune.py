import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
import logging
import wandb

from data_utils import CustomPytorchDataset

@hydra.main(version_base=None, config_path=".", config_name="finetune_config")
def main(cfg: FinetuneConfig):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting main function")
    
    try:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        data_config_path = cfg.pop("data_config_path", None)
        optimization_config_path = cfg.pop("optimization_config_path", None)

        if data_config_path:
            data_config_fp = Path(data_config_path)
            logger.info(f"Loading data_config from {data_config_fp}")
            reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
            cfg["data_config"] = reloaded_data_config
            cfg["config"]["problem_type"] = cfg["config"]["problem_type"]
            cfg = FinetuneConfig(**cfg, data_config_path=data_config_path, optimization_config_path=optimization_config_path)
        else:
            cfg = FinetuneConfig(**cfg)

        cfg.data_config.dl_reps_dir = Path("data/DL_reps")
        logger.info("Loaded FinetuneConfig:")
        logger.info(str(cfg))

        DATA_DIR = Path("data")
        
        logger.info("Loading subjects DataFrame")
        subjects_df = pl.read_parquet(DATA_DIR / "subjects_df.parquet")
        subjects_df = subjects_df.rename({"subject_id": "subject_id_right"})
        logger.debug(f"Subjects DataFrame shape: {subjects_df.shape}")

        logger.info("Loading task DataFrames")
        train_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_train.parquet")
        val_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_val.parquet")
        test_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_test.parquet")
        
        logger.debug(f"Train DataFrame shape: {train_df.shape}")
        logger.debug(f"Validation DataFrame shape: {val_df.shape}")
        logger.debug(f"Test DataFrame shape: {test_df.shape}")

        logger.info(f"dl_reps_dir: {cfg.data_config.dl_reps_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Creating CustomPytorchDataset instances")
        train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=train_df, device=device)
        tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=val_df, device=device)
        held_out_pyd = CustomPytorchDataset(cfg.data_config, split="held_out", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, task_df=test_df, device=device)

        # Calculate max_index using the datasets
        max_index = max(
            train_pyd.get_max_index(),
            tuning_pyd.get_max_index(),
            held_out_pyd.get_max_index()
        )

        # Set the vocab_size to be the maximum index + 1 (for padding/unknown token)
        cfg.config.vocab_size = max_index + 1
        logger.info(f"Set vocab_size to {cfg.config.vocab_size}")
        
        logger.debug(f"Train dataset cached data shape: {train_pyd.cached_data.shape if hasattr(train_pyd, 'cached_data') else 'No cached_data attribute'}")
        logger.debug(f"Tuning dataset cached data shape: {tuning_pyd.cached_data.shape if hasattr(tuning_pyd, 'cached_data') else 'No cached_data attribute'}")
        logger.debug(f"Held-out dataset cached data shape: {held_out_pyd.cached_data.shape if hasattr(held_out_pyd, 'cached_data') else 'No cached_data attribute'}")

        logger.info("Creating WandbLogger instance")
        wandb_logger_kwargs = {k: v for k, v in cfg.wandb_logger_kwargs.items() if k not in ['do_log_graph', 'team']}
        wandb_logger = WandbLogger(
            **wandb_logger_kwargs,
            save_dir=cfg.save_dir,
        )

        logger.info("Starting training process")
        try:
            wandb.init(config=cfg.to_dict())
            _, tuning_metrics, held_out_metrics = train(cfg, train_pyd, tuning_pyd, held_out_pyd, wandb_logger=wandb_logger)
        except Exception as e:
            logger.exception(f"Error during training: {str(e)}")
            logger.error(f"Train dataset length: {len(train_pyd)}")
            logger.error(f"Tuning dataset length: {len(tuning_pyd)}")
            logger.error(f"Held-out dataset length: {len(held_out_pyd)}")
        else:
            logger.info("Training completed successfully.")
            if tuning_metrics is not None:
                logger.info(f"Tuning metrics: {tuning_metrics}")
            if held_out_metrics is not None:
                logger.info(f"Held-out metrics: {held_out_metrics}")

        logger.info("Main function completed")
    except Exception as e:
        logger.exception(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()