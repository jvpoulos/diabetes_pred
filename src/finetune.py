import torch.multiprocessing as mp
import dill
import os

# Ensure WandB run is finished before starting a new one
import wandb
if wandb.run:
    wandb.finish()

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
import argparse

from data_utils import CustomPytorchDataset

def evaluate_expression(expression, config):
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = vars(config)
    return eval(expression, {"__builtins__": __builtins__}, config_dict)

@hydra.main(version_base=None, config_path=".", config_name="finetune_config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    use_labs = cfg.get('use_labs', False)
    sweep = cfg.get('sweep', False)
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting main function")
    logger.info(f"Using labs: {use_labs}")
    logger.info(f"Hyperparameter sweep: {sweep}")
    
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

        if sweep: 
            # Evaluate the expressions for problematic parameters
            cfg['config']['hidden_size'] = evaluate_expression(str(cfg['config']['hidden_size']), cfg['config'])
            cfg['data_config']['max_seq_len'] = evaluate_expression(str(cfg['data_config']['max_seq_len']), cfg['config'])
            cfg['data_config']['min_seq_len'] = evaluate_expression(str(cfg['data_config']['min_seq_len']), cfg['config'])
            cfg['optimization_config']['end_lr_frac_of_init_lr'] = evaluate_expression(str(cfg['optimization_config']['end_lr_frac_of_init_lr']), cfg['optimization_config'])
            cfg['optimization_config']['validation_batch_size'] = evaluate_expression(str(cfg['optimization_config']['validation_batch_size']), cfg['optimization_config'])

        if use_labs:
            DATA_DIR = Path("data/labs")
            cfg.data_config.dl_reps_dir = Path("data/labs/DL_reps")
        else:
            DATA_DIR = Path("data/")
            cfg.data_config.dl_reps_dir = Path("data/DL_reps")

        logger.info("Loaded FinetuneConfig:")
        logger.info(str(cfg))

        logger.info("Checking parquet files.")

        train_parquet_files = list(cfg.data_config.dl_reps_dir.glob("train*.parquet"))
        logger.info(f"Train Parquet files: {train_parquet_files}")
        for file in train_parquet_files:
            logger.info(f"File {file} size: {os.path.getsize(file)} bytes")
        
        logger.info("Loading subjects DataFrame")
        subjects_df = pl.read_parquet(DATA_DIR / "subjects_df.parquet")
        subjects_df = subjects_df.rename({"subject_id": "subject_id_right"})
        logger.debug(f"Subjects DataFrame shape: {subjects_df.shape}")

        logger.info("Loading task DataFrames")
        train_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_train.parquet")
        val_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_val.parquet")
        test_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_test.parquet")
        
        logger.info(f"Train task DataFrame shape: {train_df.shape}")
        logger.info(f"Train task DataFrame columns: {train_df.columns}")
        logger.info(f"Sample of train task DataFrame:\n{train_df.head()}")

        logger.info("Loading diagnosis and procedure DataFrames")
        df_dia = pl.read_parquet(DATA_DIR / "df_dia.parquet")
        df_prc = pl.read_parquet(DATA_DIR / "df_prc.parquet")
        logger.debug(f"Diagnosis DataFrame shape: {df_dia.shape}")
        logger.debug(f"Procedure DataFrame shape: {df_prc.shape}")

        logger.info(f"dl_reps_dir: {cfg.data_config.dl_reps_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Creating CustomPytorchDataset instances")
        train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=train_df, device=device)
        tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=val_df, device=device)
        held_out_pyd = CustomPytorchDataset(cfg.data_config, split="held_out", dl_reps_dir=cfg.data_config.dl_reps_dir, subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=test_df, device=device)

        logger.info(f"Train dataset length: {len(train_pyd)}")
        logger.info(f"Tuning dataset length: {len(tuning_pyd)}")
        logger.info(f"Held-out dataset length: {len(held_out_pyd)}")

        if len(train_pyd) == 0 or len(tuning_pyd) == 0 or len(held_out_pyd) == 0:
            logger.error("One or more datasets are empty. Please check your data loading process.")
            return

        max_index = max(
            train_pyd.get_max_index(),
            tuning_pyd.get_max_index(),
            held_out_pyd.get_max_index()
        )

        cfg.config.vocab_size = max_index + 1
        logger.info(f"Set vocab_size to {cfg.config.vocab_size}")

        logger.debug(f"Train dataset cached data: {[df.shape for df in train_pyd.cached_data_list] if hasattr(train_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
        logger.debug(f"Tuning dataset cached data: {[df.shape for df in tuning_pyd.cached_data_list] if hasattr(tuning_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
        logger.debug(f"Held-out dataset cached data: {[df.shape for df in held_out_pyd.cached_data_list] if hasattr(held_out_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")

        logger.info("Starting training process")
        try:
            _, tuning_metrics, held_out_metrics = train(cfg, train_pyd, tuning_pyd, held_out_pyd)
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
    mp.set_start_method('spawn', force=True)
    
    # Use dill for serialization
    import cloudpickle
    mp.reductions.ForkingPickler.dumps = cloudpickle.dumps
    mp.reductions.ForkingPickler.loads = cloudpickle.loads

    main()