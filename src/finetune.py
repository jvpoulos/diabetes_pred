import torch.multiprocessing as mp
import dill
import os
print(f"MASTER_ADDR={os.getenv('MASTER_ADDR')}")
print(f"MASTER_PORT={os.getenv('MASTER_PORT')}")

import argparse
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
from EventStream.data.vocabulary import VocabularyConfig
import polars as pl
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
if torch.cuda.is_available():
    # Set float32 matmul precision to 'medium' for better performance on Tensor Core enabled devices
    torch.set_float32_matmul_precision('medium')
import logging

from data_utils import CustomPytorchDataset, create_dataset

@hydra.main(version_base=None, config_path=".", config_name="finetune_config")
def main(cfg: DictConfig):
    # Convert DictConfig to a regular dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract use_labs from the config
    use_labs = cfg_dict.get('use_labs', False)
    
    # Now call the actual main function with both arguments
    _main(cfg_dict, use_labs)

def _main(cfg: dict, use_labs: bool = False):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info("Loading vocabulary config")
    with open("data/vocabulary_config.json", "r") as f:
        vocabulary_config_dict = json.load(f)
    vocabulary_config = VocabularyConfig.from_dict(vocabulary_config_dict)
    
    logger.info("Starting main function")
    
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

    logger.info("Loaded FinetuneConfig:")
    logger.info(str(cfg))

    print(f"FinetuneConfig created with use_static_features: {cfg.use_static_features}")
    
    if use_labs:
        DATA_DIR = Path.cwd() / "data/labs"
    else:
        DATA_DIR = Path.cwd() / "data"

    logger.info(f"Data directory: {DATA_DIR}")

    # Update dl_reps_dir using the setter method
    cfg.data_config.set_dl_reps_dir(str(DATA_DIR / "DL_reps"))
    logger.info(f"DL_reps directory: {cfg.data_config.dl_reps_dir}")

    logger.info("Checking parquet files.")

    train_parquet_files = list(Path("data/DL_reps").glob("train*.parquet"))
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

    # Load df_labs if it exists
    df_labs = None
    if (DATA_DIR / "df_labs.parquet").exists():
        df_labs = pl.read_parquet(DATA_DIR / "df_labs.parquet")
        logger.debug(f"Labs DataFrame shape: {df_labs.shape}")

    logger.info(f"dl_reps_dir: {cfg.data_config.dl_reps_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating CustomPytorchDataset instances")

    # Create datasets
    train_pyd = create_dataset(cfg.data_config, "train", cfg.data_config.dl_reps_dir, subjects_df, df_dia, df_prc, df_labs, train_df, max_seq_len=cfg.data_config.max_seq_len, min_seq_len=cfg.data_config.min_seq_len)
    tuning_pyd = create_dataset(cfg.data_config, "tuning", cfg.data_config.dl_reps_dir, subjects_df, df_dia, df_prc, df_labs, val_df, max_seq_len=cfg.data_config.max_seq_len, min_seq_len=cfg.data_config.min_seq_len)
    held_out_pyd = create_dataset(cfg.data_config, "held_out", cfg.data_config.dl_reps_dir, subjects_df, df_dia, df_prc, df_labs, test_df, max_seq_len=cfg.data_config.max_seq_len, min_seq_len=cfg.data_config.min_seq_len)

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
    oov_index = cfg.config.vocab_size  # Set oov_index to vocab_size
    logger.info(f"Set vocab_size to {cfg.config.vocab_size}")
    logger.info(f"Set oov_index to {oov_index}")

    logger.debug(f"Train dataset cached data: {[df.shape for df in train_pyd.cached_data_list] if hasattr(train_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
    logger.debug(f"Tuning dataset cached data: {[df.shape for df in tuning_pyd.cached_data_list] if hasattr(tuning_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
    logger.debug(f"Held-out dataset cached data: {[df.shape for df in held_out_pyd.cached_data_list] if hasattr(held_out_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")

    # Initialize WandbLogger
    wandb_kwargs = cfg.wandb_logger_kwargs.copy()
    project = wandb_kwargs.pop('project', 'default_project')
    name = wandb_kwargs.pop('name', 'default_run')
    
    # Remove unsupported parameters
    wandb_kwargs.pop('team', None)
    wandb_kwargs.pop('do_log_graph', None)
    
    wandb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir=cfg.save_dir,
        **wandb_kwargs
    )
    
    logger.info("Starting training process")

    _, tuning_metrics, held_out_metrics = train(cfg, train_pyd, tuning_pyd, held_out_pyd, vocabulary_config=vocabulary_config, oov_index=oov_index, wandb_logger=wandb_logger)

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Use dill for serialization
    import cloudpickle
    mp.reductions.ForkingPickler.dumps = cloudpickle.dumps
    mp.reductions.ForkingPickler.loads = cloudpickle.loads
    
    main()