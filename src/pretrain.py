#!/usr/bin/env python
"""Pre-trains a model from scratch."""

try:
    import stackprinter
    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass

import copy
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
    train,
)
from EventStream.data.dataset_polars import Dataset
from EventStream.data.pytorch_dataset import PytorchDataset

torch.set_float32_matmul_precision("high")

DATA_DIR = Path("data")

@hydra.main(version_base=None, config_path=".", config_name="pretrain_config")
def main(cfg: PretrainConfig) -> None:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = Path(cfg.save_dir) / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(cfg_dict, cfg_fp)

    # Load the EventStream Dataset
    ESD = Dataset.load(DATA_DIR)

    # Create the PytorchDatasetConfig instance with the loaded dataset
    data_config = cfg.data_config
    data_config.dataset = ESD

    # Create the PytorchDataset instances with the PytorchDatasetConfig
    train_pyd = PytorchDataset(data_config, split="train")
    tuning_pyd = PytorchDataset(data_config, split="tuning")

    train(cfg, train_pyd, tuning_pyd)

if __name__ == "__main__":
    main()