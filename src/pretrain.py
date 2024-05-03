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
from EventStream.data.dataset_config import DatasetConfig

from EventStream.data.dataset_config import DatasetConfig
from EventStream.data.types import DataModality, TemporalityType
from EventStream.data.measurement_config import MeasurementConfig
from EventStream.data.time_dependent_functor import TimeOfDayFunctor

from EventStream.data.preprocessing.standard_scaler import StandardScaler

torch.set_float32_matmul_precision("high")

DATA_DIR = Path("data")

@hydra.main(version_base=None, config_path=".", config_name="pretrain_config")
def main(cfg: PretrainConfig) -> None:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
        cfg.data_config.save_dir = Path(cfg.data_config.save_dir)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = Path(cfg.save_dir) / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(cfg_dict, cfg_fp)

    # Create an instance of the Dataset class

    dataset_config = DatasetConfig(
        measurement_configs={
            'InitialA1c': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
            ),
            'Female': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'Married': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'GovIns': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'English': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'AgeYears': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
            ),
            'SDI_score': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.UNIVARIATE_REGRESSION,
            ),
            'Veteran': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'CodeWithType': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
            # 'Result': MeasurementConfig(
            #     temporality=TemporalityType.DYNAMIC,
            #     modality=DataModality.UNIVARIATE_REGRESSION,
            # ),
        },
        min_valid_column_observations=0.01,  # Drop columns observed in less than 1% of events
        min_valid_vocab_element_observations=0.01,  # Drop vocabulary elements observed less than 1% of the time
        min_true_float_frequency=0.9,  # Treat values as floats if at least 90% of observations are float
        min_unique_numerical_observations=10,  # Treat values as categorical if fewer than 10 unique values
        outlier_detector_config=None,  # No outlier detection
        normalizer_config={'cls': 'standard_scaler'},
        save_dir=DATA_DIR,  # Save directory for the dataset
        agg_by_time_scale="1h",  # Aggregate events into 1-hour buckets
        )   

    dataset = Dataset(config=dataset_config)

    # Update the save_dir attribute
    dataset.config.save_dir = DATA_DIR

    # Split the dataset into train, validation, and test sets
    dataset.split(split_fracs=[0.7, 0.2, 0.1])

    # Preprocess the dataset
    dataset.preprocess()
    print("Finished preprocessing the dataset.")

    # Serialize the Dataset
    dataset_path = DATA_DIR / "serialized_dataset.pkl"
    dataset.save(save_path=dataset_path, do_overwrite=True)
    print("Saved the preprocessed dataset.")

    # Update the PretrainConfig with the serialized dataset path
    cfg.dataset_path = dataset_path

    # Create the PytorchDataset instances with the PytorchDatasetConfig
    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    train(cfg, train_pyd, tuning_pyd)

if __name__ == "__main__":
    main()