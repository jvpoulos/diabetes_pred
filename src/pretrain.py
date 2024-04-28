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

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = Path(cfg.save_dir) / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(cfg_dict, cfg_fp)

    # Create an instance of the Dataset class
    dataset_config = DatasetConfig(
        measurement_configs = {
            'A1cGreaterThan7': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
            'AgeYears': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,  # Changed from UNIVARIATE_REGRESSION
            ),
            'event_type': MeasurementConfig(
                temporality=TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
                functor=TimeOfDayFunctor(),
            ),
            'CodeWithType_diagnoses': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
            'CodeWithType_procedures': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),
            'Result': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTIVARIATE_REGRESSION,
                values_column='Result_value',
                modifiers={'numerical_only': True},
            ),
        },
        min_valid_column_observations=0.01,  # Drop columns observed in less than 1% of events
        min_valid_vocab_element_observations=0.01,  # Drop vocabulary elements observed less than 1% of the time
        min_true_float_frequency=0.9,  # Treat values as floats if at least 90% of observations are float
        min_unique_numerical_observations=10,  # Treat values as categorical if fewer than 10 unique values
        outlier_detector_config=None,  # No outlier detection
        normalizer_config={'cls': 'StandardScaler'},
        save_dir="./data",  # Save directory for the dataset
        agg_by_time_scale="1h",  # Aggregate events into 1-hour buckets
    )
    dataset = Dataset(config=dataset_config)

    # Load the EventStream Dataset
    dataset.load(DATA_DIR)

    # Serialize the Dataset
    dataset_path = Path("data/serialized_dataset.pkl")
    print("config object:", dataset.config)
    print("config attributes:", vars(dataset.config))
    dataset.save(save_path=dataset_path, do_overwrite=True)

    # Update the PretrainConfig with the serialized dataset path
    cfg.dataset_path = dataset_path

    # Create the PytorchDataset instances with the PytorchDatasetConfig
    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    train(cfg, train_pyd, tuning_pyd)

if __name__ == "__main__":
    main()