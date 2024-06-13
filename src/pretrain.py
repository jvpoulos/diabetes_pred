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

from EventStream.data.data_embedding_layer import DataEmbeddingLayer

from EventStream.transformer.config import StructuredTransformerConfig

from data_utils import CustomPytorchDataset

torch.set_float32_matmul_precision("high")

DATA_DIR = Path("data")

import copy
import json
import polars as pl

def convert_config_to_python_object(config):
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            new_config[key] = convert_config_to_python_object(value)
        return new_config
    elif isinstance(config, list):
        new_config = []
        for item in config:
            new_config.append(convert_config_to_python_object(item))
        return new_config
    elif isinstance(config, str):
        return config
    else:
        return copy.deepcopy(config)

@hydra.main(version_base=None, config_path=".", config_name="pretrain_config")
def main(cfg: PretrainConfig) -> None:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
        cfg.data_config.save_dir = Path(cfg.data_config.save_dir)

    print(f"cfg.save_dir: {cfg.save_dir}")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = Path(cfg.save_dir) / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(cfg_dict, cfg_fp)

    dataset_config = DatasetConfig(
        measurement_configs={
            'A1cGreaterThan7': MeasurementConfig(
                temporality=TemporalityType.STATIC,
                modality=DataModality.SINGLE_LABEL_CLASSIFICATION,
            ),
             'event_type': MeasurementConfig(
                temporality=TemporalityType.DYNAMIC,
                modality=DataModality.MULTI_LABEL_CLASSIFICATION,
            ),       
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
        },
        min_valid_column_observations=None,
        min_valid_vocab_element_observations=None,
        min_true_float_frequency=0.9,
        min_unique_numerical_observations=10,
        outlier_detector_config=None,
        normalizer_config={'cls': 'standard_scaler'},
        save_dir=DATA_DIR,
        agg_by_time_scale=None,
    )   

    dataset = Dataset(config=dataset_config)

    dataset.config.save_dir = Path(dataset.config.save_dir)

    dataset.subjects_df = dataset.subjects_df.with_columns(
        pl.col('EMPI').cast(pl.UInt32).fill_null(0).alias('subject_id')
    )

    dataset.events_df = dataset.events_df.with_columns(
        pl.col('subject_id').cast(pl.UInt32).fill_null(0)
    )

    dataset.dynamic_measurements_df = dataset.dynamic_measurements_df.with_columns(
        pl.col('subject_id').cast(pl.UInt32).fill_null(0)
    )

    dataset.config.save_dir = DATA_DIR

    dataset.split(split_fracs=[0.7, 0.2, 0.1])
    print("Dataset split.")

    dataset.preprocess()
    print("Finished preprocessing the dataset.")

    print("Vocabulary Config:")
    print(dataset.vocabulary_config)

    print("Vocabulary sizes:")
    if hasattr(dataset, 'inferred_measurement_configs'):
        vocab_sizes_by_measurement = {
            measurement: len(config.vocabulary.vocabulary)
            for measurement, config in dataset.inferred_measurement_configs.items()
            if config.vocabulary is not None
        }
        print(vocab_sizes_by_measurement)
    else:
        print("Dataset does not have inferred_measurement_configs attribute.")

    max_vocab_size = max(vocab_sizes_by_measurement.values())
    print("Max vocab size:", max_vocab_size)

    model_config = dict(cfg.config)
    model_config["measurements_idxmap"] = dataset.vocabulary_config.measurements_idxmap
    model_config["vocab_offsets_by_measurement"] = dataset.vocabulary_config.vocab_offsets_by_measurement
    model_config["vocab_sizes_by_measurement"] = dataset.vocabulary_config.vocab_sizes_by_measurement
    model_config["problem_type"] = cfg.config.problem_type
    model_config["seq_attention_types"] = OmegaConf.to_container(model_config["seq_attention_types"], resolve=True)

    config = StructuredTransformerConfig(**model_config)

    print("StructuredTransformerConfig Vocabulary Info:")
    print("vocab_offsets_by_measurement:", config.vocab_offsets_by_measurement)
    print("vocab_sizes_by_measurement:", config.vocab_sizes_by_measurement)

    dataset_path = DATA_DIR / "E.pkl"
    dataset.save(save_path=dataset_path, do_overwrite=True)
    print("Saved the preprocessed dataset.")

    print("Caching deep learning representation...")
    dataset.cache_deep_learning_representation(do_overwrite=True)
    print("Deep learning representation cached.")

    cfg.dataset_path = dataset_path

    train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=Path("data/DL_reps"))
    tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=Path("data/DL_reps"))

    config.set_to_dataset(train_pyd)

    print("\nInspecting inferred_measurement_configs:")
    for measurement, config in dataset.inferred_measurement_configs.items():
        print(f"Measurement: {measurement}, Modality: {config.modality}, Temporality: {config.temporality}")

    train(cfg, train_pyd, tuning_pyd)

if __name__ == "__main__":
    main()