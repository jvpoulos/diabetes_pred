import os
import hydra
import wandb
from omegaconf import DictConfig
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from EventStream.data.dataset_polars import Dataset
from EventStream.data.pytorch_dataset import PytorchDataset

@hydra.main(
    version_base=None,
    config_path=".",
    config_name="finetune_config",
)
def main(cfg: FinetuneConfig):
    # Initialize wandb
    wandb.init(config=cfg, project="diabetes_pred_temporal")
    
    # Load the preprocessed dataset
    dataset_path = cfg.dataset_path
    dataset = Dataset.load(dataset_path)

    # Get the hyperparameters from wandb config
    hyperparameters = wandb.config

    # Update the model config with hyperparameters
    model_config = dict(cfg.config)
    model_config.update(hyperparameters)
    cfg.config = StructuredTransformerConfig(**model_config)

    # Create the PytorchDataset instances with the PytorchDatasetConfig
    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    # Fine-tune the model
    train(cfg, train_pyd, tuning_pyd)

if __name__ == "__main__":
    main()