import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
import os
import logging
import torch
import polars as pl
from pathlib import Path
import wandb
from pytorch_lightning.loggers import WandbLogger
import ray
import json
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import ASHAScheduler

from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from data_utils import CustomPytorchDataset

# Move data loading outside of main
DATA_DIR = Path(os.path.expanduser("~/diabetes_pred/data"))
VOCABULARY_CONFIG_PATH = DATA_DIR / "vocabulary_config.json"

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.__dict__.items()}

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

def create_datasets(data_config, device):
    subjects_df = pl.read_parquet(DATA_DIR / "subjects_df.parquet")
    train_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_test.parquet")
    df_dia = pl.read_parquet(DATA_DIR / "df_dia.parquet")
    df_prc = pl.read_parquet(DATA_DIR / "df_prc.parquet")

    train_pyd = CustomPytorchDataset(data_config, split="train", dl_reps_dir=str(DATA_DIR / "DL_reps"),
                                     subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=train_df, device=device)
    tuning_pyd = CustomPytorchDataset(data_config, split="tuning", dl_reps_dir=str(DATA_DIR / "DL_reps"),
                                      subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=val_df, device=device)
    held_out_pyd = CustomPytorchDataset(data_config, split="held_out", dl_reps_dir=str(DATA_DIR / "DL_reps"),
                                        subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=test_df, device=device)
    return train_pyd, tuning_pyd, held_out_pyd

def train_function(config):
    # Setup wandb
    wandb_run = setup_wandb(config=config, project="diabetes_sweep")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_pyd, tuning_pyd, held_out_pyd = create_datasets(config["data_config"], device)

    # Load vocabulary config
    with open(VOCABULARY_CONFIG_PATH, 'r') as f:
        vocabulary_config = json.load(f)

    # Set vocab_size from vocabulary config
    vocab_size = vocabulary_config['vocab_sizes_by_measurement']['dynamic_indices'] + 2  # Add 2 instead of 1
    print(f"Adjusted vocab_size: {vocab_size}")

    max_index_train = max(train_pyd.get_max_index(), tuning_pyd.get_max_index(), held_out_pyd.get_max_index())
    print(f"Maximum index in datasets: {max_index_train}")

    vocab_size = max(vocab_size, max_index_train + 1)
    print(f"Final vocab_size after adjustment: {vocab_size}")

    # Update config with the hyperparameters from Ray Tune
    config["config"].update(config.get("config", {}))
    config["optimization_config"].update(config.get("optimization_config", {}))

    # Ensure vocab_size is set correctly after the update
    config["config"]["vocab_size"] = vocab_size

    # Update config with the hyperparameters from Ray Tune
    config["config"].update(config.get("config", {}))
    config["optimization_config"].update(config.get("optimization_config", {}))
    
    # Handle embedding dimensions based on do_split_embeddings
    if not config["config"].get("do_split_embeddings", False):
        config["config"]["categorical_embedding_dim"] = None
        config["config"]["numerical_embedding_dim"] = None

   # Ensure max_grad_norm is in the config
    if "max_grad_norm" not in config["config"]:
        config["config"]["max_grad_norm"] = config["optimization_config"].get("max_grad_norm", 1.0)

    # Ensure task_specific_params is a dictionary
    if "task_specific_params" not in config["config"] or config["config"]["task_specific_params"] is None:
        config["config"]["task_specific_params"] = {"pooling_method": "mean"}
    elif "pooling_method" not in config["config"]["task_specific_params"]:
        config["config"]["task_specific_params"]["pooling_method"] = "mean"

   # Ensure intermediate_dropout is present
    if "intermediate_dropout" not in config["config"]:
        config["config"]["intermediate_dropout"] = config["config"].get("dropout", 0.1)  # Default to 0.1 if not specified

    # Ensure max_training_steps and weight_decay are in optimization_config
    if "validation_batch_size" not in config["optimization_config"]:
        config["optimization_config"]["validation_batch_size"] = config["optimization_config"]["batch_size"]
    if "max_training_steps" not in config["optimization_config"]:
        config["optimization_config"]["max_training_steps"] = config["optimization_config"].get("max_epochs", 100) * len(train_pyd) // config["optimization_config"].get("batch_size", 32)
    if "weight_decay" not in config["optimization_config"]:
        config["optimization_config"]["weight_decay"] = config["optimization_config"].get("weight_decay", 0.01)
    
    # Ensure end_lr_frac_of_init_lr is between 0 and 1
    init_lr = config["optimization_config"]["init_lr"]
    end_lr = config["optimization_config"]["end_lr"]
    config["optimization_config"]["end_lr_frac_of_init_lr"] = min(max(end_lr / init_lr, 0), 1)

    # Remove vocab_size from config["config"] if it exists
    config["config"].pop("vocab_size", None)

    # Create StructuredTransformerConfig
    transformer_config = StructuredTransformerConfig(vocab_size=vocab_size, **config["config"])
    print(f"Transformer config vocab_size: {transformer_config.vocab_size}")

    # Ensure vocab_size is set correctly in the transformer_config
    if transformer_config.vocab_size != vocab_size:
        print(f"Warning: transformer_config.vocab_size ({transformer_config.vocab_size}) does not match expected vocab_size ({vocab_size}). Updating...")
        transformer_config.vocab_size = vocab_size

    print(f"Final vocab_size before creating train_config: {vocab_size}")

    # Create a Config object to mimic the expected cfg structure
    train_config = Config(
        config=transformer_config,
        optimization_config=Config(**config["optimization_config"]),
        data_config=Config(**config["data_config"]),
        wandb_logger_kwargs=config["wandb_logger_kwargs"],
        seed=config.get("seed", 42),
        pretrained_weights_fp=config.get("pretrained_weights_fp", None),
        vocabulary_config_path=str(VOCABULARY_CONFIG_PATH),
        save_dir=Path(config.get("save_dir", "./experiments/finetune")),
        trainer_config=config.get("trainer_config", {}),
        vocab_size=vocab_size
    )

    # Ensure vocab_size is set in the config attribute of train_config
    train_config.config.vocab_size = vocab_size
    
    print(f"train_config.config.vocab_size: {train_config.config.vocab_size}")
    print(f"train_config.vocab_size: {train_config.vocab_size}")
    
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/vocabulary_config.json'):
        os.symlink(str(VOCABULARY_CONFIG_PATH), 'data/vocabulary_config.json')

    # Ensure trainer_config is a dictionary
    if isinstance(train_config.trainer_config, Config):
        train_config.trainer_config = train_config.trainer_config.to_dict()

    # Run the training process
    print(f"Final check - train_config.config.vocab_size: {train_config.config.vocab_size}")
    print(f"Final check - train_config.vocab_size: {train_config.vocab_size}")
    print(f"Final check - transformer_config.vocab_size: {transformer_config.vocab_size}")
    _, tuning_metrics, _ = train(train_config, train_pyd, tuning_pyd, held_out_pyd, wandb_logger=wandb_run)
    
    # Report the results back to Ray Tune
    if tuning_metrics is not None and 'auroc' in tuning_metrics:
        ray.train.report({"val_auc_epoch": tuning_metrics['auroc']})
    else:
        print("Warning: tuning_metrics is None or does not contain 'auroc'")
        ray.train.report({"val_auc_epoch": 0.0})  # Report a default value

    # Make sure to finish the wandb run
    wandb_run.finish()

@hydra.main(config_path=".", config_name="finetune_config", version_base=None)
def main(cfg):
    # Initialize Ray, ignoring reinit errors
    ray.init(ignore_reinit_error=True)

    # Check if cfg is already a dict (as it would be when called through Ray Tune)
    if isinstance(cfg, dict):
        config = cfg
    else:
        # If it's an OmegaConf object, convert it to a container
        config = OmegaConf.to_container(cfg, resolve=True)

    # Ensure 'config' key exists
    if 'config' not in config:
        config['config'] = {}

    # Extract necessary configurations
    wandb_config = config
    wandb_project = config.get("wandb_logger_kwargs", {}).get("project", "diabetes_sweep")
    wandb_entity = config.get("wandb_logger_kwargs", {}).get("entity", None)
    data_config = config.get("data_config", {})
    optimization_config = config.get("optimization_config", {})
    wandb_logger_kwargs = config.get("wandb_logger_kwargs", {})
    seed = config.get("seed", 42)
    pretrained_weights_fp = config.get("pretrained_weights_fp", None)
    save_dir = config.get("save_dir", "./experiments/finetune")
    trainer_config = config.get("trainer_config", {})

    # Initialize wandb
    wandb.init(config=wandb_config,
               project=wandb_project,
               entity=wandb_entity)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_pyd, tuning_pyd, held_out_pyd = create_datasets(data_config, device)

    # Set up WandB logger
    wandb_logger = WandbLogger(**wandb_logger_kwargs)

    # Ensure max_training_steps and weight_decay are in optimization_config
    if "validation_batch_size" not in optimization_config:
        optimization_config["validation_batch_size"] = optimization_config["batch_size"]
    if "max_training_steps" not in optimization_config:
        optimization_config["max_training_steps"] = optimization_config.get("max_epochs", 100) * len(train_pyd) // optimization_config.get("batch_size", 32)
    if "weight_decay" not in optimization_config:
        optimization_config["weight_decay"] = optimization_config.get("weight_decay", 0.01)

    # Define the search space
    search_space = {
        "config": config,
        "optimization_config": optimization_config,
        "data_config": data_config,
        "wandb_logger_kwargs": wandb_logger_kwargs,
        "seed": seed,
        "pretrained_weights_fp": pretrained_weights_fp
    }

    # Add max_grad_norm to the search space if it's not already there
    if "max_grad_norm" not in search_space["config"]:
        search_space["config"]["max_grad_norm"] = tune.uniform(0.1, 1.0)

    # Get the current working directory
    cwd = os.getcwd()
    
    # Create an absolute path for ray_results
    storage_path = os.path.abspath(os.path.join(cwd, "ray_results"))

    # Configure the Ray Tune run
    analysis = tune.run(
        train_function,
        config=search_space,
        num_samples=30,  # Number of trials
        scheduler=ASHAScheduler(metric="val_auc_epoch", mode="max"),
        progress_reporter=tune.CLIReporter(metric_columns=["val_auc_epoch", "training_iteration"]),
        name="diabetes_sweep",
        storage_path=storage_path,  # Use the absolute path
        resources_per_trial={"cpu": 4, "gpu": 0.33},  # Allocate 3 CPU and 0.33 GPU per trial
        callbacks=[WandbLoggerCallback(project="diabetes_sweep")]
    )

    # Print the best config
    best_config = analysis.get_best_config(metric="val_auc_epoch", mode="max")
    print("Best hyperparameters found were: ", best_config)

    # You can also print the best trial's last result
    best_trial = analysis.get_best_trial(metric="val_auc_epoch", mode="max")
    print("Best trial last result:", best_trial.last_result)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Use dill for serialization
    import cloudpickle
    mp.reductions.ForkingPickler.dumps = cloudpickle.dumps
    mp.reductions.ForkingPickler.loads = cloudpickle.loads

    main()