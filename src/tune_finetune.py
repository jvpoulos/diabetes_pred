import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
import os
import logging
import torch
if torch.cuda.is_available():
    # Set float32 matmul precision to 'medium' for better performance on Tensor Core enabled devices
    torch.set_float32_matmul_precision('medium')
import polars as pl
from pathlib import Path
import wandb
from pytorch_lightning.loggers import WandbLogger
import ray
import json
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import sample

from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from data_utils import CustomPytorchDataset

def get_data_dir(config):
    base_dir = Path(os.path.expanduser("~/diabetes_pred/data"))
    if config.get("config", {}).get("use_labs", False):
        return base_dir / "labs"
    return base_dir

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

def create_datasets(cfg, device):
    subjects_df = pl.read_parquet(DATA_DIR / "subjects_df.parquet")
    train_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "task_dfs/a1c_greater_than_7_test.parquet")
    df_dia = pl.read_parquet(DATA_DIR / "df_dia.parquet")
    df_prc = pl.read_parquet(DATA_DIR / "df_prc.parquet")
    
    # Load df_labs if it exists
    df_labs = None
    if (DATA_DIR / "df_labs.parquet").exists():
        df_labs = pl.read_parquet(DATA_DIR / "df_labs.parquet")

    train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=cfg.data_config.dl_reps_dir,
                                     subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                     task_df=train_df, device=device)
    tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=cfg.data_config.dl_reps_dir,
                                      subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                      task_df=val_df, device=device)
    held_out_pyd = CustomPytorchDataset(cfg.data_config, split="held_out", dl_reps_dir=cfg.data_config.dl_reps_dir,
                                        subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                        task_df=test_df, device=device)
    return train_pyd, tuning_pyd, held_out_pyd

def train_function(config):
    global DATA_DIR  # We'll modify the global DATA_DIR
    
    # Setup wandb
    wandb_run = setup_wandb(config=config, project="diabetes_sweep")
    
    # Set DATA_DIR based on the config
    DATA_DIR = get_data_dir(config)
    
    # Update VOCABULARY_CONFIG_PATH
    global VOCABULARY_CONFIG_PATH
    VOCABULARY_CONFIG_PATH = DATA_DIR / "vocabulary_config.json"
    
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

    # Resolve use_lr_scheduler and lr_scheduler_type
    use_lr_scheduler = resolve_tune_value(config["optimization_config"].get("use_lr_scheduler", False))
    lr_scheduler_type = resolve_tune_value(config["optimization_config"].get("lr_scheduler_type", None)) if use_lr_scheduler else None

    config["optimization_config"]["use_lr_scheduler"] = use_lr_scheduler
    config["optimization_config"]["lr_scheduler_type"] = lr_scheduler_type

    # Log optimization configs
    wandb_run.log({
        "init_lr": config["optimization_config"]["init_lr"],
        "batch_size": config["optimization_config"]["batch_size"],
        "use_grad_value_clipping": config["optimization_config"]["use_grad_value_clipping"],
        "patience": config["optimization_config"]["patience"],
        "gradient_accumulation": config["optimization_config"]["gradient_accumulation"],
        "use_lr_scheduler": config["optimization_config"]["use_lr_scheduler"],
        "lr_scheduler_type": config["optimization_config"]["lr_scheduler_type"],
        "end_lr": config["optimization_config"]["end_lr"],
        "end_lr_frac_of_init_lr": config["optimization_config"]["end_lr_frac_of_init_lr"],
        "clip_grad_value": config["optimization_config"].get("clip_grad_value", None),
        "max_epochs": config["optimization_config"]["max_epochs"],
        "weight_decay": config["optimization_config"]["weight_decay"],
        "lr_decay_power": config["optimization_config"]["lr_decay_power"],
        "do_use_sinusoidal": config["config"]["do_use_sinusoidal"],
        "layer_norm_epsilon": config["config"].get("layer_norm_epsilon"),
        "min_seq_len": config["data_config"]["min_seq_len"],
        "max_seq_len": config["data_config"]["max_seq_len"],
        "seq_window_size": config["config"]["seq_window_size"],
    })

    def resolve_tune_value(value):
        if isinstance(value, sample.Categorical):
            return value.categories[0]
        elif isinstance(value, sample.Float):
            return value.lower
        elif isinstance(value, sample.Integer):
            return value.lower
        else:
            return value

    # Resolve Ray Tune search space objects in config
    for key in config["config"]:
        config["config"][key] = resolve_tune_value(config["config"][key])
    
    for key in config["optimization_config"]:
        config["optimization_config"][key] = resolve_tune_value(config["optimization_config"][key])

    # Ensure max_grad_norm is in the config and is a concrete value
    if "max_grad_norm" not in config["config"] or isinstance(config["config"]["max_grad_norm"], sample.Categorical):
        config["config"]["max_grad_norm"] = resolve_tune_value(config["config"].get("max_grad_norm", 1.0))

    # Remove vocab_size from config["config"] if it exists
    config["config"].pop("vocab_size", None)

    # Create StructuredTransformerConfig
    transformer_config = StructuredTransformerConfig(
        vocab_size=vocab_size,
        max_grad_norm=config["config"]["max_grad_norm"],
        **config["config"]
    )
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
    
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_DIR / 'vocabulary_config.json'):
        os.symlink(str(VOCABULARY_CONFIG_PATH), DATA_DIR / 'vocabulary_config.json')

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
    global DATA_DIR  # We'll modify the global DATA_DIR

    # Initialize Ray, ignoring reinit errors
    ray.init(ignore_reinit_error=True)

    # Check if cfg is already a dict (as it would be when called through Ray Tune)
    if isinstance(cfg, dict):
        config = cfg
    else:
        # If it's an OmegaConf object, convert it to a container
        config = OmegaConf.to_container(cfg, resolve=True)

    # Set DATA_DIR based on the config
    DATA_DIR = get_data_dir(config)

    # Update VOCABULARY_CONFIG_PATH
    global VOCABULARY_CONFIG_PATH
    VOCABULARY_CONFIG_PATH = DATA_DIR / "vocabulary_config.json"

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

    # Define the search space based on the YAML file and matching finetune.py
    search_space = {
        "config": {
            "use_layer_norm": tune.choice([True, False]),
            "use_batch_norm": tune.choice([True, False]),
            "do_use_learnable_sinusoidal_ATE": tune.choice([True, False]),
            "do_use_sinusoidal": tune.choice([True, False]),  # Added
            "do_split_embeddings": tune.choice([True, False]),
            "categorical_embedding_dim": tune.choice([16, 32, 64, 128, 256]),
            "numerical_embedding_dim": tune.choice([16, 32, 64, 128, 256]),
            "categorical_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "numerical_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "static_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "dynamic_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "num_hidden_layers": tune.choice([4, 6, 8]),
            "head_dim": tune.choice([32, 64, 128]),
            "num_attention_heads": tune.choice([4, 8, 12]),
            "intermediate_dropout": tune.choice([0.0, 0.1, 0.3]),
            "attention_dropout": tune.choice([0.0, 0.1, 0.3]),
            "input_dropout": tune.choice([0.0, 0.1, 0.3]),
            "resid_dropout": tune.choice([0.0, 0.1, 0.3]),
            "max_grad_norm": tune.choice([1, 5, 10, 15]),
            "intermediate_size": tune.choice([128, 256, 512, 1024]),
            "task_specific_params": {
                "pooling_method": tune.choice(["max", "mean"])
            },
            "layer_norm_epsilon": tune.sample_from(
                lambda spec: tune.loguniform(1e-7, 1e-5) if spec.config["config"]["use_layer_norm"] else None
            ),
        },
        "optimization_config": {
            "init_lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([256, 512, 1024, 2048]),
            "use_grad_value_clipping": tune.choice([True, False]),
            "patience": tune.choice([1, 5, 10]),
            "gradient_accumulation": tune.choice([1, 2, 4]),
            "use_lr_scheduler": tune.choice([True, False]),
            "weight_decay": tune.loguniform(1e-5, 1e-2),  # Added
            "lr_decay_power": tune.uniform(0, 1),  # Added
        },
        "data_config": {
            **data_config,
            "min_seq_len": tune.randint(2, 50),  # Added
            "max_seq_len": tune.randint(100, 750),  # Added
        }
    }

    # Ensure seq_window_size is within bounds of min_seq_len and max_seq_len
    search_space["config"]["seq_window_size"] = tune.sample_from(
        lambda spec: tune.randint(
            spec.config["data_config"]["min_seq_len"],
            spec.config["data_config"]["max_seq_len"]
        )
    )

    search_space["wandb_logger_kwargs"] = {
        "project": "diabetes_sweep",
        "entity": "jvpoulos"  # replace with your actual entity
    }

    # Add hidden_size based on head_dim and num_attention_heads
    search_space["config"]["hidden_size"] = tune.sample_from(
        lambda spec: spec.config.config.head_dim * spec.config.config.num_attention_heads
    )

    # Add end_lr and end_lr_frac_of_init_lr
    search_space["optimization_config"]["end_lr"] = tune.loguniform(1e-7, 1e-4)
    search_space["optimization_config"]["end_lr_frac_of_init_lr"] = tune.sample_from(
        lambda spec: spec.config.optimization_config.end_lr / spec.config.optimization_config.init_lr
    )

    # Add clip_grad_value only if use_grad_value_clipping is True
    search_space["optimization_config"]["clip_grad_value"] = tune.sample_from(
        lambda spec: tune.choice([0.5, 1.0, 5.0]) if spec.config.optimization_config.use_grad_value_clipping else None
    )

    search_space["optimization_config"]["use_lr_scheduler"] = tune.choice([True, False])
    search_space["optimization_config"]["lr_scheduler_type"] = tune.sample_from(
        lambda spec: tune.choice(["cosine", "linear", "one_cycle", "reduce_on_plateau"]) 
        if spec.config["optimization_config"]["use_lr_scheduler"] else None
    )

    # Add epochs to the search space
    search_space["optimization_config"]["max_epochs"] = config.get("max_epochs", 100)

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
        num_samples=20,  # Number of trials
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