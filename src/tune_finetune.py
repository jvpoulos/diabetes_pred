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
import numpy as np
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import sample

from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from EventStream.data.vocabulary import VocabularyConfig
from data_utils import CustomPytorchDataset

import tempfile
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureDetectionCallback(tune.Callback):
    def __init__(self, metric="val_auc_epoch", threshold=float('-inf'), grace_period=1):
        self.metric = metric
        self.threshold = threshold
        self.grace_period = grace_period
        self.trial_iterations = {}

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if self.metric not in result:
            return
        
        if trial not in self.trial_iterations:
            self.trial_iterations[trial] = 0
        self.trial_iterations[trial] += 1

        if self.trial_iterations[trial] > self.grace_period and result[self.metric] <= self.threshold:
            print(f"Stopping trial {trial} due to poor performance: {result[self.metric]} <= {self.threshold}")
            return True  # Stop the trial

    def on_trial_error(self, iteration, trials, trial, **info):
        print(f"Trial {trial} errored, stopping it.")
        return True  # Stop the trial
        
def resolve_tune_value(value):
    if isinstance(value, tune.search.sample.Categorical):
        return value.categories[0]  # Use the first category as a default
    elif isinstance(value, tune.search.sample.Float):
        return value.lower
    elif isinstance(value, tune.search.sample.Integer):
        return value.lower
    elif isinstance(value, bool):
        return value
    elif callable(value):  # Handle sample_from functions
        try:
            return value(None)  # Pass None as a dummy spec
        except:
            return None  # Return None if the function can't be evaluated
    else:
        return value

def get_data_dir(config):
    base_dir = Path("/home/jvp/diabetes_pred/data")
    if config.get("config", {}).get("use_labs", True):
        data_dir = base_dir / "labs"
    else:
        data_dir = base_dir
    
    logger.info(f"Data directory set to: {data_dir}")
    return data_dir

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'batch_size':
                value = int(value)
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

    @property
    def oov_index(self):
        return getattr(self.vocabulary_config, 'oov_index', self.vocab_size - 1)

    @property
    def use_labs(self):
        return getattr(self.config, 'use_labs', True)

    def update_data_config(self):
        pass

    def get_vocabulary_config_path(self):
        return getattr(self, 'vocabulary_config_path', None)

def create_datasets(cfg, device):
    data_config = cfg if isinstance(cfg, dict) else cfg.data_config
    dl_reps_dir = data_config.get('dl_reps_dir')
    
    if not dl_reps_dir:
        raise ValueError("dl_reps_dir not specified in the config")
    
    dl_reps_dir = Path(dl_reps_dir)
    
    logger.info(f"Using dl_reps_dir: {dl_reps_dir}")
    
    if not dl_reps_dir.exists():
        raise FileNotFoundError(f"dl_reps_dir not found: {dl_reps_dir}")
        
        # Check if the directory exists relative to the current working directory
        relative_path = Path.cwd() / dl_reps_dir.name
        if relative_path.exists():
            logger.info(f"Directory found at: {relative_path}")
            logger.info("Consider using an absolute path or updating the working directory.")
        else:
            logger.info("Directory not found relative to the current working directory.")
        
        raise FileNotFoundError(f"dl_reps_dir not found: {dl_reps_dir}")
    
    # List contents of dl_reps_dir
    logger.info(f"Contents of dl_reps_dir: {list(dl_reps_dir.iterdir())}")
    
    logger.info(f"DATA_DIR (absolute): {DATA_DIR.resolve()}")
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    
    subjects_df = pl.read_parquet(os.path.join(DATA_DIR, "subjects_df.parquet"))
    train_df = pl.read_parquet(os.path.join(DATA_DIR, "task_dfs/a1c_greater_than_7_train.parquet"))
    val_df = pl.read_parquet(os.path.join(DATA_DIR, "task_dfs/a1c_greater_than_7_val.parquet"))
    test_df = pl.read_parquet(os.path.join(DATA_DIR, "task_dfs/a1c_greater_than_7_test.parquet"))
    df_dia = pl.read_parquet(os.path.join(DATA_DIR, "df_dia.parquet"))
    df_prc = pl.read_parquet(os.path.join(DATA_DIR, "df_prc.parquet"))
    
    # Load df_labs if it exists
    df_labs = None
    df_labs_path = os.path.join(DATA_DIR, "df_labs.parquet")
    if os.path.exists(df_labs_path):
        df_labs = pl.read_parquet(df_labs_path)

    train_pyd = CustomPytorchDataset(data_config, split="train", dl_reps_dir=dl_reps_dir,
                                     subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                     task_df=train_df, device=device, max_seq_len=data_config.get('max_seq_len'))
    tuning_pyd = CustomPytorchDataset(data_config, split="tuning", dl_reps_dir=dl_reps_dir,
                                      subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                      task_df=val_df, device=device, max_seq_len=data_config.get('max_seq_len'))
    held_out_pyd = CustomPytorchDataset(data_config, split="held_out", dl_reps_dir=dl_reps_dir,
                                        subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, df_labs=df_labs,
                                        task_df=test_df, device=device, max_seq_len=data_config.get('max_seq_len'))
    return train_pyd, tuning_pyd, held_out_pyd

def train_function(config):
    global DATA_DIR
    
    # Set DATA_DIR based on the config
    DATA_DIR = Path(get_data_dir(config)).resolve()
    logger.info(f"DATA_DIR set to (absolute): {DATA_DIR}")

    # Ensure data_config is in the config
    if 'data_config' not in config:
        config['data_config'] = {}

    # Ensure dl_reps_dir is set in data_config
    if 'dl_reps_dir' not in config['data_config']:
        config['data_config']['dl_reps_dir'] = "/home/jvp/diabetes_pred/data/labs/DL_reps"
    
    logger.info(f"dl_reps_dir set to: {config['data_config']['dl_reps_dir']}")

    logger.info(f"Current working directory: {Path.cwd()}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    try:
        train_pyd, tuning_pyd, held_out_pyd = create_datasets(config['data_config'], device)
    except FileNotFoundError as e:
        logger.error(f"Error creating datasets: {str(e)}")
        logger.info("Config used:")
        logger.info(json.dumps(config, indent=2))
        raise

    # Set up wandb logger once
    wandb_logger_kwargs = {
        "project": "diabetes_sweep_labs",
        "entity": "jvpoulos",  # replace with your wandb entity
        "name": f"trial_{os.environ.get('TRIAL_ID', 'unknown')}",  # Use environment variable for trial ID
        "config": config,
        "reinit": True, 
        "settings": wandb.Settings(start_method="thread")
    }
    wandb_run = wandb.init(**wandb_logger_kwargs)
    logger.info(f"Wandb logger set up successfully with run ID: {wandb.run.id}")
    
    # Get maximum index from datasets
    max_index_train = max(train_pyd.get_max_index(), tuning_pyd.get_max_index(), held_out_pyd.get_max_index())

    vocab_size = max_index_train + 1

    logger.info(f"Maximum index in datasets: {max_index_train}")
    logger.info(f"Final vocab_size: {vocab_size}")

    # Ensure oov_index is set correctly
    oov_index = vocab_size - 1
    logger.info(f"OOV index: {oov_index}")

    # Create a minimal VocabularyConfig
    vocabulary_config = VocabularyConfig(
        vocab_sizes_by_measurement={'event_type': vocab_size},
        vocab_offsets_by_measurement={'event_type': 0},
        measurements_idxmap={'event_type': 0},
        measurements_per_generative_mode={'single_label_classification': ['event_type']},
        event_types_idxmap={'DIAGNOSIS': 0, 'LAB': 1, 'PROCEDURE': 2}
    )

    # Use the absolute path
    vocab_config_path = "/home/jvp/diabetes_pred/data/labs/vocabulary_config.json"
    logger.info(f"Using vocabulary config at: {vocab_config_path}")

    # Update config with the vocabulary config path
    config["vocabulary_config_path"] = vocab_config_path

    # Update config with the hyperparameters from Ray Tune
    config["config"].update(config.get("config", {}))
    config["optimization_config"].update(config.get("optimization_config", {}))

    # Ensure vocab_size is set correctly after the update
    config["config"]["vocab_size"] = vocab_size
    
    if config["config"]["use_layer_norm"]:
        if "layer_norm_epsilon" not in config["config"] or config["config"]["layer_norm_epsilon"] is None:
            config["config"]["layer_norm_epsilon"] = np.random.uniform(1e-6, 1e-4)
    else:
        config["config"]["layer_norm_epsilon"] = None

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

    # Ensure optimization_config is fully specified
    if "validation_batch_size" not in config["optimization_config"]:
        config["optimization_config"]["validation_batch_size"] = int(config["optimization_config"]["batch_size"])
    if "num_dataloader_workers" not in config["optimization_config"]:
        config["optimization_config"]["num_dataloader_workers"] = 0 
    if "weight_decay" not in config["optimization_config"]:
        config["optimization_config"]["weight_decay"] = config["optimization_config"].get("weight_decay", 0.01)
    
    # Ensure end_lr_frac_of_init_lr is between 0 and 1
    init_lr = config["optimization_config"]["init_lr"]
    end_lr = config["optimization_config"]["end_lr"]
    config["optimization_config"]["end_lr_frac_of_init_lr"] = min(max(end_lr / init_lr, 0), 1)

    # Resolve use_lr_scheduler and lr_scheduler_type
    use_lr_scheduler = config["optimization_config"].get("use_lr_scheduler", False)
    if isinstance(use_lr_scheduler, tune.search.sample.Categorical):
        use_lr_scheduler = use_lr_scheduler.categories[0]
    lr_scheduler_type = None
    if use_lr_scheduler:
        lr_scheduler_type = config["optimization_config"].get("lr_scheduler_type")
        if isinstance(lr_scheduler_type, tune.search.sample.Categorical):
            lr_scheduler_type = lr_scheduler_type.categories[0]

    # Resolve the values before calculation
    max_epochs = config["optimization_config"].get("max_epochs", 300)  # Default to 100 if not specified
    batch_size = resolve_tune_value(config["optimization_config"].get("batch_size", 32))

    # Now calculate max_training_steps with resolved values
    max_training_steps = int(max_epochs * len(train_pyd) // batch_size)

    # Update the config with the calculated value
    config["optimization_config"]["max_training_steps"] = max_training_steps

    config["optimization_config"]["use_lr_scheduler"] = use_lr_scheduler
    config["optimization_config"]["lr_scheduler_type"] = lr_scheduler_type

    # Ensure data_config is in the config
    if 'data_config' not in config:
        config['data_config'] = {}
    
    # Ensure dl_reps_dir is in data_config
    if 'dl_reps_dir' not in config['data_config']:
        config['data_config']['dl_reps_dir'] = str(DATA_DIR / "DL_reps")

    # Resolve Ray Tune search space objects in config
    for key in config["config"]:
        config["config"][key] = resolve_tune_value(config["config"][key])

    for key in config["optimization_config"]:
        config["optimization_config"][key] = resolve_tune_value(config["optimization_config"][key])

    # Log optimization configs
    wandb_run.log({
        **{f"optimization_config/{k}": v for k, v in config["optimization_config"].items()},
        **{f"config/{k}": v for k, v in config["config"].items()},
        **{f"data_config/{k}": v for k, v in config["data_config"].items()},
        **{f"trainer_config/{k}": v for k, v in config["trainer_config"].items()},
    })

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

    # Remove max_grad_norm from config["config"] if it exists
    max_grad_norm = config["config"].pop("max_grad_norm", None)

    # Create StructuredTransformerConfig
    transformer_config_args = {k: v for k, v in config["config"].items() if k != "layer_norm_epsilon"}
    if config["config"]["use_layer_norm"]:
        transformer_config_args["layer_norm_epsilon"] = config["config"]["layer_norm_epsilon"]

    transformer_config = StructuredTransformerConfig(
        vocab_size=vocab_size,
        max_grad_norm=max_grad_norm if max_grad_norm is not None else config["config"].get("max_grad_norm", 1.0),
        **transformer_config_args
    )
    print(f"Transformer config vocab_size: {transformer_config.vocab_size}")

    # Ensure vocab_size is set correctly in the transformer_config
    if transformer_config.vocab_size != vocab_size:
        print(f"Warning: transformer_config.vocab_size ({transformer_config.vocab_size}) does not match expected vocab_size ({vocab_size}). Updating...")
        transformer_config.vocab_size = vocab_size

    print(f"Final vocab_size before creating train_config: {vocab_size}")

    # Create a Config object to mimic the expected config structure
    train_config = Config(
        config=transformer_config,
        optimization_config=Config(**config["optimization_config"]),
        data_config=Config(**config["data_config"]),
        wandb_logger_kwargs=config["wandb_logger_kwargs"],
        seed=config.get("seed", 42),
        pretrained_weights_fp=config.get("pretrained_weights_fp", None),
        vocabulary_config=vocabulary_config,
        save_dir=Path(config.get("save_dir", "./experiments/finetune")),
        trainer_config=config.get("trainer_config", {}),
        vocab_size=vocab_size,
        update_data_config=lambda: None,
        do_debug_mode=False,
        vocabulary_config_path=config["vocabulary_config_path"],
    )

    # After creating train_config
    logger.info(f"Vocabulary config path: {train_config.vocabulary_config_path}")
    if not os.path.exists(train_config.vocabulary_config_path):
        raise FileNotFoundError(f"Vocabulary config file not found at: {train_config.vocabulary_config_path}")

    # Ensure vocab_size is set in the config attribute of train_config
    train_config.config.vocab_size = vocab_size
    train_config.vocabulary_config_path = os.path.abspath(vocab_config_path)
    logger.info(f"Setting vocabulary_config_path in train_config: {train_config.vocabulary_config_path}")
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Contents of temp directory: {os.listdir(os.path.dirname(train_config.vocabulary_config_path))}")

    logger.info(f"train_config.config.vocab_size: {train_config.config.vocab_size}")
    logger.info(f"train_config.vocab_size: {train_config.vocab_size}")

    # Ensure trainer_config is a dictionary
    if isinstance(train_config.trainer_config, Config):
        train_config.trainer_config = train_config.trainer_config.to_dict()

    # Run the training process
    logger.info(f"Final check - train_config.config.vocab_size: {train_config.config.vocab_size}")
    logger.info(f"Final check - train_config.vocab_size: {train_config.vocab_size}")
    logger.info(f"Final check - transformer_config.vocab_size: {transformer_config.vocab_size}")
    
    logger.info(f"train_config attributes: {vars(train_config)}")
    logger.info(f"vocabulary_config: {vocabulary_config}")
    logger.info(f"oov_index: {oov_index}")

    logger.info(f"Checking vocabulary config file at: {train_config.vocabulary_config_path}")
    if not os.path.exists(train_config.vocabulary_config_path):
        raise FileNotFoundError(f"Vocabulary config file not found at: {train_config.vocabulary_config_path}")

    logger.info(f"Contents of directory: {os.listdir(os.path.dirname(train_config.vocabulary_config_path))}")

    original_dir = os.getcwd()
    os.chdir("/home/jvp/diabetes_pred/data/labs")
    try:
        # Run the training process
        initialized = False
        try:
            _, tuning_metrics, _ = train(
                train_config, 
                train_pyd, 
                tuning_pyd, 
                held_out_pyd, 
                wandb_logger=wandb_run,
                vocabulary_config=vocabulary_config,
                oov_index=oov_index
            )
            initialized = True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise Exception("Trial failed to initialize") from e

        if initialized and tuning_metrics is not None and 'auroc' in tuning_metrics:
            tune.report(initialized=initialized, val_auc_epoch=tuning_metrics['auroc'])
        else:
            tune.report(initialized=initialized, val_auc_epoch=float('-inf'))  # Report a default value for failed trials
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        tune.report(initialized=False, val_auc_epoch=float('-inf'))
    finally:
        os.chdir(original_dir)

    # Report the initialization status
    ray.train.report({"initialized": initialized})

    # Report the results back to Ray Tune
    if tuning_metrics is not None and 'auroc' in tuning_metrics:
        ray.train.report({"val_auc_epoch": tuning_metrics['auroc']})
    else:
        print("Warning: tuning_metrics is None or does not contain 'auroc'")
        ray.train.report({"val_auc_epoch": 0.0})  # Report a default value

    # Before finishing the run, log final metrics
    if wandb.run is not None:
        if tuning_metrics is not None and 'auroc' in tuning_metrics:
            wandb.log({"val_auc_epoch": tuning_metrics['auroc']})
        wandb.finish()

@hydra.main(config_path=".", config_name="finetune_config", version_base=None)
def main(cfg):
    global DATA_DIR

    # Initialize Ray, ignoring reinit errors
    ray.init(ignore_reinit_error=True)

    # Check if cfg is already a dict (as it would be when called through Ray Tune)
    if isinstance(cfg, dict):
        config = cfg
    else:
        # If it's an OmegaConf object, convert it to a container
        config = OmegaConf.to_container(cfg, resolve=True)

    # Set DATA_DIR based on the config
    DATA_DIR = Path(get_data_dir(config))

    # Ensure 'data_config' key exists
    if 'data_config' not in config:
        config['data_config'] = {}

    # Set dl_reps_dir to the absolute path
    config['data_config']['dl_reps_dir'] = "/home/jvp/diabetes_pred/data/labs/DL_reps"

    logger.info(f"dl_reps_dir set to: {config['data_config']['dl_reps_dir']}")

    # Extract necessary configurations
    data_config = config.get("data_config", {})
    optimization_config = config.get("optimization_config", {})
    seed = config.get("seed", 42)
    pretrained_weights_fp = config.get("pretrained_weights_fp", None)
    save_dir = config.get("save_dir", "./experiments/finetune")
    trainer_config = config.get("trainer_config", {})

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_pyd, tuning_pyd, held_out_pyd = create_datasets(config['data_config'], device)  # Pass data_config directly

    # Ensure these are in optimization_config
    if "validation_batch_size" not in optimization_config:
        optimization_config["validation_batch_size"] = optimization_config["batch_size"]
    if "weight_decay" not in optimization_config:
        optimization_config["weight_decay"] = optimization_config.get("weight_decay", 0.01)

    def get_seq_window_size(spec):
        min_seq_len = resolve_tune_value(spec.config["data_config"]["min_seq_len"])
        max_seq_len = resolve_tune_value(spec.config["data_config"]["max_seq_len"])
        return np.random.randint(min_seq_len, max_seq_len + 1)

    search_space = {
        "config": {
            "use_layer_norm": tune.choice([True, False]),
            "use_batch_norm": tune.choice([True, False]),
            "do_use_sinusoidal": tune.choice([True, False]),
            "do_split_embeddings": tune.choice([True, False]),
            "use_gradient_checkpointing": tune.choice([False]),
            "categorical_embedding_dim": tune.choice([32, 64, 128]),
            "numerical_embedding_dim": tune.choice([32, 64, 128]),
            "categorical_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "numerical_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "static_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "dynamic_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "num_hidden_layers": tune.choice([4, 6, 8]),
            "head_dim": tune.choice([32, 64, 128]),
            "num_attention_heads": tune.choice([4, 8, 12]),
            "intermediate_dropout": tune.choice([0.1, 0.2, 0.3]),
            "attention_dropout": tune.choice([0.1, 0.2, 0.3]),
            "input_dropout": tune.choice([0.1, 0.2, 0.3]),
            "resid_dropout": tune.choice([0.1, 0.2, 0.3]),
            "max_grad_norm": tune.choice([1, 5, 10]),
            "intermediate_size": tune.choice([128, 256, 512]),
            "task_specific_params": {
                "pooling_method": tune.choice(["max", "mean"])
            },
            "layer_norm_epsilon": tune.sample_from(
                lambda spec: np.random.uniform(1e-6, 1e-4) if resolve_tune_value(spec.config["config"]["use_layer_norm"]) else None
            ),
            "seq_window_size": tune.sample_from(get_seq_window_size),
        },
        "optimization_config": {
            "init_lr": tune.loguniform(1e-4, 1e-01),
            "batch_size": tune.choice([128, 256, 512]),
            "use_grad_value_clipping": tune.choice([True, False]),
            "patience": tune.choice([1, 5, 10]),
            "use_lr_scheduler": tune.choice([True, False]),
            "lr_scheduler_type": tune.choice([None, "cosine", "linear", "one_cycle", "reduce_on_plateau"]),
            "weight_decay": tune.loguniform(1e-3, 1e-1),
            "lr_decay_power": tune.uniform(0.01, 0.5),
            "end_lr": tune.loguniform(1e-6, 1e-4),
            "max_epochs": tune.choice([50, 100, 150]),
        },
        "trainer_config": {
            "accumulate_grad_batches": tune.choice([1, 2, 4]),
        },
        "data_config": {
            **data_config,
            "min_seq_len": tune.randint(2, 50),
            "max_seq_len": tune.randint(100, 512),
        }
    }

    search_space["wandb_logger_kwargs"] = {
        "project": "diabetes_sweep_labs",
        "entity": "jvpoulos"  # replace with your actual entity
    }

    search_space["config"]["hidden_size"] = tune.sample_from(
        lambda spec: resolve_tune_value(spec.config["config"]["head_dim"]) * resolve_tune_value(spec.config["config"]["num_attention_heads"])
    )

    search_space["optimization_config"]["end_lr_frac_of_init_lr"] = tune.sample_from(
        lambda spec: resolve_tune_value(spec.config["optimization_config"]["end_lr"]) / resolve_tune_value(spec.config["optimization_config"]["init_lr"])
    )

    search_space["optimization_config"]["clip_grad_value"] = tune.sample_from(
        lambda spec: np.random.choice([0.5, 1.0, 5.0]) if resolve_tune_value(spec.config["optimization_config"]["use_grad_value_clipping"]) else None
    )

    def get_max_training_steps(spec):
        max_epochs = resolve_tune_value(spec.config["optimization_config"]["max_epochs"])
        batch_size = resolve_tune_value(spec.config["optimization_config"]["batch_size"])
        return int(max_epochs * len(train_pyd) // batch_size)

    search_space["optimization_config"]["max_training_steps"] = tune.sample_from(get_max_training_steps)

    # Add epochs to the search space
    search_space["optimization_config"]["max_epochs"] = config.get("max_epochs", 100)

    # Add max_grad_norm to the search space if it's not already there
    if "max_grad_norm" not in search_space["config"]:
        search_space["config"]["max_grad_norm"] = tune.uniform(0.1, 1.0)

    # Set use_cache based on use_gradient_checkpointing
    config["config"]["use_cache"] = not config["config"]["use_gradient_checkpointing"]

    search_space["optimization_config"]["max_training_steps"] = tune.sample_from(
        lambda spec: int(
            resolve_tune_value(spec.config["optimization_config"].get("max_epochs", 100)) * 
            len(train_pyd) // resolve_tune_value(spec.config["optimization_config"].get("batch_size", 32))
        )
    )
    # Get the current working directory
    cwd = os.getcwd()
    
    # Create an absolute path for ray_results
    storage_path = os.path.abspath(os.path.join(cwd, "ray_results"))

    # Configure the Ray Tune run
    analysis = tune.run(
        train_function,
        config=search_space,  
        num_samples=30,  # Number of trials
        scheduler=ASHAScheduler(
            time_attr='training_iteration',
            metric="val_auc_epoch",
            mode="max",
            max_t=config.get("max_epochs", 100),
            grace_period=1,
            reduction_factor=2
        ),
        progress_reporter=tune.CLIReporter(
            metric_columns=["initialized", "val_auc_epoch", "training_iteration"]
        ),
        name="diabetes_sweep_labs",
        storage_path=storage_path,  # Use the absolute path
        resources_per_trial={"cpu": 4, "gpu": 1},
        callbacks=[
            WandbLoggerCallback(project="diabetes_sweep_labs"),
            FailureDetectionCallback(metric="val_auc_epoch", threshold=float('-inf'), grace_period=1)
        ],
        stop={"training_iteration": config.get("max_epochs", 100)},
        raise_on_failed_trial=False  # This will allow Ray Tune to continue with other trials if one fails
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