from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

import torch.multiprocessing as mp
import dill
import os
import sys

import json
import ast
import base64

# Ensure WandB run is finished before starting a new one
import wandb
if wandb.run:
    wandb.finish()

import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig
from EventStream.transformer.lightning_modules.fine_tuning import train
from transformers import PretrainedConfig
from EventStream.transformer.config import StructuredTransformerConfig
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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from types import SimpleNamespace
from omegaconf import MISSING

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        return self.get(name)

    def to_dict(self):
        return dict(self)

class CustomESTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add attributes from StructuredTransformerConfig
        self.vocab_size = kwargs.get('vocab_size', 30522)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.intermediate_size = kwargs.get('intermediate_size', 3072)
        self.hidden_act = kwargs.get('hidden_act', "gelu")
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.attention_probs_dropout_prob = kwargs.get('attention_probs_dropout_prob', 0.1)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 512)
        self.type_vocab_size = kwargs.get('type_vocab_size', 2)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.position_embedding_type = kwargs.get('position_embedding_type', "absolute")
        self.use_cache = kwargs.get('use_cache', True)
        self._num_labels = kwargs.get('num_labels', 2)

    @property
    def num_labels(self):
        return self._num_labels

    @num_labels.setter
    def num_labels(self, value):
        self._num_labels = value

@dataclass
class FinetuneConfig:
    sweep: bool = False
    do_overwrite: bool = False
    seed: int = 42
    save_dir: str = MISSING
    dataset_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    trainer_config: Dict[str, Any] = field(default_factory=dict)
    experiment_dir: str = "./experiments"
    wandb_logger_kwargs: Dict[str, Any] = field(default_factory=dict)
    wandb_experiment_config_kwargs: Dict[str, Any] = field(default_factory=dict)
    do_final_validation_on_metrics: bool = False
    do_use_filesystem_sharing: bool = False
    data_config_path: Optional[str] = None
    optimization_config_path: Optional[str] = None
    pretrained_weights_fp: Optional[str] = None
    vocabulary_config_path: str = field(default_factory=lambda: str(Path("/home/jvp/diabetes_pred/data/vocabulary_config.json")))

cs = ConfigStore.instance()
cs.store(name="finetune_config", node=FinetuneConfig)

def evaluate_expression(expression, config):
    if expression is None:
        return None
    if isinstance(expression, (int, float, bool, str)):
        return expression
    try:
        if isinstance(config, dict):
            return eval(expression, {"__builtins__": None}, config)
        elif isinstance(config, FinetuneConfig):
            return eval(expression, {"__builtins__": None}, config.__dict__)
        elif hasattr(config, '__dict__'):
            return eval(expression, {"__builtins__": None}, config.__dict__)
        return eval(expression, {"__builtins__": None}, vars(config))
    except Exception as e:
        logger.warning(f"Failed to evaluate expression: {expression}. Error: {str(e)}")
        return None

def set_default_optimization_values(optimization_config, config):
    if isinstance(optimization_config, dict):
        if 'end_lr' not in optimization_config and 'end_lr_frac_of_init_lr' not in optimization_config:
            optimization_config['end_lr_frac_of_init_lr'] = 0.1
        
        if 'batch_size' not in optimization_config:
            optimization_config['batch_size'] = 1536  # Default batch size
        
        if 'validation_batch_size' not in optimization_config:
            optimization_config['validation_batch_size'] = optimization_config['batch_size']
    
    # Extract max_grad_norm
    max_grad_norm = optimization_config.pop('max_grad_norm', config.pop('max_grad_norm', 1.0))
    
    return optimization_config, config, max_grad_norm

def set_default_config_params(config, max_grad_norm):
    config_dict = config.to_dict()
    if 'task_specific_params' not in config_dict or config_dict['task_specific_params'] is None:
        config_dict['task_specific_params'] = {}
    
    if 'pooling_method' not in config_dict['task_specific_params']:
        config_dict['task_specific_params']['pooling_method'] = 'mean'  # Default to 'mean' pooling
    
    # Set default intermediate_dropout if not present
    if 'intermediate_dropout' not in config_dict:
        config_dict['intermediate_dropout'] = 0.1  # Default value, adjust as needed
    
    # Add max_grad_norm to the config
    config_dict['max_grad_norm'] = max_grad_norm
    
    # Update the StructuredTransformerConfig object
    for k, v in config_dict.items():
        setattr(config, k, v)
    
    return config_dict, config

def load_config():
    # Check if we're running under Hydra
    if HydraConfig.initialized():
        cfg = HydraConfig.get()
        
        # Check if sweep is enabled through Hydra's override
        is_sweep = cfg.overrides.get('sweep', 'false').lower() == 'true'
        
        if is_sweep:
            # If in sweep mode, load the encoded config
            try:
                encoded_config_path = os.path.join(get_original_cwd(), 'encoded_config.json')
                with open(encoded_config_path, 'r') as f:
                    encoded_config = f.read()
                # Decode the encoded configuration
                decoded_config = json.loads(base64.b64decode(encoded_config).decode())
                return OmegaConf.create(decoded_config), True
            except Exception as e:
                print(f"Error loading or decoding config: {str(e)}. Proceeding with default configuration.")
                return None, True
    else:
        # If not running under Hydra, use argparse for backward compatibility
        parser = argparse.ArgumentParser()
        parser.add_argument('--sweep', action='store_true', help='Run in sweep mode')
        args, unknown = parser.parse_known_args()
        
        if args.sweep:
            try:
                with open('encoded_config.json', 'r') as f:
                    encoded_config = f.read()
                # Decode the encoded configuration
                decoded_config = json.loads(base64.b64decode(encoded_config).decode())
                return OmegaConf.create(decoded_config), True
            except Exception as e:
                print(f"Error loading or decoding config: {str(e)}. Proceeding with default configuration.")
                return None, True
    
    return None, False

def clean_config(cfg):
    # Remove keys that are not expected by FinetuneConfig
    unexpected_keys = ['method', 'metric', 'parameters', 'name']
    for key in unexpected_keys:
        cfg.pop(key, None)
    # Ensure sweep is included in the cleaned config
    if 'sweep' not in cfg:
        cfg['sweep'] = False
    return cfg

def config_to_dict(config):
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    elif hasattr(config, '__dict__'):
        return {k: config_to_dict(v) for k, v in config.__dict__.items() if not k.startswith('_')}
    else:
        return config

@hydra.main(config_path=".", config_name="finetune_config")
def main(cfg: FinetuneConfig):
    print(OmegaConf.to_yaml(cfg))
    if 'sweep' not in cfg:
        OmegaConf.update(cfg, "sweep", False, merge=True)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting main function")

    # Set float32 matmul precision to 'medium' for better performance on Tensor Core enabled devices
    torch.set_float32_matmul_precision('medium')

    loaded_config, is_sweep = load_config()

    if loaded_config is not None:
        cfg = OmegaConf.merge(cfg, loaded_config)
    else:
        # If no sweep config was loaded, use the Hydra config
        cfg = OmegaConf.to_container(cfg, resolve=True)

    # Use the sweep parameter from the config
    is_sweep = cfg.get('sweep', False)

    # Ensure all necessary fields are present
    cfg['config'] = cfg.get('config', {})
    cfg['data_config'] = cfg.get('data_config', {})
    cfg['optimization_config'] = cfg.get('optimization_config', {})

    use_labs = cfg.get('use_labs', False)
    
    logger.info(f"Using labs: {use_labs}")
    logger.info(f"Hyperparameter sweep: {is_sweep}")

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    data_config_path = cfg.pop("data_config_path", None)
    optimization_config_path = cfg.pop("optimization_config_path", None)

    # Clean the configuration
    cfg = clean_config(cfg)

    if data_config_path:
        data_config_fp = Path(data_config_path)
        logger.info(f"Loading data_config from {data_config_fp}")
        reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
        cfg["data_config"] = reloaded_data_config
        cfg["config"]["problem_type"] = cfg["config"]["problem_type"]

    # Set default values for optimization config and extract max_grad_norm
    cfg["optimization_config"], cfg["config"], max_grad_norm = set_default_optimization_values(
        cfg.get("optimization_config", {}),
        cfg.get("config", {})
    )
    
    # Create StructuredTransformerConfig object
    transformer_config = StructuredTransformerConfig(**cfg["config"])
    
    # Set default values for optimization config and extract max_grad_norm
    cfg["optimization_config"], cfg["config"], max_grad_norm = set_default_optimization_values(
        cfg.get("optimization_config", {}),
        cfg.get("config", {})
    )
    
    # Set default config parameters including task-specific params, intermediate_dropout, and max_grad_norm
    config_dict, transformer_config = set_default_config_params(transformer_config, max_grad_norm)
    
    # Convert StructuredTransformerConfig back to dictionary
    cfg["config"] = config_to_dict(transformer_config)
    
    cfg_dict = {**cfg, 'data_config_path': data_config_path, 'optimization_config_path': optimization_config_path}
    cfg = FinetuneConfig(**cfg_dict)
    
    # Convert the entire cfg object to a dictionary for easier access
    cfg_dict = config_to_dict(cfg)
    
    # Log important configuration values
    logger.info(f"Batch size: {cfg_dict['optimization_config'].get('batch_size', 'Not set')}")
    logger.info(f"Validation batch size: {cfg_dict['optimization_config'].get('validation_batch_size', 'Not set')}")
    logger.info(f"Max grad norm: {cfg_dict['config'].get('max_grad_norm', 'Not set')}")
    logger.info(f"Intermediate dropout: {cfg_dict['config'].get('intermediate_dropout', 'Not set')}")

    if is_sweep:
        # Evaluate the expressions for problematic parameters
        cfg_dict['config']['hidden_size'] = evaluate_expression(cfg_dict['config'].get('hidden_size'), cfg_dict['config'])
        cfg_dict['data_config']['max_seq_len'] = evaluate_expression(cfg_dict['data_config'].get('max_seq_len'), cfg_dict['config'])
        cfg_dict['data_config']['min_seq_len'] = evaluate_expression(cfg_dict['data_config'].get('min_seq_len'), cfg_dict['config'])
        
        # Handle optimization_config
        cfg_dict['optimization_config']['end_lr_frac_of_init_lr'] = evaluate_expression(
            cfg_dict['optimization_config'].get('end_lr_frac_of_init_lr'), cfg_dict['optimization_config']
        )
        cfg_dict['optimization_config']['validation_batch_size'] = evaluate_expression(
            cfg_dict['optimization_config'].get('validation_batch_size'), cfg_dict['optimization_config']
        )
        cfg_dict['config']['max_grad_norm'] = evaluate_expression(
            cfg_dict['config'].get('max_grad_norm'), {'max_grad_norm': cfg_dict['config'].get('max_grad_norm')}
        )
        
        # Update the original cfg object with the evaluated values
        cfg.config.update(cfg_dict['config'])
        cfg.data_config.update(cfg_dict['data_config'])
        cfg.optimization_config.update(cfg_dict['optimization_config'])


    # Print current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

    original_cwd = get_original_cwd()
    logger.info(f"Original working directory: {original_cwd}")

    if use_labs:
        DATA_DIR = Path(original_cwd) / "data/labs"
    else:
        DATA_DIR = Path(original_cwd) / "data"

    logger.info(f"Data directory: {DATA_DIR}")

    # Update dl_reps_dir to use the absolute path
    cfg.data_config['dl_reps_dir'] = str(DATA_DIR / "DL_reps")
    logger.info(f"DL_reps directory: {cfg.data_config['dl_reps_dir']}")

    logger.info("Loaded FinetuneConfig:")
    logger.info(str(cfg))

    logger.info("Checking parquet files.")
    train_parquet_files = list(Path(cfg.data_config['dl_reps_dir']).glob("train*.parquet"))
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

    logger.info(f"dl_reps_dir: {cfg.data_config['dl_reps_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating CustomPytorchDataset instances")
    train_pyd = CustomPytorchDataset(cfg.data_config, split="train", dl_reps_dir=cfg.data_config['dl_reps_dir'], subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=train_df, device=device)
    tuning_pyd = CustomPytorchDataset(cfg.data_config, split="tuning", dl_reps_dir=cfg.data_config['dl_reps_dir'], subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=val_df, device=device)
    held_out_pyd = CustomPytorchDataset(cfg.data_config, split="held_out", dl_reps_dir=cfg.data_config['dl_reps_dir'], subjects_df=subjects_df, df_dia=df_dia, df_prc=df_prc, task_df=test_df, device=device)

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

    cfg.config['vocab_size'] = max_index + 1
    logger.info(f"Set vocab_size to {cfg.config['vocab_size']}")

    logger.debug(f"Train dataset cached data: {[df.shape for df in train_pyd.cached_data_list] if hasattr(train_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
    logger.debug(f"Tuning dataset cached data: {[df.shape for df in tuning_pyd.cached_data_list] if hasattr(tuning_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")
    logger.debug(f"Held-out dataset cached data: {[df.shape for df in held_out_pyd.cached_data_list] if hasattr(held_out_pyd, 'cached_data_list') else 'No cached_data_list attribute'}")

    logger.info("Starting training process")

    # Ensure all necessary attributes are present
    cfg.__dict__.setdefault('wandb_logger_kwargs', {})
    cfg.__dict__.setdefault('wandb_experiment_config_kwargs', {})
    cfg.__dict__.setdefault('pretrained_weights_fp', None) 

    # Set the vocabulary_config_path using an absolute path
    vocabulary_config_path = str(Path("/home/jvp/diabetes_pred/data/vocabulary_config.json"))
    cfg.vocabulary_config_path = vocabulary_config_path

    logger.info(f"Vocabulary config path: {vocabulary_config_path}")
    logger.info(f"Vocabulary config file exists: {os.path.exists(vocabulary_config_path)}")

    if not os.path.exists(vocabulary_config_path):
        raise FileNotFoundError(f"Vocabulary config file not found at {vocabulary_config_path}")

    # Debug logging for cfg.config
    logger.debug(f"cfg.config type: {type(cfg.config)}")
    logger.debug(f"cfg.config keys: {cfg.config.keys()}")
    logger.debug(f"cfg.config vocab_size: {cfg.config.get('vocab_size', 'Not found')}")

    # Convert FinetuneConfig to a dictionary
    cfg_dict = vars(cfg)

    # Create ConfigDict objects for nested configurations
    cfg_obj = ConfigDict()
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            cfg_obj[key] = ConfigDict(value)
        else:
            cfg_obj[key] = value

    # Convert save_dir to a Path object
    cfg_obj['save_dir'] = Path(cfg_obj['save_dir'])

    # Create and update custom_config
    custom_config = CustomESTConfig(**cfg_obj['config'])
    custom_config.vocab_size = cfg_obj['config']['vocab_size']
    custom_config.num_labels = 2  # or whatever number of labels you have
    cfg_obj['config'] = custom_config

    # Ensure wandb_logger_kwargs and wandb_experiment_config_kwargs are dictionaries
    cfg_obj['wandb_logger_kwargs'] = dict(cfg.wandb_logger_kwargs)
    cfg_obj['wandb_experiment_config_kwargs'] = dict(cfg.wandb_experiment_config_kwargs)

    # Create WandbLogger
    wandb_logger_kwargs = {k: v for k, v in cfg_obj['wandb_logger_kwargs'].items() if k not in ['do_log_graph', 'team']}
    wandb_logger = WandbLogger(**wandb_logger_kwargs, save_dir=str(cfg_obj['save_dir']))

    # Debug logging
    logger.debug(f"save_dir type: {type(cfg_obj['save_dir'])}")
    logger.debug(f"save_dir value: {cfg_obj['save_dir']}")

    # Update the train function call
    original_cwd = os.getcwd()
    os.chdir("/home/jvp/diabetes_pred")  # Change to the directory containing the 'data' folder
    try:
        wandb.init(config=cfg_obj.to_dict())
        _, tuning_metrics, held_out_metrics = train(cfg_obj, train_pyd, tuning_pyd, held_out_pyd, wandb_logger=wandb_logger)
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
    finally:
        os.chdir(original_cwd)

    logger.info("Main function completed")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Use dill for serialization
    import cloudpickle
    mp.reductions.ForkingPickler.dumps = cloudpickle.dumps
    mp.reductions.ForkingPickler.loads = cloudpickle.loads

    main()