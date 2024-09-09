import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import yaml
from tune_finetune import main as finetune_main
from tune_finetune import FailureDetectionCallback
import argparse

import time

def optimize_hyperparameters(config_path, epochs):
    # Load the base configuration
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Define the search space based on the YAML file and matching tune_finetune.py
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
                lambda config: tune.loguniform(1e-6, 1e-4).sample() if config["config"]["use_layer_norm"] else None
            ),
        },
        "optimization_config": {
            "init_lr": tune.loguniform(1e-4, 1e-01),
            "batch_size": tune.choice([128, 256, 512]),
            "use_grad_value_clipping": tune.choice([True, False]),
            "patience": tune.choice([1, 5, 10]),
            "use_lr_scheduler": tune.choice([True, False]),
            "weight_decay": tune.loguniform(1e-5, 1e-2),  
            "lr_decay_power": tune.uniform(0.01, 0.5),  
        },
        "trainer_config": {
            "accumulate_grad_batches": tune.choice([1, 2, 4]),
        },
        "data_config": {
            **base_config.get('data_config', {}),
            "min_seq_len": tune.randint(2, 50),
            "max_seq_len": tune.randint(100, 512),
        }
    }

    # Ensure seq_window_size is within bounds of min_seq_len and max_seq_len
    search_space["config"]["seq_window_size"] = tune.sample_from(
        lambda config: tune.randint(
            config["data_config"]["min_seq_len"],
            config["data_config"]["max_seq_len"]
        ).sample()
    )

    search_space["wandb_logger_kwargs"] = {
        "project": "diabetes_sweep_labs",
        "entity": "jvpoulos"  # replace with your actual entity
    }

    # Add hidden_size based on head_dim and num_attention_heads
    search_space["config"]["hidden_size"] = tune.sample_from(
        lambda config: config["config"]["head_dim"] * config["config"]["num_attention_heads"]
    ),

    # Add end_lr and end_lr_frac_of_init_lr
    search_space["optimization_config"]["end_lr"] = tune.loguniform(1e-6, 1e-4)
    search_space["optimization_config"]["end_lr_frac_of_init_lr"] = tune.sample_from(
        lambda config: config["optimization_config"]["end_lr"] / config["optimization_config"]["init_lr"]
    )

    # Add clip_grad_value only if use_grad_value_clipping is True
    search_space["optimization_config"]["clip_grad_value"] = tune.sample_from(
        lambda config: tune.choice([0.5, 1.0, 5.0]).sample() if config["optimization_config"]["use_grad_value_clipping"] else None
    )

    # Add lr_scheduler_type only if use_lr_scheduler is True
    search_space["optimization_config"]["lr_scheduler_type"] = tune.choice([None, "cosine", "linear", "one_cycle", "reduce_on_plateau"])

    # Add epochs to the search space
    search_space["optimization_config"]["max_epochs"] = epochs

    # Set use_cache based on use_gradient_checkpointing
    search_space["config"]["use_cache"] = tune.sample_from(
        lambda config: not config["config"]["use_gradient_checkpointing"]
    )

    # Remove wandb_logger_kwargs from the search space
    if 'wandb_logger_kwargs' in search_space:
        del search_space['wandb_logger_kwargs']

    # Get the current working directory
    cwd = os.getcwd()
    
    # Create an absolute path for ray_results
    storage_path = os.path.abspath(os.path.join(cwd, "ray_results"))

    # Configure the Ray Tune run
    analysis = tune.run(
        finetune_main,
        config=search_space,  
        num_samples=50,  # Number of trials
        scheduler=ASHAScheduler(
            time_attr='training_iteration',
            metric="val_auc_epoch",
            mode="max",
            max_t=epochs,
            grace_period=20,
            reduction_factor=2
        ),
        progress_reporter=tune.CLIReporter(
            metric_columns=["initialized", "val_auc_epoch", "training_iteration"]
        ),
        name="diabetes_sweep_labs",
        storage_path=storage_path,  # Use the absolute path
        resources_per_trial={"cpu": 4},
        callbacks=[
            WandbLoggerCallback(project="diabetes_sweep_labs", log_config=True),
            FailureDetectionCallback(metric="val_auc_epoch", threshold=float('-inf'), grace_period=20)
        ],
        stop={
            "training_iteration": epochs,
            "time_total_s": 48 * 60 * 60  # 2 days in seconds
        },
        raise_on_failed_trial=False,  # This will allow Ray Tune to continue with other trials if one fails
        time_budget_s=7 * 24 * 60 * 60,  # 7 days total time budget
        verbose=3
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--config', type=str, default='finetune_config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the config file (in the same directory as the script)
    config_path = os.path.join(current_dir, args.config)
    
    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    optimize_hyperparameters(config_path, args.epochs)