import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import yaml
from finetune import main as finetune_main
import argparse

def optimize_hyperparameters(config_path, epochs):
    # Load the base configuration
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Define the search space based on the YAML file and matching finetune.py
    search_space = {
        "config": {
            "use_layer_norm": tune.choice([True, False]),
            "use_batch_norm": tune.choice([True, False]),
            "do_use_learnable_sinusoidal_ATE": tune.choice([True, False]),
            "do_split_embeddings": tune.choice([True, False]),
            "categorical_embedding_dim": tune.choice([32, 64, 128]),
            "numerical_embedding_dim": tune.choice([32, 64, 128]),
            "categorical_embedding_weight": tune.choice([0.1, 0.3, 0.5]),
            "numerical_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "static_embedding_weight": tune.choice([0.3, 0.4, 0.6]),
            "dynamic_embedding_weight": tune.choice([0.3, 0.5, 0.7]),
            "num_hidden_layers": tune.choice([4, 6, 8]),
            "seq_window_size": tune.choice([168, 336, 504]),
            "head_dim": tune.choice([16, 32, 64]),
            "num_attention_heads": tune.choice([4, 8, 12]),
            "intermediate_dropout": tune.choice([0.1, 0.3]),
            "attention_dropout": tune.choice([0.1, 0.3]),
            "input_dropout": tune.choice([0.1, 0.3]),
            "resid_dropout": tune.choice([0.1, 0.3]),
            "max_grad_norm": tune.choice([1, 5, 10]),
            "intermediate_size": tune.choice([256, 512, 1024]),
            "task_specific_params": {
                "pooling_method": tune.choice(["max", "mean"])
            }
        },
        "optimization_config": {
            "init_lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([256, 512, 1024, 2048]),
            "use_grad_value_clipping": tune.choice([True, False]),
            "patience": tune.choice([5, 10]),
            "gradient_accumulation": tune.choice([1, 2, 4]),
            "use_lr_scheduler": tune.choice([True, False]),
        },
        "data_config": base_config["data_config"]
    }

    # Include data_config in the search space
    search_space["data_config"] = base_config["data_config"]

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

    # Add lr_scheduler_type only if use_lr_scheduler is True
    search_space["optimization_config"]["lr_scheduler_type"] = tune.sample_from(
        lambda spec: tune.choice(["cosine", "linear", "one_cycle", "reduce_on_plateau"]) if spec.config.optimization_config.use_lr_scheduler else None
    )

    # Add epochs to the search space
    search_space["optimization_config"]["max_epochs"] = epochs

    # Get the current working directory
    cwd = os.getcwd()
    
    # Create an absolute path for ray_results
    storage_path = os.path.abspath(os.path.join(cwd, "ray_results"))

    # Configure the Ray Tune run
    analysis = tune.run(
        finetune_main,
        config=search_space,
        num_samples=30,  # Number of trials
        scheduler=ASHAScheduler(
            time_attr='epoch',
            metric="val_auc_epoch",
            mode="max",
            max_t=epochs,
            grace_period=1,
            reduction_factor=2
        ),
        search_alg=OptunaSearch(
            metric="val_auc_epoch",
            mode="max"
        ),
        progress_reporter=tune.CLIReporter(
            metric_columns=["val_auc_epoch", "training_iteration"]
        ),
        name="diabetes_sweep",
        storage_path=storage_path,  # Use the absolute path
        resources_per_trial={"cpu": 4, "gpu": 0.33},  # Allocate 3 CPU and 0.33 GPU per trial
        callbacks=[WandbLoggerCallback(project="diabetes_sweep")]
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
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