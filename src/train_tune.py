import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
from sklearn.metrics import roc_auc_score
import os
import copy
import random
import numpy as np
from tqdm import tqdm
import pickle
from model_utils import load_model_weights, extract_epoch, load_performance_history, apply_cutmix_numerical, apply_mixup_numerical, CustomDataset, worker_init_fn, plot_auroc, plot_losses, TransformerWithInputProjection, train_model, validate_model
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis
from ray.tune import with_resources
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
import json
import functools

def custom_search_space(config):
    use_mixup = config.get("use_mixup", False)
    use_cutmix = config.get("use_cutmix", False)

    if use_mixup and use_cutmix:
        # If both are true, set use_cutmix to False
        use_cutmix = False

    return {
        "use_mixup": use_mixup,
        "use_cutmix": use_cutmix,
        "mixup_alpha": tune.sample_from(lambda spec: tune.choice([0.2, 1, 10]) if use_mixup else 1),
        "cutmix_alpha": tune.sample_from(lambda spec: tune.choice([0.2, 1, 10]) if use_cutmix else 1),
        "cutmix_prob": tune.sample_from(lambda spec: tune.choice([0.1, 0.2, 0.3]) if use_cutmix else 0),
    }

def hyperparameter_optimization(model_type, epochs):
    search_space = {
        "heads": tune.choice([4, 8, 16]),
        "disable_early_stopping": tune.choice([True, False]),
        "early_stopping_patience": tune.sample_from(lambda spec: tune.choice([5, 10, 15]) if not spec.config.get("disable_early_stopping", False) else 0),
    }

    if model_type in ['TabTransformer', 'FTTransformer']:
        search_space.update({
            "dim": tune.choice([32, 128, 192]),
            "depth": tune.choice([3, 6, 12]),
            "attn_dropout": tune.choice([0.0, 0.1, 0.2]),
            "ff_dropout": tune.choice([0.0, 0.1, 0.2]),
        })
    elif model_type == 'Transformer':
        search_space.update({
            "dim": tune.choice([256, 512, 1024]),
            "num_encoder_layers": tune.choice([2, 4, 6]),
            "dim_feedforward": tune.choice([1024, 2048, 4096]),
            "dropout": tune.choice([0.0, 0.1, 0.2]),
        })

    # Update the search space with the custom search space function
    search_space.update(custom_search_space({}))  # Pass an empty dictionary as the config argument

    scheduler = ASHAScheduler(
        metric="val_auroc",
        mode="max",
        max_t=100,
        grace_period=1,
        reduction_factor=4, # only 25% of all trials are kept each time they are reduced
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(functools.partial(tune_model, model_type=model_type, epochs=epochs)),
            resources={"cpu": 8, "gpu": 2}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,  # Use ASHAScheduler for early stopping and resource allocation
            num_samples=10,  # Explore 10 different parameter configurations
        ),
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

    return results

def tune_model(config, model_type, epochs):
    dim = config["dim"] 
    heads = config["heads"]
    use_mixup = config["use_mixup"]
    use_cutmix = config["use_cutmix"]
    mixup_alpha = config["mixup_alpha"]
    cutmix_alpha = config["cutmix_alpha"]
    cutmix_prob = config["cutmix_prob"]
    disable_early_stopping = config["disable_early_stopping"]
    early_stopping_patience = config["early_stopping_patience"].sample() if isinstance(config["early_stopping_patience"], tune.search.sample.Domain) else config["early_stopping_patience"]

    if model_type in ['TabTransformer', 'FTTransformer']:
        depth = config["depth"]
        attn_dropout = config["attn_dropout"]
        ff_dropout = config["ff_dropout"]
        if dim>128 or heads>8 or depth>6:
            batch_size = 8
        elif dim==128 and heads<16 and depth<12:
            batch_size = 16
        else:
            batch_size = 32
    elif model_type == 'Transformer':
        num_encoder_layers = config["num_encoder_layers"]
        dim_feedforward = config["dim_feedforward"]
        dropout = config["dropout"]
        batch_size = 32

    # Provide the absolute path to the train_dataset.pt file
    train_dataset_path = '/home/jvp/diabetes_pred/train_dataset.pt'
    validation_dataset_path = '/home/jvp/diabetes_pred/validation_dataset.pt'

    column_names_path = '/home/jvp/diabetes_pred/column_names.json'
    binary_feature_indices_path = '/home/jvp/diabetes_pred/binary_feature_indices.json'
    numerical_feature_indices_path = '/home/jvp/diabetes_pred/numerical_feature_indices.json'

    excluded_columns = ["A1cGreaterThan7", "A1cLessThan7",  "studyID"]
    
    with open(column_names_path, 'r') as file:
        column_names = json.load(file)

    with open(binary_feature_indices_path, 'r') as file:
        numerical_feature_indices = json.load(file)

    with open(numerical_feature_indices_path, 'r') as file:
        binary_feature_indices = json.load(file)

    # Load datasets
    train_dataset = torch.load(train_dataset_path)
    validation_dataset = torch.load(validation_dataset_path)

    # Assuming dataset is a TensorDataset containing a single tensor with both features and labels
    dataset_tensor = train_dataset.tensors[0]  # This gets the tensor from the dataset

    print(f"Original dataset tensor shape: {dataset_tensor.shape}")

    # Extracting indices for features and label
    excluded_columns_indices = [column_names.index(col) for col in excluded_columns]
    feature_indices = [i for i in range(dataset_tensor.size(1)) if i not in excluded_columns_indices]

    label_index = column_names.index('A1cGreaterThan7')
 
    # Selecting features and labels
    train_features = dataset_tensor[:, feature_indices]
    train_labels = dataset_tensor[:, label_index]

    # Repeat for validation dataset if necessary
    validation_dataset_tensor = validation_dataset.tensors[0]
    validation_features = validation_dataset_tensor[:, feature_indices]
    validation_labels = validation_dataset_tensor[:, label_index]

    train_data = CustomDataset(train_features, train_labels)
    validation_data = CustomDataset(validation_features, validation_labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    categories = [2] * len(binary_feature_indices)
    print("Categories:", len(categories))

    num_continuous = len(numerical_feature_indices)
    print("Continuous:", num_continuous)
    
    # Define the model
    if model_type == 'TabTransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=len(numerical_feature_indices),
            dim=dim,
            dim_out=1,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU()
        )
    elif model_type == 'FTTransformer':
        model = FTTransformer(
            categories=categories,
            num_continuous=len(numerical_feature_indices), 
            dim=dim,
            dim_out=1,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
    elif model_type == 'Transformer':
        input_size = len(binary_feature_indices) + len(numerical_feature_indices)
        model = TransformerWithInputProjection(
            input_size=input_size, 
            d_model=dim,
            nhead=heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters())

    best_val_auroc = float('-inf')
    patience_counter = 0

    for epoch in range(epochs):  
        train_args = {
            'model': model,
            'train_loader': train_loader,
            'criterion': criterion,
            'optimizer': optimizer,
            'device': device,
            'model_type': model_type,
            'use_cutmix': use_cutmix,
            'cutmix_prob': config["cutmix_prob"].sample() if isinstance(config["cutmix_prob"], tune.search.sample.Domain) else config["cutmix_prob"],
            'cutmix_alpha': config["cutmix_alpha"].sample() if isinstance(config["cutmix_alpha"], tune.search.sample.Domain) else config["cutmix_alpha"],
            'use_mixup': use_mixup,
            'mixup_alpha': config["mixup_alpha"].sample() if isinstance(config["mixup_alpha"], tune.search.sample.Domain) else config["mixup_alpha"],
            'binary_feature_indices': binary_feature_indices,
            'numerical_feature_indices': numerical_feature_indices
        }
        train_loss, train_auroc = train_model(**train_args)
        
        val_args = {
            'model': model,
            'validation_loader': validation_loader,
            'criterion': criterion,
            'device': device,
            'model_type': model_type,
            'binary_feature_indices': binary_feature_indices,
            'numerical_feature_indices': numerical_feature_indices
        }
        val_loss, val_auroc = validate_model(**val_args)
       
        tune.report(val_auroc=val_auroc)

        if not disable_early_stopping:
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc 
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization using Ray Tune')
    parser.add_argument('--model_type', type=str, required=True, choices=['TabTransformer', 'FTTransformer', 'Transformer'],
                        help='Type of the model to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    args = parser.parse_args()

    results = hyperparameter_optimization(args.model_type, args.epochs)