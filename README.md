# Predictive modeling in Diabetes (work-in-progress)

## Models for Prediction

### Static analyses on tabular data:

#### [Tab Transformer](https://github.com/lucidrains/tab-transformer-pytorch)
The Tab Transformer is a novel model for tabular data that applies the self-attention mechanism from Transformer models to learn complex relationships between categorical features. It encodes categorical variables into embeddings and uses multiple layers of self-attention to dynamically understand the context of each feature within a row.

#### [FT Transformer](https://github.com/lucidrains/tab-transformer-pytorch?tab=readme-ov-file#ft-transformer)
The FT Transformer extends the concepts from the Tab Transformer by incorporating feature tokenization, allowing it to effectively handle both categorical and continuous features. It improves upon the Tab Transformer by using feature-wise transformations which lead to better performance on a range of tabular datasets.

### Temporal analyses on longitudinal data:

#### [Conditionally Independent Transformer](https://github.com/mmcdermott/EventStreamGPT)
The conditionally independent point process transformer is similar to a GPT Neo-X transformer. Measurements are aggregated together within an event to form event embeddings, which are then processed via an autoregressive transformer.

## Requirements

This code has been tested on:

- Ubuntu 22.04.4 LTS
- Python 3.10.12
- PyTorch 2.2.1 compiled for CUDA 12.1 and cuDNN 8.9.7 ([Instructions for Pytorch 2 and CUDA 11.8](https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba#file-cuda_11-8_installation_on_ubuntu_22-04)) ([CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)) ([cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)) ([cuDNN instructions](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/install-guide/index.html)) or PyTorch 2.3.1 compiled for CUDA 12.4 and cuDNN 9.2.0. 

Note: it is recommended to install PyTorch in a python virtual environment (see Getting started).

## Hardware

NVIDIA Driver Version: 550.54.15 or 550.90.07

CUDA Version: 12.1 or 12.4

GPUs: GeForce RTX 2080 (x2) or NVIDIA RTX 6000 Ada Generation (x3)

## Code Files in `src/`

- `model_utils.py`
	-  Set of utility functions and classes for training and validating machine learning models in PyTorch, including support for data loading, model training with techniques like MixUp and CutMix, model validation, and custom dataset handling.

- `data_loader.py` [static analyses]
	- Loads and merges data from .txt files. Randomly splits the data into training (70%), validation (20%), and test (10%) sets. Preprocesses datasets, converts datasets into PyTorch Tensors, and saves them to file.

- `event_stream.py` [temporal analyses]
	- Preprocess and generate "Event Stream" dataset. Make sure to set the appropriate file paths and configurations within the script. The script will generate the necessary data files, including the outcomes, diagnoses, procedures, and labs data.

- `build_task.py` [temporal analyses]
	- Defines the specific task --- this case, the task is binary classification on A1cGreaterThan7.

- `finetune.py` [temporal analyses]
	- Fine-tunes a transformer (from scratch) on the binary classification task. Utilizes the `pretrain_config.yaml` config file.

- `data_analysis.py` [static analyses]
	- Visualizes one-hot encoded feature sparsity and generates training dataset summary statistics.

- `train_tune.py` [static analyses]
	- Hyperparameter optimization for transfomer models using Ray Tune.

- `hp_sweep.py` [temporal analyses]
	- Perform hyperparameter tuning for the temporal analyses by loading the dataset, creating the model, and training it.

- `train.py` [static analyses]
	- Trains transformer model, supporting Tab Transformer and FT Transformer. Optional pretraining with CutMix and Mixup. 

- `attention.py` [static analyses]
	- Load a model, either a TabTransformer or an FTTransformer, and create HTML tables for attention maps.

- `visualize_attn.py` [static analyses]
	- Load a Transformer model and visualize HTML representations for the head view, model view, and neuron view using the BertViz package.

- `embeddings.py` [static analyses]
	- Loads a trained model, extracts embeddings for the validation dataset, and then applies the t-SNE algorithm to these embeddings.

- `test.py` [static analyses]
	- Loads the validated model from `train.py` and evaluates it on a test dataset.

## Getting started ([credit](https://gist.github.com/Ravi2712/47f070a6578153d3caee92bb67134963))

1. Check if `pip` is installed:
```bash
$ pip3 --version

#If `pip` is not installed, follow steps below:
$ cd ~
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py
```

2. Install virtual environment first & then activate:
```bash
$ python3 -m pip install --user virtualenv #Install virtualenv if not installed in your system
$ python3 -m virtualenv env10 #Create virtualenv for your project
$ source env10/bin/activate #Activate virtualenv for linux/MacOS
```

3. Install PyTorch via pip by running following command:
```bash
# CUDA 12.1
$ pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4: https://github.com/pytorch/pytorch#from-source
```

4. Clone project repo and install other dependencies from `requirements.txt` file:
```bash
$ git clone https://github.com/jvpoulos/diabetes_pred.git
$ pip3 install -r diabetes_pred/requirements.txt
```

5. (Optional, for static analyses) Install git repo [TabTransformer](https://github.com/jvpoulos/TabTransformer), forked from [tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch):
```bash
$ pip3 install git+https://github.com/jvpoulos/TabTransformer.git
```
Optionally,  follow instructions for installing [flash attention](https://github.com/Dao-AILab/flash-attention). Note: FlashAttention only supports Ampere GPUs or newer.

6. Clone forked version of git repo [EventStreamGPT](https://github.com/mmcdermott/EventStreamGPT), outside of project directory:
```bash
$ git clone https://github.com/jvpoulos/EventStreamGPT.git
touch EventStreamGPT/__init__.py
touch EventStreamGPT/EventStream/__init__.py
touch EventStreamGPT/EventStream/data/__init__.py
```

7. Install Dask (optional):

```bash
$ python3 -m pip install "dask[complete]" --upgrade
```
## Static analyses

1. Load data:

```bash
$ cd diabetes_pred 
$ python3 src/data_loader.py
```

2. (Optional) Create plots and summary statistics for the training dataset (static analyses):

```bash
$ python3 src/data_analysis.py
``` 

3. (Optional) Hyperparameter optimization for transfomer model. Arguments: `--model_type` ('TabTransformer', 'FTTransformer', or 'ResNet') `--epochs`.

```bash
$ python3 src/train_tune.py --model_type FTTransformer --epochs 25
```

4. Train and evaluate transformer. Arguments: `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--batch_size` `--learning_rate` `--scheduler`  `--weight_decay` `--epochs` `--early_stopping_patience` `--use_cutmix`  `--cutmix_prob`  `--cutmix_alpha`  `--use_mixup` `--mixup_alpha` `--clipping` `use_batch_accumulation` `--max_norm` `--mixup_alpha` `--model_path`.

```bash
$ python3 src/train.py --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --use_batch_accumulation --clipping --max_norm 5 --batch_size 8 --epochs 200 --early_stopping_patience 10 --scheduler 'cosine'
```

or 
```bash
$ python3 src/train.py --model_type ResNet --dim 128 --depth 3 --dropout 0.2 --batch_size 8 --epochs 200 --early_stopping_patience 10 --use_batch_accumulation --clipping --max_norm 5 --scheduler 'cosine' --learning_rate 0.01 --normalization layernorm --use_mixup --use_cutmix --weight_decay 0.1 --d_hidden_factor 4
```

5. (Optional) Etract attention weights from the last layer of the transformer and create attention map tables. Arguments: `--nproc_per_node` (required) `--dataset_type` `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--model_path` `--batch_size` `--pruning` `--quantization`:
```bash
$ python3 src/attention.py --dataset_type 'train' --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_bs8_lr0.001_wd0.01_ep21_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_mn5.0_ucfalse_cltrue_batrue_schtrue_best.pth' --batch_size 2 --pruning --quantization
```

6. (Optional) Extract learned embeddings from the last layer of the transformer, apply the t-SNE algorithm to these embeddings, and then plot them. Arguments:`--dataset_type` `--model_type` `--dim` (optional)  `--attn_dropout` (optional) `--model_path` `--batch_size` `--pruning` `--quantization`:

```bash
$ python3 src/embeddings.py --dataset_type 'train' --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_bs8_lr0.001_wd0.01_ep21_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_mn5.0_ucfalse_cltrue_batrue_schtrue_best.pth' --batch_size 2 -n 1 -g 2 -nr 0 --pruning --quantization
```

7. Evaluate trained model on test set. Arguments: `--dataset_type` `--model_type` `--model_path` `--batch_size` `--pruning` `--quantization`:

```bash
$ python3 src/test.py  --dataset_type 'test' --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_bs8_lr0.001_wd0.01_ep21_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_mn5.0_ucfalse_cltrue_batrue_schtrue_best.pth' --batch_size 1
```

or

```bash
$ python3 src/test.py  --dataset_type 'test' --model_type ResNet --dim 128 --depth 3 --dropout 0 --model_path 'model_weights/ResNet_dep3_dr0.2_bs8_lr0.01_wd0.1_ep16_esFalse_esp10_rs42_cmp0.3_cml10_umtrue_ma0.2_mn5.0_uctrue_cltrue_batrue_schtrue_best.pth' --batch_size 1
```

## Temporal analyses

1. Create data files (arguments: `--use_dask`, `--use_labs`):
```bash
$ export PYTHONPATH=$PYTHONPATH:../EventStreamGPT
$ python3 src/event_stream.py --use_labs
```

Create the task-specific data splits (data/task_dfs/a1c_greater_than_7_{split}.parquet):

```bash
$ python3 src/build_task.py 
```

2. (Optional) Create plots and summary statistics:

```bash
$ python3 src/visualize_describe.py
```

3. (Optional) Hyperparameter optimization for transfomer model:
```bash
$ python3 ../EventStreamGPT/scripts/launch_finetuning_wandb_hp_sweep.py # create the sweep. This will create a sweep on Weights and Biases with the specified hyperparameter ranges.
$ wandb agent <sweep_id> # Start the agent(s) to run the sweep. Replace <sweep_id> with the sweep ID obtained from the previous step.
```
The agent(s) will run the `hp_sweep.py` script with different hyperparameter configurations sampled from the ranges specified in the sweep configuration file. The training results and metrics will be logged to Weights and Biases for each run.

4. Pretrain the transformer:

```bash
$ export CUDA_LAUNCH_BLOCKING=1
$ export CUDA_VISIBLE_DEVICES=0,1
$ python3 src/pretrain.py +config_name=pretrain_config
```

Train the transformer from scratch:

```bash
$ python3 src/finetune.py
```