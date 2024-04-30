# Diabetes predictive modelling
Identification of Temporal Data Patterns Predictive of Adverse Outcomes in Patients with Diabetes

## Models for Prediction

### [Tab Transformer](https://github.com/lucidrains/tab-transformer-pytorch)
The Tab Transformer is a novel model for tabular data that applies the self-attention mechanism from Transformer models to learn complex relationships between categorical features. It encodes categorical variables into embeddings and uses multiple layers of self-attention to dynamically understand the context of each feature within a row.

### [FT Transformer](https://github.com/lucidrains/tab-transformer-pytorch?tab=readme-ov-file#ft-transformer)
The FT Transformer extends the concepts from the Tab Transformer by incorporating feature tokenization, allowing it to effectively handle both categorical and continuous features. It improves upon the Tab Transformer by using feature-wise transformations which lead to better performance on a range of tabular datasets.

Both models have shown to outperform traditional deep learning and machine learning approaches on complex tabular datasets, making them a powerful tool for tasks such as classification and regression in structured data.

## CutMix and MixUp Augmentation Techniques

#### CutMix
CutMix blends portions of two different inputs and their labels, essentially "cutting" and "pasting" parts of images, but it can be adapted for numerical data by mixing features. It's designed to enhance model robustness and performance, particularly in vision tasks, but its principles can be applied to numerical data for similar benefits. The original academic paper can be found [here](https://arxiv.org/abs/1905.04899).
<!-- For details on implementing CutMix in PyTorch, see [PyTorch's documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CutMix.html) and its [use cases](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html#sphx-glr-auto-examples-transforms-plot-cutmix-mixup-py).  -->

#### MixUp
MixUp operates by taking convex combinations of pairs of inputs and their labels. This method has shown to improve model generalization by encouraging the model to behave linearly in-between training examples, reducing memorization of training data. The foundational paper for MixUp is available [here](https://arxiv.org/abs/1710.09412).
<!-- For PyTorch implementation details, refer to [PyTorch's MixUp documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.MixUp.html).  -->

Both methods have been utilized in various models, including [SAINT](https://arxiv.org/abs/2106.01342), a model specifically designed for tabular data, which demonstrates their adaptability and effectiveness beyond conventional image data applications to numerical inputs. 

## Requirements

This code has been tested on Python 3.10 on Ubuntu 22.04.4 LTS.

Requires PyTorch 2.2.1 compiled for CUDA 12.1 and cuDNN 8.9.7 ([Instructions for Pytorch 2 and CUDA 11.8](https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba#file-cuda_11-8_installation_on_ubuntu_22-04)) ([CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)) ([cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)) ([cuDNN instructions](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/install-guide/index.html)). 

Note: it is recommended to install PyTorch in a python virtual environment (see Getting started).

## Hardware

NVIDIA Driver Version: 550.54.15

CUDA Version: 12.1

GPUs: GeForce RTX 2080 (2)

## Code Files in `src/`

- `model_utils.py`
	-  Set of utility functions and classes for training and validating machine learning models in PyTorch, including support for data loading, model training with techniques like MixUp and CutMix, model validation, and custom dataset handling.

- `data_loader.py`
	- Loads and merges data from .txt files. Randomly splits the data into training (70%), validation (20%), and test (10%) sets. Preprocesses datasets, converts datasets into PyTorch Tensors, and saves them to file.

- `event_stream.py`
	- Preprocess and generate "Event Stream" dataset. Make sure to set the appropriate file paths and configurations within the script. The script will generate the necessary data files, including the outcomes, diagnoses, procedures, and labs data.

- `build_task.py`
	- Define the specific task you want to perform. In this case, the task is predicting A1cGreaterThan7.

- `labelers.py`
	- Define a labeler class specific to your task. In this case, you can use the existing A1cGreaterThan7Labeler class.

- `pretrain.py`
	- Pre-trains a model from scratch. Utilizes the `pretrain_NA.yaml` config file to pre-train the model on the task of predicting A1cGreaterThan7 using the A1cGreaterThan7Labeler from labelers.py.

- `data_analysis.py`
	- Visualizes one-hot encoded feature sparsity and generates training dataset summary statistics.

- `train_tune.py`
	- Hyperparameter optimization for transfomer models using Ray Tune.

- `train.py`
	- Trains transformer model, supporting Tab Transformer and FT Transformer. Optional pretraining with CutMix and Mixup. 

- `attention.py`
	- Load a model, either a TabTransformer or an FTTransformer, and create HTML tables for attention maps.

- `visualize_attn.py`
	- Load a Transformer model and visualize HTML representations for the head view, model view, and neuron view using the BertViz package.

- `embeddings.py`
	- Loads a trained model, extracts embeddings for the validation dataset, and then applies the t-SNE algorithm to these embeddings.

- `test.py`
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
```

4. Clone project repo and install other dependencies from `requirements.txt` file:
```bash
$ git clone https://github.com/jvpoulos/diabetes_pred.git
$ pip3 install -r diabetes_pred/requirements.txt
```

5. Install git repo [TabTransformer](https://github.com/jvpoulos/TabTransformer), forked from [tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch):
```bash
$ pip3 install git+https://github.com/jvpoulos/TabTransformer.git
```
Additionally, follow instructions for installing [flash attention](https://github.com/Dao-AILab/flash-attention). Note: FlashAttention only supports Ampere GPUs or newer.

6. Clone forked version of git repo [EventStreamGPT](https://github.com/mmcdermott/EventStreamGPT), outside of project directory:
```bash
$ git clone https://github.com/jvpoulos/EventStreamGPT.git
touch EventStreamGPT/__init__.py
touch EventStreamGPT/EventStream/__init__.py
touch EventStreamGPT/EventStream/data/__init__.py
```

7. Install Dask for data_loader.py:

```bash
$ python3 -m pip install "dask[complete]" --upgrade
```
## Run the code

1. For static analyses, run (arguments: `--use_dask`)

```bash
$ cd diabetes_pred 
$ python3 src/data_loader.py
```

For temporal analyses, run instead:
```bash
$ export PYTHONPATH=$PYTHONPATH:../EventStreamGPT
$ python3 src/event_stream.py # create data files
$ python3 src/build_task.py # create the task-specific DataFrame (data/task_dfs/a1c_greater_than_7.parquet).
```

2. (Optional) Create plots and summary statistics for the training dataset (static analyses):

```bash
$ python3 src/data_analysis.py
``` 

For temporal analyses, run instead:
```bash
$ python3 src/visualize_describe.py
```

3. (Optional) Hyperparameter optimization for transfomer model. Arguments: `--model_type` ('TabTransformer', 'FTTransformer', or 'ResNet') `--epochs`.

```bash
$ python3 src/train_tune.py --model_type FTTransformer --epochs 25
```

For temporal analyses, run instead:
```bash
$ python3 ../EventStreamGPT/scripts/launch_from_scratch_supervised_wandb_hp_sweep.py # launch the Weights and Biases sweep and manage the sweep process
$ python3 src/hp_sweep.py #  perform the actual hyperparameter tuning by loading the dataset, creating the model, and training it
```

4. Train and evaluate transformer. Arguments: `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--batch_size` `--learning_rate` `--scheduler`  `--weight_decay` `--epochs` `--early_stopping_patience` `--use_cutmix`  `--cutmix_prob`  `--cutmix_alpha`  `--use_mixup` `--mixup_alpha` `--clipping` `use_batch_accumulation` `--max_norm` `--mixup_alpha` `--model_path`.

```bash
$ python3 src/train.py --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --use_batch_accumulation --clipping --max_norm 5 --batch_size 8 --epochs 200 --early_stopping_patience 10 --scheduler 'cosine'
```

or 
```bash
$ python3 src/train.py --model_type ResNet --dim 128 --depth 3 --dropout 0.2 --batch_size 8 --epochs 200 --early_stopping_patience 10 --use_batch_accumulation --clipping --max_norm 5 --scheduler 'cosine' --learning_rate 0.01 --normalization layernorm --use_mixup --use_cutmix --weight_decay 0.1 --d_hidden_factor 4
```
For temporal analyses:

```bash
$ python3 src/pretrain.py +config_name=pretrain_NA
```

or
```bash
$ python3 src/pretrain.py +config_name=pretrain_CI
```

5. (Optional) Extract attention weights from the last layer of the transformer and create attention map tables. Arguments: `--nproc_per_node` (required) `--dataset_type` `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--model_path` `--batch_size` `--pruning` `--quantization`:
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