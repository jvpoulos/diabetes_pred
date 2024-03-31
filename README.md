# Diabetes predictive modelling
Identification of Temporal Data Patterns Predictive of Adverse Outcomes in Patients with Diabetes

## Models for Prediction

### [Tab Transformer](https://github.com/lucidrains/tab-transformer-pytorch)
The Tab Transformer is a novel model for tabular data that applies the self-attention mechanism from Transformer models to learn complex relationships between categorical features. It encodes categorical variables into embeddings and uses multiple layers of self-attention to dynamically understand the context of each feature within a row.

### [FT Transformer](https://github.com/lucidrains/tab-transformer-pytorch?tab=readme-ov-file#ft-transformer)
The FT Transformer extends the concepts from the Tab Transformer by incorporating feature tokenization, allowing it to effectively handle both categorical and continuous features. It improves upon the Tab Transformer by using feature-wise transformations which lead to better performance on a range of tabular datasets.

Both models have shown to outperform traditional deep learning and machine learning approaches on complex tabular datasets, making them a powerful tool for tasks such as classification and regression in structured data.

## Pretraining

### CutMix and MixUp Augmentation Techniques

#### CutMix
CutMix blends portions of two different inputs and their labels, essentially "cutting" and "pasting" parts of images, but it can be adapted for numerical data by mixing features. It's designed to enhance model robustness and performance, particularly in vision tasks, but its principles can be applied to numerical data for similar benefits. The original academic paper can be found [here](https://arxiv.org/abs/1905.04899).
<!-- For details on implementing CutMix in PyTorch, see [PyTorch's documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CutMix.html) and its [use cases](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html#sphx-glr-auto-examples-transforms-plot-cutmix-mixup-py).  -->

#### MixUp
MixUp operates by taking convex combinations of pairs of inputs and their labels. This method has shown to improve model generalization by encouraging the model to behave linearly in-between training examples, reducing memorization of training data. The foundational paper for MixUp is available [here](https://arxiv.org/abs/1710.09412).
<!-- For PyTorch implementation details, refer to [PyTorch's MixUp documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.MixUp.html).  -->

Both methods have been utilized in various models, including [SAINT](https://arxiv.org/abs/2106.01342), a model specifically designed for tabular data, which demonstrates their adaptability and effectiveness beyond conventional image data applications to numerical inputs. 

## Requirements

This code has been tested on Python 3.6.9 and Python 3.8.0 on Ubuntu 18.04.1.

Requires PyTorch 1.8.1 compiled for CUDA 11.2.

## Hardware

NVIDIA Driver Version: 460.91.03

GPU: GeForce RTX 2080

## Code Files in `src/`

- `model_utils.py`
	-  Set of utility functions and classes for training and validating machine learning models in PyTorch, including support for data loading, model training with techniques like MixUp and CutMix, model validation, and custom dataset handling.

- `data_loader.py`
	- Loads and merges data from .txt files. Randomly splits the data into training (70%), validation (20%), and test (10%) sets. Preprocesses datasets, converts datasets into PyTorch Tensors, and saves them to file.

- `data_analysis.py`
	- Visualizes one-hot encoded feature sparsity and generates training dataset summary statistics.

- `train_tune.py`
	- Hyperparameter optimization for transfomer models using Ray Tune.

- `train.py`
	- Trains transformer model, supporting Tab Transformer and FT Transformer. Optional pretraining with CutMix and Mixup. 

- `plot_losses.py`
	- Plot losses and validation AUROC from saved training history. 

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
$ cd <project-directory>
$ python3 -m pip install --user virtualenv #Install virtualenv if not installed in your system
$ python3 -m virtualenv env #Create virtualenv for your project
$ source env/bin/activate #Activate virtualenv for linux/MacOS
```

3. Install PyTorch via pip by running following command:
```bash
$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install other dependencies from `requirements.txt` file:
```bash
$ pip3 install -r requirements.txt
```

5. Install git repo [TabTransformer](https://github.com/jvpoulos/TabTransformer), forked from [tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch):
```bash
$ pip3 install git+https://github.com/jvpoulos/TabTransformer.git
```

6. Clone git repo [EventStreamGPT](https://github.com/jvpoulos/EventStreamGPT), forked from https://github.com/mmcdermott/EventStreamGPT outside of project directory:
```bash
$ git clone ../https://github.com/jvpoulos/EventStreamGPT.git
touch ../EventStreamGPT/__init__.py
touch ../EventStreamGPT/EventStream/__init__.py
touch ../EventStreamGPT/EventStream/data/__init__.py
```

7. Install Dask for data_loader.py (optional):

```bash
$ python3.8 -m pip install "dask[dataframe]" --upgrade
$ python3.8 -m pip install "dask[distributed]" --upgrade
```
## Run the code

1. Load and merge data from .txt files. Split the dataset into training, validation, and test sets in a 70-20-10 ratio. Preprocess the data and convert the datasets to PyTorch Tensors:

```bash
$ python3.8 src/data_loader.py # need Python 3.8 to run
```

For temporal analyses, run instead:
```bash
$ export PYTHONPATH=$PYTHONPATH:/home/jvp/EventStreamGPT
$ python3.8 src/event_stream.py
```

2. (Optional) Create plots and summary statistics for the training dataset:

```bash
$ python3 src/data_analysis.py
``` 
3. (Optional) Hyperparameter optimization for transfomer model. Arguments: `--model_type` ('Transformer', 'TabTransformer', or 'FTTransformer') `--epochs`.

```bash
$ export CUDA_VISIBLE_DEVICES="0,1" 
$ export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
$ python3 src/train_tune.py --model_type FTTransformer --epochs 20
```
or

```bash
$ python3 src/train_tune.py --model_type Transformer --epochs 20
```
3. Train and evaluate transformer. Arguments: `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--outcome` (required) `--batch_size` `--learning_rate` `--epochs` `--early_stopping_patience` `--use_cutmix`  `--cutmix_prob`  `--cutmix_alpha`  `--use_mixup` `--mixup_alpha`  `--model_path`.

```bash
$ python3 src/train.py --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0 --attn_dropout 0 --outcome 'A1cGreaterThan7' --batch_size 8 --epochs 200 --early_stopping_patience 10
```

or 
```bash
$ python3 src/train.py --model_type Transformer --dim 128 --depth 3 --heads 8 --dropout 0.0 --outcome 'A1cGreaterThan7' --batch_size 8 --epochs 200 --disable_early_stopping
```

4. (Optional) Plot losses and validation AUROC from saved training history. Arguments: `--file_path`:

```bash
$ python3 src/plot_losses.py 'losses/training_performance_model_type-FTTransformer_dim-128_attn_dropout-0_1_outcome-A1cGreaterThan7_batch_size-8_lr-0_1_ep-24_esp-10_cutmix_prob-0_3_cutmix_alpha-10_use_mixup-false_mixup_alpha-0_2_use_cutmix-true.pkl'
```

4. (Optional) Extract attention weights from the last layer of the transformer and create attention map tables. Arguments: `--nproc_per_node` (required) `--dataset_type` `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--outcome` (required) `--model_path` `--batch_size`:
```bash
$ python3 src/attention.py --dataset_type 'train' --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0.0 --attn_dropout 0.0 --outcome 'A1cGreaterThan7' --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_el6_ffdim2048_dr0.1_A1cGreaterThan7_bs16_lr0.001_ep17_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_ucfalse_best.pth' --batch_size 2
```

5. (Optional) Generate HTML representations for the head view, model view, and neuron view using the BertViz package (model_type = Transformer only).
```bash
$ python3 src/visualize_attn.py --dataset_type 'train' --model_type Transformer --dim 128 --depth 3 --heads 16 --ff_dropout 0.0 --attn_dropout 0.0 --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_el6_ffdim2048_dr0.1_A1cGreaterThan7_bs16_lr0.001_ep17_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_ucfalse_best.pth' --batch_size 2
```

5. (Optional) Extract learned embeddings from the last layer of the transformer, apply the t-SNE algorithm to these embeddings, and then plot them. Arguments:`--dataset_type` `--model_type` `--dim` (optional)  `--attn_dropout` (optional) `--outcome`  `--model_path` `--batch_size`:

```bash
$ python3 src/embeddings.py --dataset_type train --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0.0 --attn_dropout 0.0 --outcome A1cGreaterThan7 --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_el6_ffdim2048_dr0.1_A1cGreaterThan7_bs16_lr0.001_ep17_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_ucfalse_best.pth' --batch_size 2 -n 1 -g 2 -nr 0
```

6. Evaluate trained model on test set. Arguments: `--model_type` `--outcome`  `--model_path` `--batch_size`:

```bash
$ python3 src/test.py --model_type FTTransformer --dim 128 --depth 3 --heads 16 --ff_dropout 0.0 --attn_dropout 0.0 --outcome 'A1cGreaterThan7' --model_path 'model_weights/FTTransformer_dim128_dep3_heads16_fdr0.0_adr0.0_el6_ffdim2048_dr0.1_A1cGreaterThan7_bs16_lr0.001_ep17_esFalse_esp10_rs42_cmp0.3_cml10_umfalse_ma0.2_ucfalse_best.pth' --batch_size 2
```