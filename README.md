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

- `data_loader.py`
	- Loads and merges data from .txt files. Randomly splits the data into training (70%), validation (20%), and test (10%) sets. Preprocesses datasets, converts datasets into PyTorch Tensors, and saves them to file.

- `data_analysis.py`
	- Visualizes one-hot encoded feature sparsity and generates training dataset summary statistics.

- `train.py`
	- Trains transformer model, supporting Tab Transformer and FT Transformer. Optional pretraining with CutMix and Mixup. 

- `attention.py`
	- Load a model, either a TabTransformer or an FTTransformer, and visualize its attention maps using a validation dataset.

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

6. Install Dask for data_loader.py (optional):

```bash
$ python3 -m pip install "dask[dataframe]" --upgrade
$ python3.8 -m pip install "dask[distributed]" --upgrade
```
## Run the code

1. Load and merge data from .txt files. Split the dataset into training, validation, and test sets in a 70-20-10 ratio. Preprocess the data and convert the datasets to PyTorch Tensors:

```bash
$ python3.8 src/data_loader.py # need Python 3.8 to run
```

2. (Optional) Create plots and summary statistics for the training dataset:

```bash
$ python3 src/data_analysis.py
```

3. Train and evaluate transformer ('Transformer', 'TabTransformer', or 'FTTransformer'). Arguments: `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--outcome` (required) `--batch_size` `--learning_rate` `--epochs` `--early_stopping_patience` `--use_cutmix`  `--cutmix_prob`  `--cutmix_alpha`  `--use_mixup` `--mixup_alpha`  `--model_path` :

```bash
# Set CUDA_VISIBLE_DEVICES environment variable within the script (optional)
$ export CUDA_VISIBLE_DEVICES="0,1" 
$ export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
$ python3 src/train.py --model_type FTTransformer --dim 128 --attn_dropout 0.1 --outcome 'A1cGreaterThan7' --batch_size 16 --epochs 200 --early_stopping_patience 10 --use_cutmix --cutmix_prob 0.1 --cutmix_alpha 1 --run_id '8zthc8tx' --wandb_path 'model_weights/FTTransformer_dim128_dim3_heads8_fdr0.1_adr0.1_A1cGreaterThan7_bs16_lr0.001_ep42_esp15_cmp0.2_cml1.0_umfalse_ma0.2_uctrue_best.pth' 
```

4. (Optional) Plot losses and validation AUROC from saved training history. Arguments: `--file_path`:

```bash
$ python3 src/plot_losses.py 'losses/training_performance_model_type-FTTransformer_dim-128_attn_dropout-0_1_outcome-A1cGreaterThan7_batch_size-8_lr-0_1_ep-24_esp-10_cutmix_prob-0_3_cutmix_alpha-10_use_mixup-false_mixup_alpha-0_2_use_cutmix-true.pkl'
```

4. Extract attention weights from the last layer of the transformer and plot attention maps. Arguments: `--dataset_type` `--model_type` (required) `--dim` `--depth` `--heads` `--ff_dropout` `--attn_dropout` `--outcome` (required) `--model_path` `--batch_size`:

```bash
$ python3 src/attention.py --dataset_type 'train' --model_type FTTransformer --dim 128 --depth 6 --attn_dropout 0.1 --outcome 'A1cGreaterThan7' --model_path 'model_weights/FTTransformer_dim128_adr0.1_A1cGreaterThan7_bs8_lr0.1_ep13_esp10_cmp0.3_cml10_umfalse_ma0.2_uctrue_best.pth' --batch_size 4 
```

5. Extract learned embeddings from the last layer of the transformer, apply the t-SNE algorithm to these embeddings, and then plot them. Arguments:`--dataset_type` `--model_type` `--dim` (optional)  `--attn_dropout` (optional) `--outcome`  `--model_path` `--batch_size`:

```bash
$ python3 src/embeddings.py --dataset_type 'train' --model_type FTTransformer --outcome 'A1cGreaterThan7' --model_path 'model_weights/FTTransformer_A1cGreaterThan7_bs8_lr0.001_ep0_esp10_cmp0.3_cml10.0_umfalse_ma0.2_uctrue_best.pth' --batch_size 8 
```

6. Evaluate trained model on test set. Arguments: `--model_type` `--outcome`  `--model_path` `--batch_size`:

```bash
$ python3 src/test.py --model_type FTTransformer --outcome 'A1cGreaterThan7' --model_path 'model_weights/FTTransformer_A1cGreaterThan7_bs8_lr0.001_ep0_esp10_cmp0.3_cml10.0_umfalse_ma0.2_uctrue_best.pth' --batch_size 8
```