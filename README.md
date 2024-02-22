# Diabetes predictive modelling
Identification of Temporal Data Patterns Predictive of Adverse Outcomes in Patients with Diabetes

## Models for Prediction

### [Tab Transformer](https://github.com/lucidrains/tab-transformer-pytorch)
The Tab Transformer is a novel model for tabular data that applies the self-attention mechanism from Transformer models to learn complex relationships between categorical features. It encodes categorical variables into embeddings and uses multiple layers of self-attention to dynamically understand the context of each feature within a row.

### [FT Transformer](https://github.com/lucidrains/tab-transformer-pytorch?tab=readme-ov-file#ft-transformer)
The FT Transformer extends the concepts from the Tab Transformer by incorporating feature tokenization, allowing it to effectively handle both categorical and continuous features. It improves upon the Tab Transformer by using feature-wise transformations which lead to better performance on a range of tabular datasets.

Both models have shown to outperform traditional deep learning and machine learning approaches on complex tabular datasets, making them a powerful tool for tasks such as classification and regression in structured data.

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
	- Loads ICD-9 and ICD-10 diagnostic codes, extracts code descriptions, and identifies infrequent categories. Visualizes one-hot encoded feature sparsity, and generates summary statistics for analysis.

- `train.py`
	- Trains transformer model, supporting Tab Transformer and FT Transformer.

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

4. Install dependencies for your project from `requirements.txt` file:
```bash
$ pip3 install -r requirements.txt
```

5. Install git repo [TabTransformer](https://github.com/jvpoulos/TabTransformer), forked from [tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch):
```bash
$ pip3 install git+https://github.com/jvpoulos/TabTransformer.git
```

6. Install Dask for processing and aggregating results more efficiently:

```bash
$ python3 -m pip install "dask[dataframe]" --upgrade
```
## Run the code

1. Load and merge data from .txt files. Split the dataset into training, validation, and test sets in a 70-20-10 ratio. Preprocess the data and convert the datasets to PyTorch Tensors. 

```bash
$ python3.8 src/data_loader.py # need Python 3.8 to run 
```

2. (Optional) Create plots and summary statistics for the training dataset:

```bash
$ python3 src/data_analysis.py
```

3. Train and evaluate transformer ('TabTransformer' or 'FTTransformer'). Arguments: `--model_type` `--batch_size` `--learning_rate` `--epochs` `--early_stopping_patience` `--noise_level` `--flip_prob` `--model_path` (optional):

```bash
# You must explicitly set CUDA_VISIBLE_DEVICES if you want to use GPU
$ export CUDA_VISIBLE_DEVICES="0"

$ python3 src/train.py --model_type TabTransformer --batch_size 32 --learning_rate 0.001 --epochs 100 --noise_level 0.01 --flip_prob 0.05 --early_stopping_patience 15
```

4. Extract attention weights from the last layer of the transformer and plot attention maps. Arguments: `--model_type` `--model_path` `--batch_size`:

```bash
$ python3 src/attention.py --model_type TabTransformer --model_path TabTransformer_bs32_lr0.001_ep100_nl0.01_fp0.05.pth --batch_size 32
```

5. Extract learned embeddings from the last layer of the transformer, apply the t-SNE algorithm to these embeddings, and then plot them. Arguments: `--model_type` `--model_path`  `--batch_size`:

```bash
$ python3 src/embeddings.py --model_type TabTransformer --model_path TabTransformer_bs32_lr0.001_ep100_nl0.01_fp0.05.pth --batch_size 32
```

6. Evaluate trained model on test set. Arguments: `--model_type` `--model_path` `--batch_size`:

```bash
$ python3 src/test.py --model_type TabTransformer --model_path TabTransformer_bs32_lr0.001_ep100_nl0.01_fp0.05.pth --batch_size 32
```