# Diabetes predictive modelling
Identification of Temporal Data Patterns Predictive of Adverse Outcomes in Patients with Diabetes

## Models for Prediction

### [Tab Transformer](https://github.com/lucidrains/tab-transformer-pytorch)
The Tab Transformer is a novel model for tabular data that applies the self-attention mechanism from Transformer models to learn complex relationships between categorical features. It encodes categorical variables into embeddings and uses multiple layers of self-attention to dynamically understand the context of each feature within a row.

### [FT Transformer](https://github.com/lucidrains/tab-transformer-pytorch?tab=readme-ov-file#ft-transformer)
The FT Transformer extends the concepts from the Tab Transformer by incorporating feature tokenization, allowing it to effectively handle both categorical and continuous features. It improves upon the Tab Transformer by using feature-wise transformations which lead to better performance on a range of tabular datasets.

Both models have shown to outperform traditional deep learning and machine learning approaches on complex tabular datasets, making them a powerful tool for tasks such as classification and regression in structured data.

## Requirements

This code has been tested on Python 3.6.9 on Ubuntu 18.04.1.

Requires PyTorch 1.8.1 compiled for CUDA 11.2.

## Hardware

NVIDIA Driver Version: 460.91.03

GPU: GeForce RTX 2080

## Code Files in `src/`

- `data_loader.py`
	- Loads and merges data from .txt files. Preprocesses merged data. Converts the processed data into a PyTorch Tensor, and saves it to file.

- `data_splitter.py`
	- Randomly splits the processed dataset into training (70%), validation (20%), and test (10%) sets. Saves the split datasets to disk.

- `data_preprocessor.py`
	- Further preprocesses the datasets by applying feature selection techniques and visualizes the extent of sparsity in the training set.

- `train.py`
	- Trains deep learning models, supporting Tab Transformer and FT Transformer. Grid searches for the best model hyperparaters on the validation set and saves the best model.

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

4. Install all dependencies for your project from `requirements.txt` file:
```bash
$ pip3 install -r requirements.txt
```

## Run the code

1. Load and merge data from .txt files. Preprocess the data and convert the merge dataset to PyTorch Tensor:

```bash
$ python3 src/data_loader.py
```

2. Split the dataset into training, validation, and test sets in a 70-20-10 ratio:

```bash
$ python3 src/data_splitter.py
```

3. Further preprocess the datasets by applying feature selection techniques and visualize the extent of sparsity:

```bash
$ python3 src/data_preprocessor.py
```

4. Train neural network ('TabTransformer' or 'FTTransformer') on training set, perform 5-fold cross-validation, and evaluate on validation set:

```bash
$ python3 train.py --model_type TabTransformer
```

5. Evaluate best model on test set:

```bash
$ python3 test.py --model_type TabTransformer --model_path best_model.pth
```