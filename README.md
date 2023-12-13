# diabetes_pred
Identification of Temporal Data Patterns Predictive of Adverse Outcomes in Patients with Diabetes

## Requirements

This code has been tested on Python 3.6.9 on Ubuntu 18.04.1.

Requires PyTorch 1.8.1 compiled for CUDA 11.2.

## Hardware

NVIDIA Driver Version: 460.91.03

GPU: GeForce RTX 2080

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

1. Load and preprocess the data. Extract features from provider notes using NLP methods, and convert to PyTorch Tensor.

```bash
$ python3 src/data_loader.py
```

2. Split a PyTorch tensor into training, validation, and test sets in a 70-20-10 ratio:

```bash
$ python3 src/data_splitter.py
```

3. Train neural network (LSTMAttention or TCN) on training set, perform 5-fold cross-validation, and evaluate on validation set:

```bash
$ python3 train.py --model_type LSTMAttention
```
or 

```bash
$ python3 train.py --model_type TCN
```

4. Evaluate best model on test set:

```bash
$ python3 test.py --model_type LSTMAttention --model_path best_model.pth
```
or 

```bash
$ python3 test.py --model_type TCN --model_path best_model.pth
```