import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset 
import io
from tqdm import tqdm
import numpy as np
import gc
from math import ceil
import json
import re
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_sparse
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import argparse
import dask
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, as_completed
from data_utils import read_file, dask_df_to_tensor
from rdpr_dict import outcomes_columns, dia_columns, prc_columns, outcomes_columns_select, dia_columns_select, prc_columns_select

# Define the column types for each file type

def main(use_dask=False):
    # Define file path and selected columns
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'

    # Read each file into a DataFrame
    df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select)
    df_prc = read_file(procedures_file_path, prc_columns, prc_columns_select)

    print("Number of diagnoses before or on Index date:", len(df_dia))
    print("Number of procedures before or on Index date:", len(df_prc))

    print("Number of patients:", len(df_outcomes))
    print("Total features (original):", df_outcomes.shape[1])

    print("Splitting outcomes data")
    # Splitting the data into train, validation, and test sets
    train_df, temp_df = train_test_split(df_outcomes, test_size=0.3, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)

    del df_outcomes

    print("Training set size:", train_df.shape[0])
    print("Validation set size:", validation_df.shape[0])
    print("Test set size:", test_df.shape[0])

    # Extracting 'EMPI' identifiers for each set
    train_empi = train_df['EMPI']
    validation_empi = validation_df['EMPI']
    test_empi = test_df['EMPI']

    # Save the 'EMPI' identifiers to files
    train_empi.to_csv('train_empi.csv', index=False)
    validation_empi.to_csv('validation_empi.csv', index=False)
    test_empi.to_csv('test_empi.csv', index=False)

    # List of columns with known missing values
    columns_with_missing_values = ['SDI_score']

    with open('columns_with_missing_values.json', 'w') as file:
        json.dump(columns_with_missing_values, file)

    print("Replacing missing values in ", columns_with_missing_values)

    # Create the imputer object
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    # Fit the imputer on the training data and transform the training data
    train_df[columns_with_missing_values] = imputer.fit_transform(train_df[columns_with_missing_values])

    # Apply the same imputation to validation and test datasets without fitting again
    validation_df[columns_with_missing_values] = imputer.transform(validation_df[columns_with_missing_values])
    test_df[columns_with_missing_values] = imputer.transform(test_df[columns_with_missing_values])

    # Normalize specified numeric columns in outcomes data using the Min-Max scaling approach. 
    columns_to_normalize = ['InitialA1c', 'AgeYears', 'SDI_score']

    print("Normalizing numeric colums: ", columns_to_normalize)

    with open('columns_to_normalize.json', 'w') as file:
        json.dump(columns_to_normalize, file)

    scaler = MaxAbsScaler()

    # Fit on training data
    train_df[columns_to_normalize] = scaler.fit_transform(train_df[columns_to_normalize])

    # Transform validation and test data
    validation_df[columns_to_normalize] = scaler.transform(validation_df[columns_to_normalize])
    test_df[columns_to_normalize] = scaler.transform(test_df[columns_to_normalize])

    print("Preprocessing diagnoses data")

    # Handle categorical columns
    categorical_columns = ['CodeWithType']

    # Save the list of categorical columns to a JSON file
    with open('categorical_columns.json', 'w') as file:
        json.dump(categorical_columns, file)

    print("Converting categorical columns to string and handling missing values.")

    for col in categorical_columns:
        # Convert to string, fill missing values, then convert back to categorical if needed
        df_dia[col] = df_dia[col].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_dia[categorical_columns] .isnull().any().any(), "NaN values found in the diagnoses categorical columns"

    print("Initializing one-hot encoder for diagnoses data.")

    # Splitting the DataFrame while preserving the order
    train_rows_dia = df_dia[df_dia['EMPI'].isin(train_empi)].copy()
    validation_rows_dia = df_dia[df_dia['EMPI'].isin(validation_empi)].copy()
    test_rows_dia = df_dia[df_dia['EMPI'].isin(test_empi)].copy()

    # Add an 'order' column to each subset based on the original DataFrame's index or another unique identifier
    train_rows_dia['order'] = train_rows_dia.index
    validation_rows_dia['order'] = validation_rows_dia.index
    test_rows_dia['order'] = test_rows_dia.index

    # Verify the unique values in the 'CodeWithType' column
    print(f"Unique values in training set 'CodeWithType': {train_rows_dia['CodeWithType'].unique()}")

    # Initialize OneHotEncoder with limited categories and sparse output
    one_hot_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore', min_frequency=ceil(37908*0.01))

    # Fit the encoder on the training subset
    one_hot_encoder.fit(train_rows_dia[categorical_columns].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded_dia = one_hot_encoder.transform(train_rows_dia[categorical_columns].astype(str))
    validation_encoded_dia = one_hot_encoder.transform(validation_rows_dia[categorical_columns].astype(str))
    test_encoded_dia = one_hot_encoder.transform(test_rows_dia[categorical_columns].astype(str))

    print("Extracting dia encoded feature names.")

    # Get feature names from the encoder
    dia_encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    # Process the feature names to remove the prefix
    dia_encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in dia_encoded_feature_names]

    # Ensure encoded_feature_names is a list
    dia_encoded_feature_names = list(dia_encoded_feature_names)

    print("Preprocessing procedures data")

    print("Converting categorical columns to string and handling missing values.")

    for col in categorical_columns:
        # Convert to string, fill missing values, then convert back to categorical if needed
        df_prc[col] = df_prc[col].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_prc[categorical_columns] .isnull().any().any(), "NaN values found in the procedures categorical columns"

    print("Initializing one-hot encoder for procedures data.")

    # Splitting the DataFrame while preserving the order
    train_rows_prc = df_prc[df_prc['EMPI'].isin(train_empi)].copy()
    validation_rows_prc = df_prc[df_prc['EMPI'].isin(validation_empi)].copy()
    test_rows_prc = df_prc[df_prc['EMPI'].isin(test_empi)].copy()

    # Add an 'order' column to each subset based on the original DataFrame's index or another unique identifier
    train_rows_prc['order'] = train_rows_prc.index
    validation_rows_prc['order'] = validation_rows_prc.index
    test_rows_prc['order'] = test_rows_prc.index

    # Verify the unique values in the 'CodeWithType' column
    print(f"Unique values in training set 'CodeWithType': {train_rows_prc['CodeWithType'].unique()}")

    # Fit the encoder on the training subset
    one_hot_encoder.fit(train_rows_prc[categorical_columns].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded_prc = one_hot_encoder.transform(train_rows_prc[categorical_columns].astype(str))
    validation_encoded_prc = one_hot_encoder.transform(validation_rows_prc[categorical_columns].astype(str))
    test_encoded_prc = one_hot_encoder.transform(test_rows_prc[categorical_columns].astype(str))

    print("Extracting infrequent categories.")

    infrequent_categories = one_hot_encoder.infrequent_categories_

    # Prepare a dictionary to hold the infrequent categories for each feature
    infrequent_categories_dict = {
        categorical_columns[i]: list(infrequent_categories[i])
        for i in range(len(categorical_columns))
    }

    # Save the infrequent categories to a JSON file
    with open('infrequent_categories.json', 'w') as f:
        json.dump(infrequent_categories_dict, f)

    print("Infrequent categories saved to infrequent_categories.json.")

    print("Extracting prc encoded feature names.")

    # Get feature names from the encoder
    prc_encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    # Process the feature names to remove the prefix
    prc_encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in prc_encoded_feature_names]

    # Ensure encoded_feature_names is a list
    prc_encoded_feature_names = list(prc_encoded_feature_names)

    print("Combining diagnoses encoded splits into a single sparse matrix.")

    # Combine the horizontally stacked train, validation, and test data into a single sparse matrix
    encoded_data_dia = vstack([train_encoded_dia, validation_encoded_dia, test_encoded_dia])

    encoded_df_dia = pd.DataFrame.sparse.from_spmatrix(encoded_data_dia, columns=dia_encoded_feature_names) 

    # Concatenate the 'order' column to encoded_df to preserve original row order
    orders_dia = pd.concat([train_rows_dia['order'], validation_rows_dia['order'], test_rows_dia['order']])
    encoded_df_dia['order'] = orders_dia.values

    # Sort encoded_df by 'order' to match the original df_dia row order and reset the index
    encoded_df_dia.sort_values(by='order', inplace=True)
    encoded_df_dia.reset_index(drop=True, inplace=True)

    # Drop the 'order' column if it's no longer needed
    encoded_df_dia.drop(columns=['order'], inplace=True)

    # Drop the infrequent columns from encoded_df
    infrequent_sklearn_columns = ["infrequent_sklearn"]
    encoded_df_dia = encoded_df_dia.drop(columns=infrequent_sklearn_columns)

    print("Combining procedures encoded splits into a single sparse matrix.")

    # Combine the horizontally stacked train, validation, and test data into a single sparse matrix
    encoded_data_prc = vstack([train_encoded_prc, validation_encoded_prc, test_encoded_prc])

    encoded_df_prc = pd.DataFrame.sparse.from_spmatrix(encoded_data_prc, columns=prc_encoded_feature_names)

    # Concatenate the 'order' column to encoded_df to preserve original row order
    orders_prc = pd.concat([train_rows_prc['order'], validation_rows_prc['order'], test_rows_prc['order']])
    encoded_df_prc['order'] = orders_prc.values

    # Sort encoded_df by 'order' to match the original df_dia row order and reset the index
    encoded_df_prc.sort_values(by='order', inplace=True)
    encoded_df_prc.reset_index(drop=True, inplace=True)

    # Drop the 'order' column if it's no longer needed
    encoded_df_prc.drop(columns=['order'], inplace=True)

    # Drop the infrequent columns from encoded_df
    encoded_df_prc = encoded_df_prc.drop(columns=infrequent_sklearn_columns)

    print("Combining and updating encoded feature names.")

    dia_encoded_feature_names = [col for col in dia_encoded_feature_names if col not in infrequent_sklearn_columns]
    prc_encoded_feature_names = [col for col in prc_encoded_feature_names if col not in infrequent_sklearn_columns]

    encoded_feature_names = dia_encoded_feature_names + prc_encoded_feature_names
    with open('encoded_feature_names.json', 'w') as file:
        json.dump(encoded_feature_names, file)

    print("Combining the original DataFrame with the encoded DataFrame.")

    print("Number of one-hot encoded diagnoses:", len(encoded_df_dia)) 
    print("Number of one-hot encoded procedures:", len(encoded_df_prc)) 
    print("Number of diagnoses prior to concatenatation:", len(df_dia)) 
    print("Number of procedures prior to concatenatation:", len(df_prc)) 

    # Reset the index of all DataFrames to ensure alignment
    df_dia = df_dia.reset_index(drop=True)
    df_prc = df_prc.reset_index(drop=True)
    encoded_df_dia = encoded_df_dia.reset_index(drop=True)
    encoded_df_prc = encoded_df_prc.reset_index(drop=True)

    # Verify that the number of rows matches to ensure a logical one-to-one row correspondence across all DataFrames
    assert len(df_dia) == len(encoded_df_dia), "Diagnoses row counts do not match."
    assert len(df_prc) == len(encoded_df_prc), "Procedures row counts do not match."

    # Check if the indexes are aligned
    assert df_dia.index.equals(encoded_df_dia.index), "Diagnoses indexes are not aligned."
    assert df_prc.index.equals(encoded_df_prc.index), "Procedures indexes are not aligned."

    # Concatenate the DataFrames side-by-side
    df_dia = pd.concat([df_dia, encoded_df_dia], axis=1, sort=False)
    df_prc = pd.concat([df_prc, encoded_df_prc], axis=1, sort=False)

    print("Number of diagnoses after concatenatation:", len(df_dia))
    # Verify row counts match expected transformations
    assert len(df_dia) == len(encoded_df_dia), f"Unexpected row count. Expected: {len(encoded_df_dia)}, Found: {len(df_dia)}"

    print("Number of diagnoses after concatenatation:", len(df_prc))
    # Verify row counts match expected transformations
    assert len(df_prc) == len(encoded_df_prc), f"Unexpected row count. Expected: {len(encoded_df_prc)}, Found: {len(df_prc)}"
 
    print("Dropping the original categorical columns ", categorical_columns)
    df_dia.drop(categorical_columns, axis=1, inplace=True)
    df_prc.drop(categorical_columns, axis=1, inplace=True)

    print("Starting aggregation by EMPI.")

    agg_dict_dia = {col: 'max' for col in dia_encoded_feature_names}
    agg_dict_prc = {col: 'max' for col in prc_encoded_feature_names}
    if use_dask:
        print("Starting aggregation by EMPI directly with Dask DataFrame operations.")
        print("Converting diagnoses to Dask DataFrame.")
        df_dia = dd.from_pandas(df_dia, npartitions=npartitions)
        df_prc = dd.from_pandas(df_prc, npartitions=npartitions)

        print("Perform the groupby and aggregation in parallel.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict_dia)
        df_prc_agg = df_prc.groupby('EMPI').agg(agg_dict_prc)

        print("Convert splits to Dask DataFrame.")
        train_df = dd.from_pandas(train_df, npartitions=npartitions)
        validation_df = dd.from_pandas(validation_df, npartitions=npartitions)
        test_df = dd.from_pandas(test_df, npartitions=npartitions)
    else:
        print("Perform groupby and aggregation using pandas.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict_dia)
        df_prc_agg = df_prc.groupby('EMPI').agg(agg_dict_prc)

    print("Number of diagnoses before or on Index, after aggregation:", len(df_dia_agg))
    print("Number of procedures before or on Index, after aggregation:", len(df_prc_agg))

    print("Merge outcomes splits with aggregated diagnoses.")
    merged_train_df = train_df.merge(df_dia_agg, on='EMPI', how='inner').merge(df_prc_agg, on='EMPI', how='inner')
    merged_validation_df = validation_df.merge(df_dia_agg, on='EMPI', how='inner').merge(df_prc_agg, on='EMPI', how='inner')
    merged_test_df = test_df.merge(df_dia_agg, on='EMPI', how='inner').merge(df_prc_agg, on='EMPI', how='inner')

    print("Select numeric columns and drop EMPI using Pandas.")
    numeric_train_df = merged_train_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')
    numeric_validation_df = merged_validation_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')
    numeric_test_df = merged_test_df.select_dtypes(include=[np.number]).drop(columns=['EMPI'], errors='ignore')

    print("Merged training set size:", numeric_train_df.shape[0])
    print("Merged validation set size:", numeric_validation_df.shape[0])
    print("Merged test set size:", numeric_test_df.shape[0])

    column_names = numeric_train_df.columns.tolist()

    # Save column names to a JSON file
    with open('column_names.json', 'w') as file:
        json.dump(column_names, file)

    print("Clean up unused dataframes.")
    del df_dia, df_prc, df_dia_agg, df_prc_agg
    gc.collect()
    
    if use_dask:
        print("Parallel conversion of training set Dask DataFrame to PyTorch tensor.")
        train_tensor = dask_df_to_tensor(numeric_train_df)
        print("Parallel conversion of validation set Dask DataFrame to PyTorch tensor.")
        validation_tensor = dask_df_to_tensor(numeric_validation_df)
        print("Parallel conversion of test set Dask DataFrame to PyTorch tensor.")
        test_tensor = dask_df_to_tensor(numeric_test_df)
        client.close()
    else:
        print("Convert to PyTorch tensors using Pandas.")
        train_tensor = torch.tensor(numeric_train_df.values, dtype=torch.float32)
        validation_tensor = torch.tensor(numeric_validation_df.values, dtype=torch.float32)
        test_tensor = torch.tensor(numeric_test_df.values, dtype=torch.float32)

    print("Wrap tensors in TensorDataset.")
    train_dataset = TensorDataset(train_tensor)
    validation_dataset = TensorDataset(validation_tensor)
    test_dataset = TensorDataset(test_tensor)

    print("Save datasets to file.")
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(validation_dataset, 'validation_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dask', action='store_true', help='Use Dask for processing')
    args = parser.parse_args()
    if args.use_dask:
        # Adjust timeout settings
        dask.config.set({'distributed.comm.timeouts.connect': '120s'})

        # Determine the total number of available cores
        total_cores = multiprocessing.cpu_count()

        # Don't use all available cores
        n_jobs = total_cores // 2

        npartitions = int(n_jobs*7) # number of partitions for Dask - Aim for partitions with around 100 MB of data each.

        client = Client(processes=True)  # Use processes instead of threads for potentially better CPU utilization
    main(use_dask=args.use_dask)