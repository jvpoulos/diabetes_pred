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
from data_utils import read_file, dask_df_to_tensor, custom_one_hot_encoder
from data_dict import outcomes_columns, dia_columns, prc_columns, labs_columns, outcomes_columns_select, dia_columns_select_static, prc_columns_select_static, labs_columns_select_static
from itertools import zip_longest

# Define the column types for each file type

def main(use_dask=False, use_labs=False):

    # Define file path and selected columns
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'
    procedures_file_path = 'data/Procedures.txt'
    if use_labs:
        labs_file_path = 'data/Labs.txt'

    # Read each file into a DataFrame
    df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select_static)
    df_prc = read_file(procedures_file_path, prc_columns, prc_columns_select_static)
    if use_labs:
        df_labs = read_file(labs_file_path, labs_columns, labs_columns_select_static, chunk_size=700000)

    print("Number of patients:", len(df_outcomes))
    print("Total features (original):", df_outcomes.shape[1])

    print("Number of diagnoses before or on Index date:", len(df_dia))
    print("Number of procedures before or on Index date:", len(df_prc))
    if use_labs:
        print("Number of labs before or on Index date:", len(df_labs))

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

    print("Replacing missing values in ", 'SDI_score')

    # Create the imputer object
    outcomes_imputer = SimpleImputer(strategy='constant', fill_value=0)

    # Fit the imputer on the training data and transform the training data
    train_df[['SDI_score']] = outcomes_imputer.fit_transform(train_df[['SDI_score']])

    # Apply the same imputation to validation and test datasets without fitting again
    validation_df[['SDI_score']] = outcomes_imputer.transform(validation_df[['SDI_score']])
    test_df[['SDI_score']] = outcomes_imputer.transform(test_df[['SDI_score']])

    # Normalize specified numeric columns in outcomes data using the Min-Max scaling approach. 
    columns_to_normalize = ['InitialA1c', 'AgeYears', 'SDI_score']

    print("Normalizing numeric colums: ", columns_to_normalize)

    with open('columns_to_normalize.json', 'w') as file:
        json.dump(columns_to_normalize, file)

    outcomes_scaler = MaxAbsScaler()

    # Fit on training data
    train_df[columns_to_normalize] = outcomes_scaler.fit_transform(train_df[columns_to_normalize])

    # Transform validation and test data
    validation_df[columns_to_normalize] = outcomes_scaler.transform(validation_df[columns_to_normalize])
    test_df[columns_to_normalize] = outcomes_scaler.transform(test_df[columns_to_normalize])

    print("Preprocessing diagnoses data")

    print("Converting categorical columns to string and handling missing values.")

    # Convert to string, fill missing values, then convert back to categorical if needed
    df_dia['CodeWithType'] = df_dia['CodeWithType'].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_dia['CodeWithType'] .isnull().any().any(), "NaN values found in the diagnoses categorical columns"

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
    one_hot_encoder.fit(train_rows_dia[['CodeWithType']].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded_dia = one_hot_encoder.transform(train_rows_dia[['CodeWithType']].astype(str))
    validation_encoded_dia = one_hot_encoder.transform(validation_rows_dia[['CodeWithType']].astype(str))
    test_encoded_dia = one_hot_encoder.transform(test_rows_dia[['CodeWithType']].astype(str))

    print("Extracting diagnoses encoded feature names.")

    # Get feature names from the encoder
    dia_encoded_feature_names = one_hot_encoder.get_feature_names_out()

    # Process the feature names to remove the prefix
    dia_encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in dia_encoded_feature_names]

    # Ensure encoded_feature_names is a list
    dia_encoded_feature_names = list(dia_encoded_feature_names)

    print("Preprocessing procedures data")

    print("Converting categorical columns to string and handling missing values.")

    # Convert to string, fill missing values, then convert back to categorical if needed
    df_prc['CodeWithType'] = df_prc['CodeWithType'].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_prc['CodeWithType'] .isnull().any().any(), "NaN values found in the procedures categorical columns"

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
    one_hot_encoder.fit(train_rows_prc[['CodeWithType']].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded_prc = one_hot_encoder.transform(train_rows_prc[['CodeWithType']].astype(str))
    validation_encoded_prc = one_hot_encoder.transform(validation_rows_prc[['CodeWithType']].astype(str))
    test_encoded_prc = one_hot_encoder.transform(test_rows_prc[['CodeWithType']].astype(str))

    print("Extracting procedures encoded feature names.")

    # Get feature names from the encoder
    prc_encoded_feature_names = one_hot_encoder.get_feature_names_out()

    # Process the feature names to remove the prefix
    prc_encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in prc_encoded_feature_names]

    # Ensure encoded_feature_names is a list
    prc_encoded_feature_names = list(prc_encoded_feature_names)

    if use_labs:
        print("Preprocessing labs data")

        print("Converting categorical columns to string and handling missing values.")

        # Convert to string, fill missing values, then convert back to categorical if needed
        df_labs['Code'] = df_labs['Code'].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
         
        # Verify no NaN values exist
        assert not df_labs['Code'] .isnull().any().any(), "NaN values found in the labs categorical columns"

        print("Initializing one-hot encoder for labs data.")

        # Splitting the DataFrame while preserving the order
        train_rows_labs = df_labs[df_labs['EMPI'].isin(train_empi)].copy()
        validation_rows_labs = df_labs[df_labs['EMPI'].isin(validation_empi)].copy()
        test_rows_labs = df_labs[df_labs['EMPI'].isin(test_empi)].copy()

        # Add an 'order' column to each subset based on the original DataFrame's index or another unique identifier
        train_rows_labs['order'] = train_rows_labs.index
        validation_rows_labs['order'] = validation_rows_labs.index
        test_rows_labs['order'] = test_rows_labs.index

        # Verify the unique values in the 'Code' column
        print(f"Unique values in training set 'Code': {train_rows_labs[['Code']].nunique()}")

        print("Impute and standardize results and one-hot encode codes for labs data.")

        labs_scaler = MaxAbsScaler()

        if use_dask:
            # Impute and standardize results and one-hot encode codes for labs data.
            train_encoded_labs, labs_scaler, labs_encoded_feature_names = custom_one_hot_encoder(train_rows_labs, labs_scaler, fit=True, use_dask=True, chunk_size=10000, min_frequency=0.04)
            validation_encoded_labs, _ = custom_one_hot_encoder(validation_rows_labs, labs_scaler, fit=False, use_dask=True, chunk_size=10000, min_frequency=0.04)
            test_encoded_labs, _ = custom_one_hot_encoder(test_rows_labs, labs_scaler, fit=False, use_dask=True, chunk_size=10000, min_frequency=0.04)

            # Convert the encoded Dask DataFrame to a Pandas DataFrame
            train_encoded_labs = train_encoded_labs.compute()
            validation_encoded_labs = validation_encoded_labs.compute()
            test_encoded_labs = test_encoded_labs.compute()
        else:
            # Iterate over the encoded chunks for the training data
            train_encoded_labs_chunks = []
            for chunk, encoded_features, feature_names in custom_one_hot_encoder(train_rows_labs, labs_scaler, fit=True, use_dask=False, chunk_size=10000, min_frequency=0.04):
                # Concatenate the original chunk with the encoded features
                encoded_chunk = pd.concat([chunk, pd.DataFrame.sparse.from_spmatrix(encoded_features)], axis=1)
                train_encoded_labs_chunks.append(encoded_chunk)
                labs_encoded_feature_names = feature_names

            # Concatenate the encoded chunks into a single DataFrame
            train_encoded_labs = pd.concat(train_encoded_labs_chunks, ignore_index=True)

            # Iterate over the encoded chunks for the validation data
            validation_encoded_labs_chunks = []
            for chunk, encoded_features, _ in custom_one_hot_encoder(validation_rows_labs, labs_scaler, fit=False, use_dask=False, chunk_size=10000, min_frequency=0.04):
                encoded_chunk = pd.concat([chunk, pd.DataFrame.sparse.from_spmatrix(encoded_features)], axis=1)
                validation_encoded_labs_chunks.append(encoded_chunk)

            # Concatenate the encoded chunks into a single DataFrame
            validation_encoded_labs = pd.concat(validation_encoded_labs_chunks, ignore_index=True)

            # Iterate over the encoded chunks for the test data
            test_encoded_labs_chunks = []
            for chunk, encoded_features, _ in custom_one_hot_encoder(test_rows_labs, labs_scaler, fit=False, use_dask=False, chunk_size=10000, min_frequency=0.04):
                encoded_chunk = pd.concat([chunk, pd.DataFrame.sparse.from_spmatrix(encoded_features)], axis=1)
                test_encoded_labs_chunks.append(encoded_chunk)

            # Concatenate the encoded chunks into a single DataFrame
            test_encoded_labs = pd.concat(test_encoded_labs_chunks, ignore_index=True)

        # Process the feature names to remove the prefix
        labs_encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in labs_encoded_feature_names]

        # Ensure encoded_feature_names is a list
        labs_encoded_feature_names = list(labs_encoded_feature_names)

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

    if use_labs:
        print("Combining labs features splits into a single sparse matrix.")

        # Convert the encoded DataFrames to sparse matrices
        train_encoded_labs_matrix = sp.csr_matrix(train_encoded_labs.drop(['EMPI', 'Code', 'Result'], axis=1).astype(float).values)
        validation_encoded_labs_matrix = sp.csr_matrix(validation_encoded_labs.drop(['EMPI', 'Code', 'Result'], axis=1).astype(float).values)
        test_encoded_labs_matrix = sp.csr_matrix(test_encoded_labs.drop(['EMPI', 'Code', 'Result'], axis=1).astype(float).values)

        # Combine the horizontally stacked train, validation, and test data into a single sparse matrix
        encoded_data_labs = sp.vstack([train_encoded_labs_matrix, validation_encoded_labs_matrix, test_encoded_labs_matrix], format='csr')

        # Create a new index for encoded_df_labs
        new_index = pd.RangeIndex(start=0, stop=encoded_data_labs.shape[0], step=1)

        # Create a new DataFrame from the sparse matrix
        encoded_df_labs = pd.DataFrame.sparse.from_spmatrix(encoded_data_labs, columns=labs_encoded_feature_names, index=new_index)

        # Concatenate the encoded DataFrame with the original DataFrame
        encoded_df_labs = pd.concat([df_labs.reset_index(drop=True)[['EMPI', 'Code', 'Result']], encoded_df_labs], axis=1)

    print("Combining and updating encoded feature names.")

    dia_encoded_feature_names = [col for col in dia_encoded_feature_names if col not in infrequent_sklearn_columns]
    prc_encoded_feature_names = [col for col in prc_encoded_feature_names if col not in infrequent_sklearn_columns]
    if use_labs:
        labs_encoded_feature_names = [col for col in labs_encoded_feature_names if col not in infrequent_sklearn_columns]

    encoded_feature_names = dia_encoded_feature_names + prc_encoded_feature_names + (labs_encoded_feature_names if use_labs else [])

    with open('encoded_feature_names.json', 'w') as file:
        json.dump(encoded_feature_names, file)

    print("Combining the original DataFrame with the encoded DataFrame.")

    print("Number of one-hot encoded diagnoses:", len(encoded_df_dia)) 
    print("Number of one-hot encoded procedures:", len(encoded_df_prc)) 
    print("Number of diagnoses prior to concatenatation:", len(df_dia)) 
    print("Number of procedures prior to concatenatation:", len(df_prc))
    if use_labs:
        print("Number of labs prior to concatenatation:", len(df_labs))
        print("Number of one-hot encoded labs:", len(encoded_df_labs)) 

    # Reset the index of all DataFrames to ensure alignment
    df_dia = df_dia.reset_index(drop=True)
    df_prc = df_prc.reset_index(drop=True)
    encoded_df_dia = encoded_df_dia.reset_index(drop=True)
    encoded_df_prc = encoded_df_prc.reset_index(drop=True)

    # Verify that the number of rows matches to ensure a logical one-to-one row correspondence across all DataFrames
    assert len(df_dia) == len(encoded_df_dia), "Diagnoses row counts do not match."
    assert len(df_prc) == len(encoded_df_prc), "Procedures row counts do not match."
    if use_labs:
        assert len(df_labs) == len(encoded_df_labs), "Labs row counts do not match."

    # Check if the indexes are aligned
    assert df_dia.index.equals(encoded_df_dia.index), "Diagnoses indices are not aligned."
    assert df_prc.index.equals(encoded_df_prc.index), "Procedures indices are not aligned."
    if use_labs:
        assert df_labs.index.equals(encoded_df_labs.index), "Labs indices are not aligned."

    # Concatenate the DataFrames side-by-side
    df_dia = pd.concat([df_dia, encoded_df_dia], axis=1, sort=False)
    df_prc = pd.concat([df_prc, encoded_df_prc], axis=1, sort=False)
    if use_labs:
        df_labs = pd.concat([df_labs, encoded_df_labs], axis=1, sort=False)

    print("Number of diagnoses after concatenatation:", len(df_dia))
    # Verify row counts match expected transformations
    assert len(df_dia) == len(encoded_df_dia), f"Unexpected row count. Expected: {len(encoded_df_dia)}, Found: {len(df_dia)}"

    print("Number of procedures after concatenatation:", len(df_prc))
    assert len(df_prc) == len(encoded_df_prc), f"Unexpected row count. Expected: {len(encoded_df_prc)}, Found: {len(df_prc)}"

    if use_labs:
        print("Number of labs after concatenatation:", len(df_labs))
        assert len(df_labs) == len(encoded_df_labs), f"Unexpected row count. Expected: {len(encoded_df_labs)}, Found: {len(df_labs)}"
 
    print("Dropping the original categorical columns")
    df_dia.drop('CodeWithType', axis=1, inplace=True)
    df_prc.drop('CodeWithType', axis=1, inplace=True)
    if use_labs:
        df_labs.drop('Code', axis=1, inplace=True)

    print("Starting aggregation by EMPI.")

    agg_dict_dia = {col: 'max' for col in dia_encoded_feature_names}
    agg_dict_prc = {col: 'max' for col in prc_encoded_feature_names}
    if use_labs:
        agg_dict_labs = {col: 'max' for col in labs_encoded_feature_names}

    if use_dask:
        print("Starting aggregation by EMPI directly with Dask DataFrame operations.")
        print("Converting diagnoses to Dask DataFrame.")
        df_dia = dd.from_pandas(df_dia, npartitions=npartitions)
        df_prc = dd.from_pandas(df_prc, npartitions=npartitions)
        if use_labs:
            df_labs = dd.from_pandas(df_labs, npartitions=npartitions)

        print("Perform the groupby and aggregation in parallel.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict_dia)
        df_prc_agg = df_prc.groupby('EMPI').agg(agg_dict_prc)
        df_labs_agg = df_prc.groupby('EMPI').agg(agg_dict_labs)

        print("Convert splits to Dask DataFrame.")
        train_df = dd.from_pandas(train_df, npartitions=npartitions)
        validation_df = dd.from_pandas(validation_df, npartitions=npartitions)
        test_df = dd.from_pandas(test_df, npartitions=npartitions)
        client.close()
    else:
        print("Perform groupby and aggregation using pandas.")
        df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict_dia)
        df_prc_agg = df_prc.groupby('EMPI').agg(agg_dict_prc)
        if use_labs:
            df_labs_agg = df_labs.groupby('EMPI').agg(agg_dict_labs)

    print("Number of diagnoses before or on Index, after aggregation:", len(df_dia_agg))
    print("Number of procedures before or on Index, after aggregation:", len(df_prc_agg))
    if use_labs:
        print("Number of labs before or on Index, after aggregation:", len(df_labs_agg))

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
    parser.add_argument('--use_labs', action='store_true', help='Use labs data')
    args = parser.parse_args()
    if args.use_dask:
        # Adjust timeout settings
        dask.config.set({"distributed.comm.timeouts.connect": "60s"})
        dask.config.set({"distributed.comm.timeouts.tcp": "300s"})

        # Determine the total number of available cores
        total_cores = multiprocessing.cpu_count()

        # Don't use all available cores
        n_jobs = total_cores // 2

        npartitions = int(n_jobs*7) # number of partitions for Dask - Aim for partitions with around 100 MB of data each.

        client = Client(processes=True, memory_limit='8GB')  # Use processes instead of threads for potentially better CPU utilization
    main(use_dask=args.use_dask, use_labs=args.use_labs)