import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset 
import io
from tqdm import tqdm
import numpy as np
import gc
import json
import re
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
from pandas.api.types import is_categorical_dtype, is_sparse
import logging
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import dask
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar

# Determine the total number of available cores
total_cores = multiprocessing.cpu_count()

# Don't use all available cores
n_jobs = total_cores - 8

npartitions = int(n_jobs*2) # number of partitions for Dask

logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def optimize_dataframe(df):
    # Convert columns to more efficient types if possible
    # For example, convert object columns to category if they have limited unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Adjust this threshold as needed
                df[col] = df[col].astype('category')
    return df

def read_file(file_path, columns_type, columns_select, parse_dates=None, chunk_size=50000):
    """
    Reads a CSV file with a progress bar, selecting specific columns.

    Parameters:
        file_path (str): Path to the file.
        columns_type (dict): Dictionary of column names and their data types.
        columns_select (list): List of columns to read.
        parse_dates (list or None): List of columns to parse as dates.
        chunk_size (int): Number of rows per chunk.

    Returns:
        DataFrame: The read DataFrame.
    """
    # Filter columns_type to include only those columns in columns_select
    filtered_columns_type = {col: columns_type[col] for col in columns_select if col in columns_type}

    # Initialize a DataFrame to store the data
    data = pd.DataFrame()

    # Estimate the number of chunks
    total_rows = sum(1 for _ in open(file_path))
    total_chunks = (total_rows // chunk_size) + 1

    # Read the file in chunks with a progress bar
    with tqdm(total=total_chunks, desc="Reading CSV") as pbar:
        for chunk in pd.read_csv(file_path, sep='|', dtype=filtered_columns_type, usecols=columns_select,
                                 parse_dates=parse_dates, chunksize=chunk_size, low_memory=False):
            data = pd.concat([data, chunk], ignore_index=True)
            pbar.update(1)

    return data

def optimize_date_column(df, col, start_idx, end_idx):
    try:
        # Convert categorical columns to numerical codes
        if not pd.api.types.is_numeric_dtype(df[col]):
            df.loc[start_idx:end_idx, col] = df[col].astype('category').cat.codes
        
        # Perform the multiplication
        multiplied_result = df.loc[start_idx:end_idx, col] * df.loc[start_idx:end_idx, 'DiagnosisBeforeOrOnIndexDate']
        
        # Calculate sparsity ratio
        sparsity_ratio = (multiplied_result == 0).mean()

        # If the column is mostly zeros, convert to a sparse type to save memory
        if sparsity_ratio > 0.95:
            # Create a SparseArray and replace the existing column with it
            df[col] = pd.arrays.SparseArray(df[col], fill_value=0, dtype=df[col].dtype)
        else:
            # Directly assign the result to the DataFrame in the specified slice
            df.loc[start_idx:end_idx, col] = multiplied_result
    except Exception as e:
        print(f"Error processing column {col}: {e}")

def process_date_column(df, col, chunk_size=50000):
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        optimize_date_column(df, col, chunk_start, chunk_end)

def preprocess_and_save_info(df, encoded_feature_names, dataset_name):

    # Drop unnecessary columns
    df.drop(['EMPI', 'DiagnosisBeforeOrOnIndexDate'], axis=1, inplace=True)

    # Print dataset info
    print(f"Total patients (preprocessed) in {dataset_name}:", df.shape[0])
    print(f"Total features (preprocessed) in {dataset_name}:", df.shape[1])

    # Save updated encoded feature names and column names
    with open(f'{dataset_name}_encoded_feature_names.json', 'w') as file:
        json.dump(updated_feature_names, file)

    with open(f'{dataset_name}_column_names.json', 'w') as file:
        json.dump(df.columns.tolist(), file)

    return df, updated_feature_names

def dask_df_to_tensor(dask_df):
    # Convert the Dask DataFrame to a Dask Array
    dask_array = dask_df.to_dask_array(lengths=True)
    # Convert the Dask Array to a NumPy Array (this step triggers computation)
    numpy_array = dask_array.compute()
    # Finally, convert the NumPy Array to a PyTorch Tensor
    tensor = torch.tensor(numpy_array, dtype=torch.float32)
    return tensor

# Define the column types for each file type

def main():
    outcomes_columns = {
        'studyID': 'int32',
        'EMPI': 'int32',
        'IndexDate': 'object',
        'InitialA1c': 'float32',
        'A1cAfter12Months': 'float32',
        'A1cGreaterThan7': 'int32',
        'Female': 'int32',
        'Married': 'int32',
        'GovIns': 'int32',
        'English': 'int32',
        'DaysFromIndexToInitialA1cDate': 'int32',
        'DaysFromIndexToA1cDateAfter12Months': 'int32',
        'DaysFromIndexToFirstEncounterDate': 'int32',
        'DaysFromIndexToLastEncounterDate': 'int32',
        'DaysFromIndexToLatestDate': 'int32',
        'DaysFromIndexToPatientTurns18': 'int32',
        'AgeYears': 'int32',
        'BirthYear': 'int32',
        'NumberEncounters': 'int32',
        'SDI_score': 'float32',
        'Veteran': 'int32'
    }

    dia_columns = {
        'EMPI': 'int32',
        'Date': 'object', 
        'Code_Type': 'object', 
        'Code': 'object', 
        'IndexDate': 'object', 
        'DiagnosisBeforeOrOnIndexDate': 'int32', 
        'CodeWithType': 'object'
    }

    # Select columns to read in each dataset

    outcomes_columns_select = ['studyID', 'EMPI', 'InitialA1c', 'A1cAfter12Months', 'A1cGreaterThan7', 'Female', 'Married', 'GovIns', 'English', 'DaysFromIndexToInitialA1cDate', 'DaysFromIndexToA1cDateAfter12Months', 'DaysFromIndexToFirstEncounterDate', 'DaysFromIndexToLastEncounterDate', 'DaysFromIndexToLatestDate', 'DaysFromIndexToPatientTurns18', 'AgeYears', 'BirthYear', 'NumberEncounters', 'SDI_score', 'Veteran']

    dia_columns_select = ['EMPI', 'DiagnosisBeforeOrOnIndexDate', 'CodeWithType']

    # Define file path and selected columns
    outcomes_file_path = 'data/DiabetesOutcomes.txt'
    diagnoses_file_path = 'data/Diagnoses.txt'

    # Read each file into a DataFrame
    df_outcomes = read_file(outcomes_file_path, outcomes_columns, outcomes_columns_select)
    df_dia = read_file(diagnoses_file_path, dia_columns, dia_columns_select)

    # Check dimensions in outcomes data
    print("Number of patients:", len(df_outcomes)) # 55667
    print("Total features (original):", df_outcomes.shape[1]) # 20

    # Optimize dataFrames
    df_outcomes = optimize_dataframe(df_outcomes)
    df_dia = optimize_dataframe(df_dia)

    print("Preprocessing outcomes data")
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

    # Create the imputer object with strategy set to 'most_frequent' for mode imputation
    imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputer on the training data and transform the training data
    train_df[columns_with_missing_values] = imputer.fit_transform(train_df[columns_with_missing_values])

    # Apply the same imputation to validation and test datasets without fitting again
    validation_df[columns_with_missing_values] = imputer.transform(validation_df[columns_with_missing_values])
    test_df[columns_with_missing_values] = imputer.transform(test_df[columns_with_missing_values])

    # Normalize specified numeric columns in outcomes data using the Min-Max scaling approach. 
    # Handles negative and zero values well, scaling the data to a [0, 1] range.
    # NaNs are treated as missing values: disregarded in fit, and maintained in transform.

    columns_to_normalize = ['InitialA1c', 'A1cAfter12Months', 'DaysFromIndexToInitialA1cDate', 
                            'DaysFromIndexToA1cDateAfter12Months', 'DaysFromIndexToFirstEncounterDate', 
                            'DaysFromIndexToLastEncounterDate', 'DaysFromIndexToLatestDate', 
                            'DaysFromIndexToPatientTurns18', 'AgeYears', 'BirthYear', 
                            'NumberEncounters', 'SDI_score']

    print("Normalizing numeric colums: ", columns_to_normalize)

    with open('columns_to_normalize.json', 'w') as file:
        json.dump(columns_to_normalize, file)

    scaler = MinMaxScaler()

    # Fit on training data
    train_df[columns_to_normalize] = scaler.fit_transform(train_df[columns_to_normalize])

    # Transform validation and test data
    validation_df[columns_to_normalize] = scaler.transform(validation_df[columns_to_normalize])
    test_df[columns_to_normalize] = scaler.transform(test_df[columns_to_normalize])

    print("Preprocessing diagnoses data")

    # Create categorical feature for 'DiagnosisBeforeOrOnIndexDate' based on interaction with CodeWithType
    df_dia['Date'] = df_dia.apply(
        lambda row: row['CodeWithType'] if row['DiagnosisBeforeOrOnIndexDate'] == 1 else None,
        axis=1
    )

    # Handle categorical columns in diagnoses data
    categorical_columns = ['CodeWithType', 'Date']

    # Save the list of categorical columns to a JSON file
    with open('categorical_columns.json', 'w') as file:
        json.dump(categorical_columns, file)

    print("Converting categorical columns to string and handling missing values.")

    for col in categorical_columns:
        # Convert to string, fill missing values, then convert back to categorical if needed
        df_dia[col] = df_dia[col].astype(str).fillna('missing').replace({'': 'missing'}).astype('category')
     
    # Verify no NaN values exist
    assert not df_dia[categorical_columns] .isnull().any().any(), "NaN values found in the diagnoses categorical columns"

    # Verify the unique values in the 'CodeWithType' column
    print(f"Unique values in 'CodeWithType': {df_dia['CodeWithType'].unique()}")

    # Verify the unique values in the 'DiagnosisBeforeOrOnIndexDate' column
    print(f"Unique values in 'DiagnosisBeforeOrOnIndexDate': {df_dia['DiagnosisBeforeOrOnIndexDate'].unique()}")

    print("Initializing one-hot encoder for diagnoses data.")

    # ID rows in diagnoses data by split
    train_rows = df_dia[df_dia['EMPI'].isin(train_empi)]
    validation_rows = df_dia[df_dia['EMPI'].isin(validation_empi)]
    test_rows = df_dia[df_dia['EMPI'].isin(test_empi)]

     # Initialize OneHotEncoder with limited categories and sparse output
    one_hot_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore', min_frequency=389)

    # Fit the encoder on the training subset
    one_hot_encoder.fit(train_rows[categorical_columns].dropna().astype(str))

    # Apply the encoding to the validation and test subsets
    train_encoded = one_hot_encoder.transform(train_rows[categorical_columns].astype(str))
    validation_encoded = one_hot_encoder.transform(validation_rows[categorical_columns].astype(str))
    test_encoded = one_hot_encoder.transform(test_rows[categorical_columns].astype(str))

    # Extracting infrequent categories
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

    print("Combining encoded splits into a single sparse matrix.")

    # Combine the encoded train, validation, and test data into a single sparse matrix
    encoded_data = vstack([train_encoded, validation_encoded, test_encoded])

    # Get feature names from the encoder
    encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    # Process the feature names to remove the prefix
    encoded_feature_names  = [name.split('_', 1)[1] if 'CodeWithType' in name else name for name in encoded_feature_names]

    # Ensure encoded_feature_names is a list
    encoded_feature_names = list(encoded_feature_names)

    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoded_feature_names)

    # Drop the infrequent columns from encoded_df
    infrequent_sklearn_columns = ['Date_infrequent_sklearn',"infrequent_sklearn", "Date_None"]
    encoded_df = encoded_df.drop(columns=infrequent_sklearn_columns)

    encoded_feature_names = [col for col in encoded_feature_names if col not in infrequent_sklearn_columns]

    print("Dropping the original categorical columns ", categorical_columns)
    # Drop the original categorical columns
    df_dia.drop(categorical_columns, axis=1, inplace=True)

    print("Combining the original DataFrame with the encoded DataFrame.")

    # Concatenate using a method that preserves the sparse structure
    df_dia = pd.concat([df_dia, encoded_df], axis=1, sort=False)

    print("Binarizing one-hot encoded variables based on 'DiagnosisBeforeOrOnIndexDate'.")

    # Identify all one-hot encoded columns for 'Date'
    encoded_date_columns = [col for col in df_dia.columns if 'Date' in col]

    # Convert 'DiagnosisBeforeOrOnIndexDate' to a numeric type if it's categorical
    if is_categorical_dtype(df_dia['DiagnosisBeforeOrOnIndexDate']):
        df_dia['DiagnosisBeforeOrOnIndexDate'] = df_dia['DiagnosisBeforeOrOnIndexDate'].astype('int8', copy=False)

    # Preprocess encoded date columns
    print("Preprocess encoded date columns.")
    filtered_columns = [col for col in encoded_date_columns if col != 'DiagnosisBeforeOrOnIndexDate']
    tasks = [delayed(process_date_column)(df_dia, col) for col in filtered_columns]

    if tasks:
        with tqdm(total=len(tasks), desc="Processing Columns") as progress_bar:
            results = Parallel(n_jobs=n_jobs)(tasks)
            for _ in results:
                progress_bar.update()
    else:
        print("No columns meet the criteria for processing.")

    print("Starting aggregation by EMPI directly with Dask DataFrame operations.")

    # Check if df_dia is not a Dask DataFrame and convert it if necessary
    if not isinstance(df_dia, dd.DataFrame):
        df_dia = dd.from_pandas(df_dia, npartitions=npartitions)

    agg_dict = {col: 'max' for col in encoded_feature_names}
    print("Aggregation dictionary set up.")

    print("Perform the groupby and aggregation in parallel.")
    df_dia_agg = df_dia.groupby('EMPI').agg(agg_dict)

    print("Convert Dask DataFrame to pandas DataFrame.")
    with ProgressBar():
        df_dia_agg = df_dia_agg.compute()

    print("Merge diagnoses and outcomes datasets")

    merged_train_df = pd.merge(train_df, df_dia_agg, on='EMPI', how='inner')
    merged_validation_df = pd.merge(validation_df, df_dia_agg, on='EMPI', how='inner')
    merged_test_df = pd.merge(test_df, df_dia_agg, on='EMPI', how='inner')

    del df_dia, df_dia_agg
    gc.collect()

    # Apply the function to each merged dataset
    merged_train_df, train_encoded_feature_names = preprocess_and_save_info(merged_train_df, encoded_feature_names, 'train')
    merged_validation_df, validation_encoded_feature_names = preprocess_and_save_info(merged_validation_df, encoded_feature_names, 'validation')
    merged_test_df, test_encoded_feature_names = preprocess_and_save_info(merged_test_df, encoded_feature_names, 'test')

    print("Save training set as text file.")
    merged_train_df.to_csv('train_df.csv', index=False)

    print("Convert the DataFrames back to tensors for PyTorch processing.")
    dask_train_df = dd.from_pandas(merged_train_df, npartitions=npartitions)
    dask_validation_df = dd.from_pandas(merged_validation_df, npartitions=npartitions)
    dask_test_df = dd.from_pandas(merged_test_df, npartitions=npartitions)

    # Select numeric columns using Dask (this step is immediate and doesn't trigger computation)
    numeric_dask_train_df = dask_train_df.select_dtypes(include=[np.number])
    numeric_dask_validation_df = dask_validation_df.select_dtypes(include=[np.number])
    numeric_dask_test_df = dask_test_df.select_dtypes(include=[np.number])

    # Parallel conversion of Dask DataFrames to PyTorch tensors
    train_tensor = dask_df_to_tensor(numeric_dask_train_df)
    validation_tensor = dask_df_to_tensor(numeric_dask_validation_df)
    test_tensor = dask_df_to_tensor(numeric_dask_test_df)

    # Wrap tensors in TensorDataset
    train_dataset = TensorDataset(train_tensor)
    validation_dataset = TensorDataset(validation_tensor)
    test_dataset = TensorDataset(test_tensor)

    print("Save datasets as torch.")
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(validation_dataset, 'validation_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

if __name__ == '__main__':
    # This ensures the multiprocessing code only runs when the script is executed directly
    main()