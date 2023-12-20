# 1. Read Each Text File: Use Pandas to read each of the eight types of text files.
# 2. Merge Data: Merge the data from all files by the 'EMPI' column.
# 3. Convert to PyTorch Tensor: Convert the merged data into a PyTorch Tensor and save it to file.

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import io
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc

# Check if 'vader_lexicon' is already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # If not present, download it
    nltk.download('vader_lexicon')

def extract_nlp_features(df, text_columns):
    """
    Extracts NLP features from specified text columns using TF-IDF.

    Parameters:
        df (DataFrame): The DataFrame containing text columns.
        text_columns (list): List of column names to extract NLP features from.

    Returns:
        DataFrame: The DataFrame with added TF-IDF features.
    """
    tfidf_vectorizer = TfidfVectorizer()

    for column in text_columns:
        # Convert categorical columns to object type
        if pd.api.types.is_categorical_dtype(df[column]):
            df[column] = df[column].astype('object')

        # Ensure that the operation is not performed on a copy
        df.loc[:, column] = df.loc[:, column].fillna('')  # Directly modify the DataFrame

        # Generate TF-IDF features
        tfidf_features = tfidf_vectorizer.fit_transform(df[column])

        # Create a DataFrame from the TF-IDF features
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names())

        # Concatenate the new features with the original DataFrame
        df = pd.concat([df, tfidf_df], axis=1)

    return df

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

def read_file(file_path, columns_type, columns_select, parse_dates=None, chunk_size=10000):
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

def encode_batch(batch, encoder):
    """
    Encodes a batch of data using one-hot encoding.

    Parameters:
        batch (DataFrame): The batch of data to encode.
        encoder (OneHotEncoder): The encoder to use for one-hot encoding.

    Returns:
        DataFrame: The encoded data.
    """
    encoded = encoder.transform(batch)
    return pd.DataFrame(encoded, index=batch.index)
    
# Define the column types for each file type
all_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'System': 'object', 'Noted_Date': 'object', 
    'Allergen': 'object', 'Allergen_Type': 'object', 'Allergen_Code': 'float64', 
    'Reactions': 'object', 'Severity': 'object', 'Reaction_Type': 'object', 
    'Comments': 'object', 'Status': 'object', 'Deleted_Reason_Comments': 'object'
}

con_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Last_Name': 'object', 'First_Name': 'object', 
    'Middle_Name': 'object', 'Research_Invitations': 'object', 
    'Address1': 'object', 'Address2': 'object', 'City': 'object', 
    'State': 'object', 'Zip': 'object', 'Country': 'object', 'Home_Phone': 'object', 
    'Day_Phone': 'object', 'SSN': 'object', 'VIP': 'object', 'Previous_Name': 'object', 
    'Patient_ID_List': 'object', 'Insurance_1': 'object', 'Insurance_2': 'object', 
    'Insurance_3': 'object', 'Primary_Care_Physician': 'object', 
    'Resident_Primary_Care_Physician': 'object'
}

dem_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Gender_Legal_Sex': 'object', 'Date_of_Birth': 'object', 
    'Age': 'float64', 'Sex_At_Birth': 'object', 'Gender_Identity': 'object', 
    'Language': 'object', 'Language_group': 'object', 'Race1': 'object', 
    'Race2': 'object', 'Race_Group': 'object', 'Ethnic_Group': 'object', 
    'Marital_status': 'object', 'Religion': 'object', 'Is_a_veteran': 'object', 
    'Zip_code': 'object', 'Country': 'object', 'Vital_status': 'object', 
    'Date_Of_Death': 'object'
}

dia_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Date': 'object', 'Diagnosis_Name': 'object', 
    'Code_Type': 'object', 'Code': 'object', 'Diagnosis_Flag': 'object', 
    'Provider': 'object', 'Clinic': 'object', 'Hospital': 'object', 
    'Inpatient_Outpatient': 'object', 'Encounter_number': 'object'
}

enc_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Encounter_number': 'object', 'Encounter_Status': 'object', 
    'Hospital': 'object', 'Inpatient_Outpatient': 'object', 'Service_Line': 'object', 
    'Attending_MD': 'object', 'Admit_Date': 'object', 'Discharge_Date': 'object', 
    'LOS_Days': 'float64', 'Clinic_Name': 'object', 'Admit_Source': 'object', 
    'Discharge_Disposition': 'object', 'Payor': 'object', 'Admitting_Diagnosis': 'object', 
    'Principal_Diagnosis': 'object', 'Diagnosis_1': 'object', 'Diagnosis_2': 'object', 
    'Diagnosis_3': 'object', 'Diagnosis_4': 'object', 'Diagnosis_5': 'object', 
    'Diagnosis_6': 'object', 'Diagnosis_7': 'object', 'Diagnosis_8': 'object', 
    'Diagnosis_9': 'object', 'Diagnosis_10': 'object', 'DRG': 'object', 
    'Patient_Type': 'object', 'Referrer_Discipline': 'object'
}

phy_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Date': 'object', 'Concept_Name': 'object', 
    'Code_Type': 'object', 'Code': 'object', 'Result': 'object', 
    'Units': 'object', 'Provider': 'object', 'Clinic': 'object', 
    'Hospital': 'object', 'Inpatient_Outpatient': 'object', 
    'Encounter_number': 'object'
}

prc_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'Date': 'object', 'Procedure_Name': 'object', 
    'Code_Type': 'object', 'Code': 'object', 'Procedure_Flag': 'object', 
    'Quantity': 'float64', 'Provider': 'object', 'Clinic': 'object', 
    'Hospital': 'object', 'Inpatient_Outpatient': 'object', 
    'Encounter_number': 'object'
}

# Select columns to read in each dataset

all_columns_select = ['EMPI', 'System', 'Allergen', 'Allergen_Type', 'Allergen_Code', 
    'Reactions', 'Severity', 'Reaction_Type', 'Status'] # 'Comments', 
con_columns_select = ['EMPI', 'Research_Invitations', 'City', 'State', 'Zip', 'Country', 'VIP', 'Insurance_1', 'Insurance_2', 'Insurance_3']
dem_columns_select = ['EMPI', 'Gender_Legal_Sex','Age', 'Sex_At_Birth', 'Gender_Identity', 
    'Language', 'Language_group',
    'Marital_status', 'Religion', 'Is_a_veteran','Vital_status']
dia_columns_select = ['EMPI', 'Code', 'Code_Type','Date','Diagnosis_Flag', 'Diagnosis_Name',  'Clinic', 'Hospital','Inpatient_Outpatient']
enc_columns_select = [ 'EMPI', 'Admit_Date', 'Encounter_Status', 
    'Hospital', 'Inpatient_Outpatient', 'Service_Line', 
    'LOS_Days', 'Clinic_Name', 'Admit_Source', 
    'Discharge_Disposition', 'Payor', 'Admitting_Diagnosis', 
    'Principal_Diagnosis', 'Diagnosis_1', 'Diagnosis_2', 
    'Diagnosis_3', 'Diagnosis_4', 'Diagnosis_5', 
    'Diagnosis_6', 'Diagnosis_7', 'Diagnosis_8','Diagnosis_9', 'Diagnosis_10'] # 'DRG','Patient_Type', 'Referrer_Discipline'
phy_columns_select = ['EMPI', 'Concept_Name', 'Date',
    'Code_Type', 'Code', 'Result', 
    'Units', 'Clinic', 
    'Hospital', 'Inpatient_Outpatient']
prc_columns_select = ['EMPI', 'Procedure_Name', 'Date',
    'Code_Type', 'Code', 'Procedure_Flag', 
    'Quantity', 'Clinic', 
    'Hospital', 'Inpatient_Outpatient']

# Define file paths (read in partial data for now)
path_to_all_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_All.txt'
path_to_con_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Con.txt'
path_to_dem_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Dem.txt'
path_to_dia_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Dia.txt'
path_to_enc_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Enc.txt'
path_to_prc_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Prc.txt'
path_to_phy_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Phy.txt'

# Read each file into a DataFrame
df_all = read_file(path_to_all_file, all_columns, all_columns_select, parse_dates=None)
df_con = read_file(path_to_con_file, con_columns, con_columns_select, parse_dates=None)
df_dem = read_file(path_to_dem_file, dem_columns, dem_columns_select, parse_dates=None)
df_dia = read_file(path_to_dia_file, dia_columns, dia_columns_select, parse_dates=['Date'])
df_enc = read_file(path_to_enc_file, enc_columns, enc_columns_select, parse_dates=['Admit_Date']) 
df_prc = read_file(path_to_prc_file, prc_columns, prc_columns_select, parse_dates=['Date'])
df_phy = read_file(path_to_phy_file, phy_columns, phy_columns_select, parse_dates=['Date'])

# Manually trigger garbage collection
gc.collect()

# Optimize DataFrames before merging
df_all = optimize_dataframe(df_all)
df_con = optimize_dataframe(df_con)
df_dem = optimize_dataframe(df_dem)
df_dia = optimize_dataframe(df_dia)
df_enc = optimize_dataframe(df_enc)
df_prc = optimize_dataframe(df_prc)
df_phy = optimize_dataframe(df_phy)

# Check and handle duplicates
df_all = df_all.drop_duplicates(subset='EMPI')
df_con = df_con.drop_duplicates(subset='EMPI')
df_dem = df_dem.drop_duplicates(subset='EMPI')
df_dia = df_dia.drop_duplicates(subset='EMPI')
df_enc = df_enc.drop_duplicates(subset='EMPI')
df_prc = df_prc.drop_duplicates(subset='EMPI')
df_phy = df_phy.drop_duplicates(subset='EMPI')

# Merge all DataFrames on 'EMPI'

temp_merge = df_all.merge(df_con, on='EMPI', how='outer')
del df_all, df_con
gc.collect()

temp_merge = temp_merge.merge(df_dem, on='EMPI', how='outer')
del df_dem
gc.collect()

temp_merge = temp_merge.merge(df_dia, on='EMPI', how='outer')
del df_dia
gc.collect()

temp_merge = temp_merge.merge(df_enc, on='EMPI', how='outer')
del df_enc
gc.collect()

temp_merge = temp_merge.merge(df_phy, on='EMPI', how='outer')
del df_phy
gc.collect()

merged_df = temp_merge.merge(df_prc, on='EMPI', how='outer')
del df_prc
gc.collect()

# Apply NLP Feature Extraction (skip for comments)
# text_columns = ['Comments']
# merged_df = extract_nlp_features(merged_df, text_columns)

# Identify categorical columns for one-hot encoding
categorical_columns = merged_df.select_dtypes(include=['object']).columns.tolist()

# Replace missing values with -1
# Fill NaN in numeric columns
numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
merged_df[numeric_columns] = merged_df[numeric_columns].fillna(-1)

# Handle NaN in categorical columns
for col in categorical_columns:
    if pd.api.types.is_categorical_dtype(merged_df[col]):
        # Add -1 as a new category and fill NaN values
        merged_df[col] = merged_df[col].cat.add_categories([-1])
        merged_df[col] = merged_df[col].fillna(-1)

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoder.fit(merged_df[categorical_columns])  # Fit the encoder on the full data

# Define batch size
batch_size = 10000  # Adjust this based on your system's capabilities

# Process in batches
encoded_batches = []
for start in range(0, merged_df.shape[0], batch_size):
    end = min(start + batch_size, merged_df.shape[0])
    batch = merged_df.iloc[start:end]
    encoded_batch = encode_batch(batch[categorical_columns], one_hot_encoder)
    encoded_batches.append(encoded_batch)

# Concatenate all encoded batches
encoded_categorical = pd.concat(encoded_batches)

# Drop original categorical columns and add encoded columns
merged_df.drop(categorical_columns, axis=1, inplace=True)
encoded_df = pd.concat([merged_df, encoded_categorical], axis=1)

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(encoded_df.values, dtype=torch.float32)

# Save the tensor to a file
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')