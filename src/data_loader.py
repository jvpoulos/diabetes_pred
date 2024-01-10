# 1. Read Each Text File: Use Pandas to read each of the eight types of text files.
# 2. Merge Data: Merge the data from all files by the 'EMPI' column.
# 3. Convert to PyTorch Tensor: Convert the merged data into a PyTorch Tensor and save it to file.

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import io
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
import json

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
   # Ensure all data in batch is string type for consistent encoding
    batch_str = batch.astype(str)
    encoded = encoder.transform(batch_str)
    return pd.DataFrame(encoded, index=batch.index)

# Define the column types for each file type

created_columns = {
    'Dataset_EMPI': 'float32', 
    'ElevatedA1cDate': 'object', 
    'IndexDate': 'object', 
    'InitialA1cDate': 'object', 
    'InitialA1c': 'float32',
    'ElevatedA1cDate': 'object', 
    'A1cAfter12Months': 'object'
}

all_columns = {
    'Allp_EPIC_PMRN': 'float32', 
    'Allp_MRN_Type': 'object', 
    'Allp_MRN': 'object', 
    'Allp_System': 'object', 
    'Allp_Noted_Date': 'object', 
    'Allp_Allergen': 'object', 
    'Allp_Allergen_Type': 'object', 
    'Allp_Allergen_Code': 'float32', 
    'Allp_Reactions': 'object', 
    'Allp_Severity': 'object', 
    'Allp_Reaction_Type': 'object', 
    'Allp_Comments': 'object', 
    'Allp_Status': 'object', 
    'Allp_Deleted_Reason_Comments': 'object'
}

con_columns = {
    'Conp_EPIC_PMRN': 'float32', 
    'Conp_MRN_Type': 'object', 
    'Conp_MRN': 'object', 
    'Conp_Last_Name': 'object', 
    'Conp_First_Name': 'object', 
    'Conp_Middle_Name': 'object', 
    'Conp_Research_Invitations': 'object', 
    'Conp_Address1': 'object', 
    'Conp_Address2': 'object', 
    'Conp_City': 'object', 
    'Conp_State': 'object', 
    'Conp_Zip': 'object', 
    'Conp_Country': 'object', 
    'Conp_Home_Phone': 'object', 
    'Conp_Day_Phone': 'object', 
    'Conp_SSN': 'object',
    'Conp_VIP': 'object', 
    'Conp_Previous_Name': 'object', 
    'Conp_Patient_ID_List': 'object', 
    'Conp_Insurance_1': 'object', 
    'Conp_Insurance_2': 'object', 
    'Conp_Insurance_3': 'object', 
    'Conp_Primary_Care_Physician': 'object', 
    'Conp_Resident_Primary_Care_Physician': 'object'
}

dem_columns = {
    'Demp_EPIC_PMRN': 'float32', 
    'Demp_MRN_Type': 'object', 
    'Demp_MRN': 'object', 
    'Demp_Gender_Legal_Sex': 'object', 
    'Demp_Date_of_Birth': 'object', 
    'Demp_Age': 'float32', 
    'Demp_Sex_At_Birth': 'object', 
    'Demp_Gender_Identity': 'object', 
    'Demp_Language': 'object', 
    'Demp_Language_group': 'object', 
    'Demp_Race1': 'object', 
    'Demp_Race2': 'object', 
    'Demp_Race_Group': 'object', 
    'Demp_Ethnic_Group': 'object', 
    'Demp_Marital_status': 'object', 
    'Demp_Religion': 'object', 
    'Demp_Is_a_veteran': 'object', 
    'Demp_Zip_code': 'object', 
    'Demp_Country': 'object', 
    'Demp_Vital_status': 'object', 
    'Demp_Date_Of_Death': 'object'
}

dia_columns = {
    'Diap_EPIC_PMRN': 'float32', 
    'Diap_MRN_Type': 'object', 
    'Diap_MRN': 'object', 'Date': 'object', 
    'Diap_Diagnosis_Name': 'object', 
    'Diap_Code_Type': 'object', 
    'Diap_Code': 'object', 
    'Diap_Diagnosis_Flag': 'object', 
    'Diap_Provider': 'object', 
    'Diap_Clinic': 'object', 
    'Diap_Hospital': 'object', 
    'Diap_Inpatient_Outpatient': 'object', 
    'Diap_Encounter_number': 'object'
}

enc_columns = {
    'Encp_EPIC_PMRN': 'float32', 
    'Encp_MRN_Type': 'object', 
    'Encp_MRN': 'object', 
    'Encp_Encounter_number': 'object', 
    'Encp_Encounter_Status': 'object', 
    'Encp_Hospital': 'object', 
    'Encp_Inpatient_Outpatient': 'object', 
    'Encp_Service_Line': 'object', 
    'Encp_Attending_MD': 'object', 
    'Encp_Admit_Date': 'object', 
    'Encp_Discharge_Date': 'object', 
    'Encp_LOS_Days': 'float32', 
    'Encp_Clinic_Name': 'object', 
    'Encp_Admit_Source': 'object', 
    'Encp_Discharge_Disposition': 'object', 
    'Encp_Payor': 'object', 
    'Encp_Admitting_Diagnosis': 'object', 
    'Encp_Principal_Diagnosis': 'object', 
    'Encp_Diagnosis_1': 'object', 
    'Encp_Diagnosis_2': 'object', 
    'Encp_Diagnosis_3': 'object', 
    'Encp_Diagnosis_4': 'object', 
    'Encp_Diagnosis_5': 'object', 
    'Encp_Diagnosis_6': 'object', 
    'Encp_Diagnosis_7': 'object', 
    'Encp_Diagnosis_8': 'object', 
    'Encp_Diagnosis_9': 'object', 
    'Encp_Diagnosis_10': 'object', 
    'Encp_DRG': 'object', 
    'Encp_Patient_Type': 'object', 
    'Encp_Referrer_Discipline': 'object'
}

phy_columns = {
    'Phyp_EPIC_PMRN': 'float32', 
    'Phyp_MRN_Type': 'object', 
    'Phyp_MRN': 'object', 
    'Phyp_Date': 'object', 
    'Phyp_Concept_Name': 'object', 
    'Phyp_Code_Type': 'object', 
    'Phyp_Code': 'object', 
    'Phyp_Result': 'object', 
    'Phyp_Units': 'object', 
    'Phyp_Provider': 'object', 
    'Phyp_Clinic': 'object', 
    'Phyp_Hospital': 'object', 
    'Phyp_Inpatient_Outpatient': 'object', 
    'Phyp_Encounter_number': 'object'
}

prc_columns = {
    'Prcp_EPIC_PMRN': 'float32', 
    'Prcp_MRN_Type': 'object', 
    'Prcp_MRN': 'object', 
    'Prcp_Date': 'object', 
    'Prcp_Procedure_Name': 'object', 
    'Prcp_Code_Type': 'object', 
    'Prcp_Code': 'object', 
    'Prcp_Procedure_Flag': 'object', 
    'Prcp_Quantity': 'float32', 
    'Prcp_Provider': 'object', 
    'Prcp_Clinic': 'object', 
    'Prcp_Hospital': 'object', 
    'Prcp_Inpatient_Outpatient': 'object', 
    'Prcp_Encounter_number': 'object'
}

# Select columns to read in each dataset

created_columns_select = ['Dataset_EMPI', 'ElevatedA1cDate','IndexDate','ElevatedA1cDate','InitialA1c','A1cDateAfter12Months','A1cAfter12Months']

all_columns_select = ['Allp_Allergen', 'Allp_Allergen_Type', 'Allp_Allergen_Code', 
    'Allp_Reactions', 'Allp_Severity', 'Allp_Reaction_Type', 'Allp_Status'] # many
con_columns_select = ['Conp_Research_Invitations', 'Conp_City', 'Conp_State', 'Conp_Zip', 'Conp_Country', 'Conp_VIP', 'Conp_Insurance_1', 'Conp_Insurance_2', 'Conp_Insurance_3']
dem_columns_select = ['Demp_Gender_Legal_Sex','Demp_Age', 'Demp_Sex_At_Birth',
    'Demp_Language', 'Demp_Language_group',
    'Demp_Marital_status', 'Demp_Religion', 'Demp_Is_a_veteran', 'Demp_Zip_code','Demp_Country','Demp_Vital_status'] # Demp_Date_of_Death'
dia_columns_select = ['Diap_Diagnosis_Name','Diap_Code', 'Diap_Code_Type','Diap_Date','Diap_Diagnosis_Flag', 'Diap_Clinic', 'Diap_Hospital','Diap_Inpatient_Outpatient'] # many # Diap_Date
enc_columns_select = [ 'Encp_Admit_Date', 'Encp_Encounter_Status', 
    'Encp_Hospital', 'Encp_Inpatient_Outpatient', 'Encp_Service_Line', 
    'Encp_LOS_Days', 'Encp_Clinic_Name', 'Encp_Admit_Source', 
    'Encp_Discharge_Disposition', 'Encp_Payor', 'Encp_Admitting_Diagnosis', 
    'Encp_Principal_Diagnosis', 'Encp_Diagnosis_1', 'Encp_Diagnosis_2', 
    'Encp_Diagnosis_3', 'Encp_Diagnosis_4', 'Encp_Diagnosis_5', 
    'Encp_Diagnosis_6', 'Encp_Diagnosis_7', 'Encp_Diagnosis_8','Encp_Diagnosis_9', 'Encp_Diagnosis_10'] # many # Encp_Admit_Date
phy_columns_select = ['Phyp_Concept_Name', 'Phyp_Date',
    'Phyp_Code_Type', 'Phyp_Code', 'Phyp_Result', 
    'Phyp_Units', 'Phyp_Clinic', 
    'Phyp_Hospital', 'Phyp_Inpatient_Outpatient'] # MANY Phyp_Date
prc_columns_select = ['Prcp_Procedure_Name', 'Prcp_Date',
    'Prcp_Code_Type', 'Prcp_Code', 'Prcp_Procedure_Flag', 
    'Prcp_Quantity', 'Prcp_Clinic', 
    'Prcp_Hospital', 'Prcp_Inpatient_Outpatient'] # MANY Prcp_Date

# Define file path and selected columns
file_path= 'data/DiabetesOutcomes.txt'

selected_columns = list(created_columns_select) + con_columns_select + dem_columns_select

selected_column_types = {**created_columns, **con_columns, **dem_columns}

# Read each file into a DataFrame
df = read_file(file_path, selected_column_types, selected_columns, parse_dates=['ElevatedA1cDate','IndexDate']) # ,'Demp_Date_of_Death'

# Optimize DataFrame
df = optimize_dataframe(df)
gc.collect()

# Check and handle duplicates
print("Number of rows before dropping duplicates:", len(df))
df = df.drop_duplicates(subset='Dataset_EMPI') # keeps first EMPI
print("Number of rows after dropping duplicates:", len(df))

print("Total features (original):", df.shape[1])

# Avoid chained assignment warnings
df.loc[:, 'Dataset_EMPI'] = df['Dataset_EMPI'].fillna(-1)

# Handle missing values safely
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_columns:
    df.loc[:, col] = df[col].fillna(-1)

# Handle categorical data
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

with open('categorical_columns.json', 'w') as file:
    json.dump(categorical_columns, file)

with open('numeric_columns.json', 'w') as file:
    json.dump(numeric_columns, file)

# # Summary statistics for categorical features
# categorical_stats = df[categorical_columns].apply(lambda col: col.value_counts(normalize=True))
# print(categorical_stats)

# Convert categorical columns to string and limit categories
# filter each categorical column to keep only the top x most frequent categories and replace other categories with a common placeholder like 'Other'
max_categories = 150 # Adjust this value as needed

for col in categorical_columns:
    top_categories = df[col].value_counts().nlargest(max_categories).index
    df[col] = df[col].astype(str)
    df[col] = df[col].where(df[col].isin(top_categories), other='Other')

# Initialize OneHotEncoder with limited categories
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoder.fit(df[categorical_columns])

batch_size = 1000  # Adjust batch size as needed
encoded_batches = []
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end]
    encoded_batch = one_hot_encoder.transform(batch[categorical_columns].astype(str))
    encoded_batches.append(encoded_batch)

# Concatenate all encoded batches and drop original categorical columns
encoded_batches = []
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end]
    encoded_batch = one_hot_encoder.transform(batch[categorical_columns].astype(str))
    # Convert numpy array to DataFrame
    encoded_batch_df = pd.DataFrame(encoded_batch, index=batch.index)
    encoded_batches.append(encoded_batch_df)

# concatenate using pd.concat
encoded_categorical = pd.concat(encoded_batches)

df.drop(categorical_columns, axis=1, inplace=True)
encoded_df = pd.concat([df, encoded_categorical], axis=1)

# Ensure all columns are numeric and convert to float32
for col in encoded_df.columns:
    if not pd.api.types.is_numeric_dtype(encoded_df[col]):
        # Handle non-numeric columns (e.g., drop or encode)
        # For example, dropping the column
        encoded_df.drop(col, axis=1, inplace=True)
    else:
        # Convert numeric columns to float32
        encoded_df[col] = encoded_df[col].astype('float32')

print("Total features (preprocessed):", encoded_df.shape[1])

# Convert to PyTorch tensor and save
pytorch_tensor = torch.tensor(encoded_df.values, dtype=torch.float32)
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')