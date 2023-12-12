# 1. Read Each Text File: Use Pandas to read each of the eight types of text files.
# 2. Merge Data: Merge the data from all files by the 'EMPI' column.
# 3. Pre-process Data: Convert categorical data to numerical form using one-hot encoding and denote missing values with -999.
# 4. Extract features from provider notes using NLP methods.
# 5. Convert to PyTorch Tensor: Finally, convert the processed data into a PyTorch Tensor and save it to file.

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# To do: time var

# Check if 'vader_lexicon' is already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # If not present, download it
    nltk.download('vader_lexicon')

# Function to process the text columns and extract features, then using PCA to reduce the dimensionality of the NLP features.
def extract_nlp_features(df, text_columns):
    # Initialize TF-IDF Vectorizer and Sentiment Analyzer
    tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
    sia = SentimentIntensityAnalyzer()
    pca = PCA(n_components=50)  # Adjust n_components as needed

    # Create a DataFrame to hold all NLP features
    all_nlp_features = pd.DataFrame()

    for column in text_columns:
        # Replace NaNs with empty strings
        df[column].fillna('', inplace=True)

        # TF-IDF Features
        tfidf_features = tfidf_vectorizer.fit_transform(df[column])

        # Sentiment Analysis Features
        sentiment_scores = df[column].apply(lambda x: sia.polarity_scores(x))

        # Add TF-IDF features to the NLP features DataFrame
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        all_nlp_features = pd.concat([all_nlp_features, tfidf_df], axis=1)

        # Add sentiment scores to the NLP features DataFrame
        sentiment_df = sentiment_scores.apply(pd.Series)
        all_nlp_features = pd.concat([all_nlp_features, sentiment_df], axis=1)

    # Apply PCA to the NLP features
    pca_features = pca.fit_transform(all_nlp_features)

    # Create a DataFrame for the PCA-reduced features
    pca_df = pd.DataFrame(pca_features, columns=[f'pc{i+1}' for i in range(pca_features.shape[1])])

    # Add the PCA-reduced features back to the original DataFrame
    df = pd.concat([df, pca_df], axis=1)

    return df

# Function to read a file and return a DataFrame
def read_file(file_path, columns_type, columns_select):
    return pd.read_csv(file_path, sep='|', dtype=columns_type, low_memory=False, usecols=columns_select)

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
    'MRN': 'object', 'Date': 'object', 'Procedure_Name': 'object', 
    'Code_Type': 'object', 'Code': 'object', 'Procedure_Flag': 'object', 
    'Quantity': 'float64', 'Provider': 'object', 'Clinic': 'object', 
    'Hospital': 'object', 'Inpatient_Outpatient': 'object', 
    'Encounter_number': 'object'
}

mrn_columns = {
    'EMPI': 'float64', 'EPIC_PMRN': 'float64', 'MRN_Type': 'object', 
    'MRN': 'object', 'System': 'object', 'Noted_Date': 'object',
    'Allergen': 'object', 'Allergen_Type': 'object', 'Allergen_Code': 'float64',
    'Reactions': 'object', 'Severity': 'object', 'Reaction_Type': 'object', 'Comments': 'object',
    'Status': 'object','Deleted_Reason_Comments': 'object'
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

all_columns_select = ['EMPI', 'System', 'Noted_Date', 'Allergen', 'Allergen_Type', 'Allergen_Code', 
    'Reactions', 'Severity', 'Reaction_Type', 'Comments', 'Status']
con_columns_select = ['EMPI', 'Research_Invitations', 'City', 'State', 'Zip', 'VIP']
dem_columns_select = ['EMPI', 'Gender_Legal_Sex','Age', 'Sex_At_Birth', 'Gender_Identity', 
    'Language', 'Language_group', 'Race1', 
    'Race2', 'Race_Group', 'Ethnic_Group', 
    'Marital_status', 'Religion', 'Is_a_veteran', 
    'Zip_code', 'Country', 'Vital_status']
dia_columns_select = ['EMPI', 'Diagnosis_Name', 
    'Code_Type', 'Code', 'Diagnosis_Flag', 
    'Provider', 'Clinic', 'Hospital', 
    'Inpatient_Outpatient']
enc_columns_select = ['EMPI', 'Procedure_Name', 'Code', 'Procedure_Flag', 
    'Quantity', 'Provider', 'Clinic', 
    'Hospital', 'Inpatient_Outpatient']
# mrn_columns_select = ['EMPI', 'Allergen', 'Allergen_Type', 'Allergen_Code',
#     'Reactions', 'Severity', 'Reaction_Type', 'Comments',
#     'Status']
phy_columns_select = ['EMPI', 'Concept_Name', 
    'Code_Type', 'Code', 'Result', 
    'Units', 'Provider', 'Clinic', 
    'Hospital', 'Inpatient_Outpatient']
prc_columns_select = ['EMPI', 'Procedure_Name', 
    'Code_Type', 'Code', 'Procedure_Flag' 
    'Quantity', 'Provider', 'Clinic', 
    'Hospital', 'Inpatient_Outpatient']

# Define file paths (read in partial data for now)
path_to_all_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_All.txt'
path_to_con_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Con.txt'
path_to_dem_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Dem.txt'
path_to_dia_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Dia.txt'
path_to_enc_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Enc.txt'
#path_to_mrn_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Mrn.txt'
path_to_prc_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Prc.txt'
path_to_phy_file = 'data/2023P001659_20231129_153637/AT43_20231129_153637_Phy.txt'

# Read each file into a DataFrame
df_all = read_file(path_to_all_file, all_columns, all_columns_select)
df_con = read_file(path_to_con_file, con_columns, con_columns_select)
df_dem = read_file(path_to_dem_file, dem_columns, dem_columns_select)
df_dia = read_file(path_to_dia_file, dia_columns, dia_columns_select)
df_enc = read_file(path_to_enc_file, enc_columns, enc_columns_select)
#df_mrn = read_file(path_to_mrn_file, mrn_columns, phy_columns_select)
df_prc = read_file(path_to_prc_file, prc_columns, prc_columns_select)
df_phy = read_file(path_to_phy_file, phy_columns, phy_columns_select)

# Merge all DataFrames on 'EMPI'
merged_df = df_all.merge(df_con, on='EMPI', how='outer')
merged_df = merged_df.merge(df_dem, on='EMPI', how='outer')
merged_df = merged_df.merge(df_dia, on='EMPI', how='outer')
merged_df = merged_df.merge(df_enc, on='EMPI', how='outer')
#merged_df = merged_df.merge(df_mrn, on='EMPI', how='outer')
merged_df = merged_df.merge(df_phy, on='EMPI', how='outer')
merged_df = merged_df.merge(df_prc, on='EMPI', how='outer')

# Apply NLP Feature Extraction
text_columns = ['Comments']
merged_df = extract_nlp_features(merged_df, text_columns)

# Identify categorical columns for one-hot encoding
categorical_columns = [col for col in merged_df.columns if merged_df[col].dtype == 'object']

# Replace missing values with -999
merged_df.fillna(-999, inplace=True)

# One-hot encode categorical columns
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_categorical = one_hot_encoder.fit_transform(merged_df[categorical_columns])

# Drop original categorical columns and add encoded columns
merged_df.drop(categorical_columns, axis=1, inplace=True)
encoded_df = pd.concat([merged_df, pd.DataFrame(encoded_categorical)], axis=1)

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(encoded_df.values)

# Save the tensor to a file
torch.save(pytorch_tensor, 'preprocessed_tensor.pt')