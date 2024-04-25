import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import plotly.express as px
import argparse
import os

# This function formats ICD codes by removing dots and leading zeros
def format_icd_code(icd_code):
    # Convert the ICD code to a string and remove trailing spaces
    return str(icd_code).rstrip()
    # # Convert the ICD code to a string, remove any dots, and trailing spaces
    # return str(icd_code).replace('.', '').lstrip('0').rstrip()

def get_icd_description(icd_code, code_type, icd9_df, icd10_df, icd9_txt_df, icd10_txt_df, cpt_df, cpt_txt_df):
    formatted_icd_code = format_icd_code(icd_code)
    try:
        if code_type.upper() == 'ICD9':
            if formatted_icd_code in icd9_txt_df.index:
                result = icd9_txt_df.loc[formatted_icd_code, 'Description']
                return result if isinstance(result, str) else result.iloc[0]
            elif formatted_icd_code in icd9_df.index:
                result = icd9_df.loc[formatted_icd_code, 'LONG DESCRIPTION (VALID ICD-9 FY2024)']
                return result if isinstance(result, str) else result.iloc[0]
        elif code_type.upper() == 'ICD10':
            if formatted_icd_code in icd10_txt_df.index:
                result = icd10_txt_df.loc[formatted_icd_code, 'Description']
                return result if isinstance(result, str) else result.iloc[0]
            elif formatted_icd_code in icd10_df.index:
                result = icd10_df.loc[formatted_icd_code, 'LONG DESCRIPTION (VALID ICD-10 FY2024)']
                return result if isinstance(result, str) else result.iloc[0]
        elif code_type.upper() == 'CPT':
            if formatted_icd_code in cpt_df.index:
                result = cpt_df.loc[formatted_icd_code, 'Description']
                return result if isinstance(result, str) else result.iloc[0]
            elif formatted_icd_code in cpt_txt_df.index:
                result = cpt_txt_df.loc[formatted_icd_code, 'long_description']
                return result if isinstance(result, str) else result.iloc[0]       
    except KeyError:
        # Log the error or print a message if needed
        pass
    return 'Description not found'

# Extract ICD code and type
def extract_icd_info(col_name):
    parts = col_name.split('_')
    if len(parts) != 2:
        return None, None
    code = parts[0]
    code_type = 'ICD10' if 'ICD10' in parts[1] else 'ICD9'
    return code, code_type

def load_cpt_codes(file_path, cpt_txt_file_path):
    # Read the file with tab delimiter and specify the header
    cpt_df = pd.read_csv(file_path, sep="\t", header=0, usecols=['Code', 'Description'])

    # Convert code column to string to ensure proper matching
    cpt_df['Code'] = cpt_df['Code'].astype(str)

    # Set the 'Code' column as the index for fast lookup
    cpt_df.set_index('Code', inplace=True)

    # Apply the format_icd_code function to the index
    cpt_df.index = cpt_df.index.map(format_icd_code)

    # Load the CPT.txt file with explicit encoding
    cpt_txt_df = pd.read_csv(cpt_txt_file_path, sep='\t', header=None, usecols=[0, 3], names=['CPTcode', 'long_description'], encoding='latin1')

    # Convert code column to string to ensure proper matching
    cpt_txt_df['CPTcode'] = cpt_txt_df['CPTcode'].astype(str)

    # Set the 'CPTcode' column as the index for fast lookup
    cpt_txt_df.set_index('CPTcode', inplace=True)

    # Apply the format_icd_code function to the index
    cpt_txt_df.index = cpt_txt_df.index.map(format_icd_code)

    return cpt_df, cpt_txt_df

def main(use_labs=False):

    print("Loading ICD .csv files")
    # Load the ICD-9 DataFrame with the correct columns
    icd9_df = pd.read_excel('data/Section111ValidICD9-Jan2024.xlsx', engine='openpyxl', dtype=str)

    # Load the ICD-10 DataFrame with the correct columns
    icd10_df = pd.read_excel('data/Section111ValidICD10-Jan2024.xlsx', engine='openpyxl', dtype=str)

    # Load the ICD-9 and ICD-10 descriptions into a dictionary for quick access
    icd9_descriptions = icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].to_dict()
    icd10_descriptions = icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].to_dict()

    # Check if the column exists and replace NaN with a placeholder if needed
    if 'LONG DESCRIPTION (VALID ICD-9 FY2024)' in icd9_df.columns:
        icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].fillna('Description not available', inplace=True)
    else:
        raise ValueError("Expected column 'LONG DESCRIPTION (VALID ICD-9 FY2024)' not found in icd9_df")

    if 'LONG DESCRIPTION (VALID ICD-10 FY2024)' in icd10_df.columns:
        icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].fillna('Description not available', inplace=True)
    else:
        raise ValueError("Expected column 'LONG DESCRIPTION (VALID ICD-10 FY2024)' not found in icd10_df")

    # Now the assertions should not fail as we have ensured the columns exist and have no NaN values
    assert not icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].isnull().any()
    assert not icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].isnull().any()

    # Continue with setting the index and other operations
    # Assuming 'CODE' column contains the ICD codes and 'LONG DESCRIPTION...' contains descriptions
    icd9_df.set_index('CODE', inplace=True)
    icd10_df.set_index('CODE', inplace=True)

    # Apply formatting function to the DataFrame indices
    icd9_df.index = icd9_df.index.astype(str).map(format_icd_code)
    icd10_df.index = icd10_df.index.astype(str).map(format_icd_code)

    # Fill NaN values in description columns with a default message
    icd9_df['LONG DESCRIPTION (VALID ICD-9 FY2024)'].fillna('Description not available', inplace=True)
    icd10_df['LONG DESCRIPTION (VALID ICD-10 FY2024)'].fillna('Description not available', inplace=True)

    print("Loading ICD .txt files.")

    # Load the text files into DataFrames
    icd9_txt_df = pd.read_csv('data/ICD9.txt', sep=',', header=0, names=['ICD9code', 'Description', 'Type', 'Source', 'Dt'], encoding='ISO-8859-1', on_bad_lines='skip')
    icd10_txt_df = pd.read_csv('data/ICD10.txt', sep=',', header=0, names=['GUID', 'ICD10code', 'Description', 'Type', 'Source', 'Language', 'LastUpdateDTS', 'Dt'], encoding='ISO-8859-1', on_bad_lines='skip')

    # Convert code columns to strings to ensure proper matching
    icd9_txt_df['ICD9code'] = icd9_txt_df['ICD9code'].astype(str)
    icd10_txt_df['ICD10code'] = icd10_txt_df['ICD10code'].astype(str)

    # Index the DataFrames on the code columns for fast lookup
    icd9_txt_df.set_index('ICD9code', inplace=True)
    icd10_txt_df.set_index('ICD10code', inplace=True)

    # After setting the index with the ICD codes
    icd9_txt_df.index = icd9_txt_df.index.astype(str).map(format_icd_code)
    icd10_txt_df.index = icd10_txt_df.index.astype(str).map(format_icd_code)

    print("Loading CPT codes.")

    # Load the CPT codes into a DataFrame
    cpt_codes_df, cpt_txt_df = load_cpt_codes('data/2024_DHS_Code_List_Addendum_03_01_2024.txt', 'data/CPT.txt')

    print("Loading data splits")
    # Load the preprocessed tensor
    loaded_train_dataset = torch.load('train_dataset.pt')
    loaded_validation_dataset = torch.load('validation_dataset.pt')
    loaded_test_dataset = torch.load('test_dataset.pt')

    # Load column names
    with open('column_names.json', 'r') as file:
        column_names = json.load(file)

    # Convert to DataFrames for analysis
    df_train = pd.DataFrame(loaded_train_dataset.tensors[0].numpy(), columns=column_names)
    df_validation = pd.DataFrame(loaded_validation_dataset.tensors[0].numpy(), columns=column_names)
    df_test = pd.DataFrame(loaded_test_dataset.tensors[0].numpy(), columns=column_names)

    # Print dimensions of datasets
    print("Training Data Dimensions:", df_train.shape)
    print("Validation Data Dimensions:", df_validation.shape)
    print("Test Data Dimensions:", df_test.shape)

    # Create the "data_summaries" folder if it doesn't exist
    os.makedirs('data_summaries', exist_ok=True)

    # Get the description of the training dataset
    train_description = df_train.describe()

    # Save the description to an HTML file
    with open('data_summaries/train_dataset_description.html', 'w') as f:
        f.write(train_description.to_html())

    # Calculate sparsity rate for one-hot encoded columns

    with open('encoded_feature_names.json', 'r') as file:
        encoded_feature_names = json.load(file)

    one_hot_sparsity_rate = 1-df_train[encoded_feature_names].mean()
    print("One-hot sparsity rate (training set): ", one_hot_sparsity_rate)

    # Limit the number of features or samples due to visual and performance constraints
    sampled_df = df_train[encoded_feature_names].sample(n=min(100, len(df_train)), axis=1, random_state=42) # limits to 100 features

    # Calculate sparsity rate directly for features in sampled_df
    sparsity_rate_sampled = 1 - sampled_df.mean()
    print("One-hot sparsity rate (sampled filtered training set): ", sparsity_rate_sampled)

    # Plotting the sparsity rate for one-hot encoded features in sampled_df
    plt.figure(figsize=(15, 8))
    sns.barplot(x=sparsity_rate_sampled.index, y=sparsity_rate_sampled.values)
    plt.title("Sparsity Rate per One-Hot Encoded Feature in Training Set (100 Sampled Features)")
    plt.xlabel("Feature")
    plt.ylabel("Sparsity Rate")
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()  # Adjust layout
    plt.savefig("data_summaries/one_hot_sparsity_rate_sampled_plot.png")  # Save plot to file

    df_train_summary = df_train[encoded_feature_names].agg(['mean', 'std', 'sum']).T

    df_train_summary['Description'] = df_train_summary.index.map(
        lambda code: get_icd_description(
            code.split('_')[0],
            code.split('_')[1],
            icd9_df,
            icd10_df,
            icd9_txt_df,
            icd10_txt_df,
            cpt_codes_df,
            cpt_txt_df
        )
    )

    # Then you can sort by 'mean' and proceed with the mapping for descriptions.
    df_train_summary_sorted = df_train_summary.sort_values(by='mean', ascending=False)

    # Save as csv
    df_train_summary.to_csv('data_summaries/df_train_summary_sorted.csv')

    # Save to HTML
    # Assuming `df_train_summary_sorted` is your final DataFrame
    title = "Summary statistics of one-hot encoded diagnoses and procedures in training data"
    #subtitle = "(n=36439, 704 procedures + 982 diagnoses = 1686 one-hot encoded features)"

    # Add HTML for the title
    html_title = f"<h2>{title}</h2>"
    #html_subtitle = f"<h3>{subtitle}</h3>"

    # Convert DataFrame to HTML
    df_train_summary_html = df_train_summary_sorted.to_html()

    # Combine the title HTML with the DataFrame HTML
    full_html = html_title + df_train_summary_html #+ html_subtitle 

    # Write the combined HTML to a file
    with open('data_summaries/df_train_summary_statistics.html', 'w') as f:
        f.write(full_html)

    # Transpose the DataFrame for the heatmap display (features on y-axis, samples on x-axis)
    transposed_df = sampled_df.transpose()

    # Create the heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(transposed_df, cmap="viridis", cbar_kws={'label': 'Feature Presence (1) or Absence (0)'})
    plt.title("Heatmap of One-Hot Encoded Feature Sparsity in Training Set (100 Sampled Features)")
    plt.xlabel("Sample Index")
    plt.ylabel("One-Hot Encoded Feature")

    # Optional: Adjust the layout and aspect ratio for better visibility depending on the number of features
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("data_summaries/one_hot_encoded_features_sparsity_sampled_heatmap.png")

    if use_labs:
        # Assuming df_train, df_validation, and df_test are the DataFrames containing the loaded datasets
        print("Analyzing labs data")

        # Filter columns starting with "Code_" and "Result"
        code_columns = [col for col in df_train.columns if col.startswith("Code_")]
        result_columns = [col for col in df_train.columns if col.startswith("Result")]

        # Melt the DataFrame to convert code and result columns into rows
        melted_train_df = pd.melt(df_train, value_vars=code_columns + result_columns, var_name='Column', value_name='Value')

        # Extract code and result from the 'Column' column
        melted_train_df['Code'] = melted_train_df['Column'].apply(lambda x: x.split('_')[1] if x.startswith('Code_') else '')
        melted_train_df['Result'] = melted_train_df['Value'].where(melted_train_df['Column'].isin(result_columns), '')

        # Drop unnecessary columns
        melted_train_df = melted_train_df.drop(['Column', 'Value'], axis=1)

        # Remove rows with missing code or result
        melted_train_df = melted_train_df.dropna(subset=['Code', 'Result'])

        # Group by 'Code' and calculate mean and standard deviation of 'Result'
        labs_summary = melted_train_df.groupby('Code')['Result'].agg(['mean', 'std']).reset_index()

        # Sort by mean in descending order
        labs_summary_sorted = labs_summary.sort_values(by='mean', ascending=False)

        # Add a 'Description' column using the get_icd_description function
        labs_summary_sorted['Description'] = labs_summary_sorted['Code'].apply(lambda code: get_icd_description(code, 'CPT', icd9_df, icd10_df, icd9_txt_df, icd10_txt_df, cpt_codes_df, cpt_txt_df))

        # Save the sorted summary to a CSV file
        labs_summary_sorted.to_csv('labs_summary_sorted.csv', index=False)

        # Create an HTML table
        title = "Summary statistics of lab results"
        subtitle = f"(n={melted_train_df.shape[0]} lab results, {labs_summary_sorted.shape[0]} unique lab codes)"

        html_title = f"<h2>{title}</h2>"
        html_subtitle = f"<h3>{subtitle}</h3>"

        labs_summary_html = labs_summary_sorted.to_html(index=False)

        full_html = html_title + html_subtitle + labs_summary_html

        # Write the HTML table to a file
        with open('data_summaries/labs_summary_sorted.html', 'w') as f:
            f.write(full_html)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_labs', action='store_true', help='Use labs data')
    args = parser.parse_args()

    main(use_labs=args.use_labs)