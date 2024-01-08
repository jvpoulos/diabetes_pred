from sklearn.feature_extraction.text import TfidfVectorizer

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

# Apply NLP Feature Extraction (skip for comments)
# text_columns = ['Comments']
# merged_df = extract_nlp_features(merged_df, text_columns)
