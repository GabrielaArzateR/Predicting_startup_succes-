"""
Module Description:
This module contains code for working with pandas DataFrame.

Author: Your Name
Date: Date when the module was created or last updated
"""
import pandas as pd


def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Read the CSV file into a DataFrame
    # data: pd.DataFrame = pd.read_csv(file_path)
    data = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Map 'status' column to binary values
    data['status'] = data['status'].map({'acquired': 1, 'closed': 0})

    clean_data = data.drop(
        [
            'Unnamed: 0',
            'latitude',
            'longitude',
            'zip_code',
            'id',
            'Unnamed: 6',
            'name',
            'labels',
            'state_code.1',
            'is_CA',
            'is_NY',
            'is_MA',
            'is_TX',
            'is_otherstate',
            'is_software',
            'is_web',
            'is_mobile',
            'is_enterprise',
            'is_advertising',
            'is_gamesvideo',
            'is_ecommerce',
            'is_biotech',
            'is_consulting',
            'is_othercategory',
            'object_id',
            'status',
        ],
        axis=1,
    ).copy()

    # Handle Missing Values of 2 variables
    # 2. Impute missing values with 0 to 'age_first_milestone_year'
    clean_data['age_first_milestone_year'].fillna(0, inplace=True)
    # 3. Impute missing values with 0 to 'age_last_milestone_year'
    clean_data['age_last_milestone_year'].fillna(0, inplace=True)

    # Convert date columns to datetime
    date_columns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    for date_column in date_columns:
        clean_data[date_column] = pd.to_datetime(clean_data[date_column], errors='coerce')

    # Convert datetime to timestamp (numerical)
    for date_column in date_columns:
        clean_data[date_column] = clean_data[date_column].astype(int)

    # List of categorical columns
    categorical_columns = ['state_code', 'city', 'category_code']

    # Create a function to generate mappings
    def create_mapping(column: str) -> dict:
        unique_values = clean_data[column].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        return mapping

    # Apply mapping for each categorical column
    for column in categorical_columns:
        mapping = create_mapping(column)
        clean_data[column] = clean_data[column].map(mapping)

    return clean_data
