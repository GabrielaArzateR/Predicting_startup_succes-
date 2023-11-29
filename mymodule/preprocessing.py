"""
Module Description:
This module contains code for working with pandas DataFrame.

Author: Your Name
Date: Date when the module was created or last updated
"""
import pandas as pd


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Read the CSV file into a DataFrame
    data: pd.DataFrame = pd.read_csv(file_path)

    # Map 'status' column to binary values
    data['status'] = data['status'].map({'acquired': 1, 'closed': 0})

    # Convert date columns to datetime
    date_columns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    for date_column in date_columns:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    # Convert datetime to timestamp (numerical)
    for date_column in date_columns:
        data[date_column] = data[date_column].astype(int)

    # List of categorical columns
    categorical_columns = ['state_code', 'city', 'category_code']

    # Create a function to generate mappings
    def create_mapping(column: str) -> dict:
        unique_values = data[column].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        return mapping

    # Apply mapping for each categorical column
    for column in categorical_columns:
        mapping = create_mapping(column)
        data[column] = data[column].map(mapping)

    return data
