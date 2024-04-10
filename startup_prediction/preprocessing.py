"""
Module Description:
This module contains code for working with pandas DataFrame.

Author: Your Name
Date: 5/Apr/ 2024
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    #### Read the CSV file into a DataFrame
    dataframe = pd.read_csv(file_path, encoding="ISO-8859-1")

    #### Dropping variables
    dataframe = dataframe.drop(
        [
            'Unnamed: 0',  # Irrelevant data
            'latitude',  # Irrelevant data
            'longitude',  # Irrelevant data
            'zip_code',  # Irrelevant data
            'id',  # Irrelevant data
            'Unnamed: 6',  # Irrelevant data
            'name',  # Irrelevant data
            'labels',  # Same data as status
            'state_code.1',  # Same data as state_code
            'is_CA',  # Binary column, not needed
            'is_NY',  # Binary column, not needed
            'is_MA',  # Binary column, not needed
            'is_TX',  # Binary column, not needed
            'is_otherstate',  # Binary column, not needed
            'is_software',  # Binary column, not needed
            'is_web',  # Binary column, not needed
            'is_mobile',  # Binary column, not needed
            'is_enterprise',  # Binary column, not needed
            'is_advertising',  # Binary column, not needed
            'is_gamesvideo',  # Binary column, not needed
            'is_ecommerce',  # Binary column, not needed
            'is_biotech',  # Binary column, not needed
            'is_consulting',  # Binary column, not needed
            'is_othercategory',  # Binary column, not needed
            'object_id',
        ],  # Irrelevant data
        axis=1,
    ).copy()

    #### Handle Missing Values of 2 variables
    # 2. Impute missing values with Fill() Methode mean
    dataframe["age_first_milestone_year"] = dataframe["age_first_milestone_year"].fillna(
        dataframe["age_first_milestone_year"].mean()
    )
    dataframe["age_last_milestone_year"] = dataframe["age_last_milestone_year"].fillna(
        dataframe["age_last_milestone_year"].mean()
    )

    # First convert the following columns into datetime format
    dataframe.founded_at = pd.to_datetime(dataframe.founded_at)
    dataframe.first_funding_at = pd.to_datetime(dataframe.first_funding_at)
    dataframe.last_funding_at = pd.to_datetime(dataframe.last_funding_at)

    # Outlier Analysis
    dataframe['diff_founded_first_funding'] = (
        dataframe["first_funding_at"].dt.year - dataframe["founded_at"].dt.year
    )
    dataframe['diff_founded_last_funding'] = (
        dataframe["last_funding_at"].dt.year - dataframe["founded_at"].dt.year
    )

    dataframe['negative_first_funding'] = (dataframe["age_first_funding_year"] < 0).astype(int)
    dataframe['negative_last_funding'] = (dataframe["age_last_funding_year"] < 0).astype(int)
    dataframe['negative_first_milestone'] = (dataframe["age_first_milestone_year"] < 0).astype(int)
    dataframe['negative_last_milestone'] = (dataframe["age_last_milestone_year"] < 0).astype(int)

    #### Handle Negative Values
    # Use abs function
    dataframe["age_first_funding_year"] = np.abs(dataframe["age_first_funding_year"])
    dataframe["age_last_funding_year"] = np.abs(dataframe["age_last_funding_year"])
    dataframe["age_first_milestone_year"] = np.abs(dataframe["age_first_milestone_year"])
    dataframe["age_last_milestone_year"] = np.abs(dataframe["age_last_milestone_year"])

    ###Identify_outliers
    threshold = 4
    column_name = "age_first_funding_year"
    column = dataframe[column_name]
    mean = column.mean()
    std = column.std()
    is_outlier = np.abs(column - mean) > threshold * std
    dataframe['is_outlier'] = is_outlier.astype(int)

    #### Handle Outliers
    # Define the list of features to be log-transformed
    age_features = [
        "age_first_funding_year",
        "age_last_funding_year",
        "age_first_milestone_year",
        "age_last_milestone_year",
    ]

    # Apply log transformation to the selected features
    # Create a figure with a specific size
    for variable in age_features:
        log_variable = np.log(dataframe[variable] + 1)
        print(f"log_{variable}: {log_variable}")

    #### Creating Variables
    # All calculations
    dataframe = dataframe

    # We convert the 'closed_at' column to datetime format, handling any conversion errors.
    dataframe['closed_at'] = pd.to_datetime(dataframe['closed_at'], errors='coerce')

    # Sort DataFrame by 'closed_at' in descending order revealing the most recent startup closures.
    dataframe = dataframe.sort_values(by='closed_at', ascending=False)

    # Find the last closing date
    last_closed_date = dataframe['closed_at'].dropna().iloc[0]

    # Confirm is the right value
    print("Last startup closing date:", last_closed_date)

    # temporary variable, 'closed_temp,' is created to preserve non-null values of the 'closed_at' column for calculations.
    closed_temp = dataframe['closed_at'].copy()

    # Fill the null values in 'closed_temp' with the last closed date(2013-10-30)
    closed_temp.fillna(last_closed_date, inplace=True)

    # Calculate the relative age based on 'founded_at' and 'closed_temp'
    dataframe['age'] = ((closed_temp - dataframe['founded_at']).dt.days / 365.25).round(4)

    # Missing values in 'closed_at' are replaced with 'x' to signify operating startups.
    dataframe['closed_at'] = dataframe['closed_at'].fillna(value="x")

    #'Missing values in 'closed_at' are replaced with 'x' to signify operating startups.
    dataframe['closed_at'] = dataframe.closed_at.apply(lambda x: 1 if x == 'x' else 0)

    # Convert 'founded_at' column to datetime objects
    dataframe['founded_at'] = pd.to_datetime(dataframe['founded_at'])

    # Extract and format the year from 'founded_at' as 'founded_year'
    dataframe['founded_year'] = dataframe['founded_at'].dt.strftime('%Y')

    # Group the DataFrame by 'founded_year' and count occurrences
    prop_df = dataframe.groupby('founded_year').size().reset_index(name='counts')

    # Calculate the proportions of startups founded each year
    prop_df['proportions'] = prop_df['counts'] / prop_df['counts'].sum()

    #### Transforming Non-numeric Data into Numeric Values
    categorical_columns = ['state_code', 'city', 'category_code', 'founded_year', 'status']
    # Dictionary to store mappings
    column_mappings = {}

    # Create a function to generate mappings
    from typing import Dict, Any

    # Create a function to generate mappings
    def create_mapping(column: str) -> Dict[Any, int]:
        unique_values = dataframe[column].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        return mapping

    # Apply mapping for each categorical column
    for column in categorical_columns:
        mapping = create_mapping(column)
        dataframe[column] = dataframe[column].map(mapping)
        # Save mapping in the dictionary
        column_mappings[column] = mapping

    #### Converting Datatime features
    # Set a reference date. You could choose the earliest date in your dataset or a specific date.
    reference_date = dataframe[['founded_at', 'first_funding_at', 'last_funding_at']].min().min()

    # List of columns to convert
    columns_to_convert = ['founded_at', 'first_funding_at', 'last_funding_at']

    # Convert each specified column into days since the reference date
    for column in columns_to_convert:
        dataframe[f'{column}_days'] = (dataframe[column] - reference_date).dt.days

    # Now, 'founded_at_days', 'first_funding_at_days', and 'last_funding_at_days' are numerical
    # columns representing the number of days from the reference date.
    # 'closed_at' remains unchanged as it is already in numerical format (int64).

    #### Optionally, you might drop the original datetime columns if they're not needed anymore.
    dataframe.drop(['founded_at', 'first_funding_at', 'last_funding_at'], axis=1, inplace=True)

    ##  Ignore: (Changes for modeling)
    dataframe.sort_index().head(2)
    dataframe = dataframe.drop(['closed_at'], axis=1).copy()
    dataframe = dataframe.drop(['age'], axis=1).copy()
    numerical_features = dataframe.select_dtypes(include=['number']).columns.tolist()
    categorical_features = dataframe.select_dtypes(include=['object']).columns.tolist()
    datetime_features = dataframe.select_dtypes(include=['datetime']).columns.tolist()

    print("\nColumn Names in processed_data:")
    print(dataframe.columns)

    print("\nSample Rows in processed_data:")
    print(dataframe.head(32))

    inputs = dataframe.drop('status', axis=1)
    target = dataframe['status']

    x_train, x_test, y_train, y_test = train_test_split(
        inputs, target, test_size=0.2, random_state=42
    )
    return dataframe
