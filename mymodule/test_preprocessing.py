"""
Module Description:
This module contains unit tests for the preprocess_data function in
the mymodule.preprocessing module.

Note: Adjust the import statements and module names as needed based on
your actual project structure.
"""

import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

# Import preprocess_data from mymodule.preprocessing
from mymodule.preprocessing import (  # pylint: disable=wrong-import-position
    preprocess_data,
)


def test_preprocess_data(data_path: str) -> None:
    """
    Test the preprocess_data function with sample data.

    Args:
        data_path (str): Path to the sample data file.

    Returns:
        None
    """
    # Call the preprocess_data function
    preprocessed_data = preprocess_data(data_path)

    # Print the preprocessed data to inspect the result
    print(preprocessed_data.head())


# Replace 'data/startup.csv' with the path to your actual data file
# pylint: disable=C0103
data_file = 'data/startup.csv'

# Call the test_preprocess_data function
test_preprocess_data(data_file)
