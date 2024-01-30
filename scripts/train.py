"""
This module provides a command-line interface for running the
Ramdom Forest Model on a specified file,
allowing users to analyze and classify data with the trained model.
"""

import os
import sys
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

# Import the StartupPredictor class from mymodule.class
# pylint: disable=wrong-import-position
from startup_prediction.script import (  # pylint: disable=wrong-import-position
    StartupPredictor,
)


def main() -> None:
    """
    This function parses command-line arguments and runs predictions using the Random Forest model.

    It reads a CSV file from the providesd path, preprocesses the data,
    and uses the model to make predictions.

    Returns:
        None

    Example:
    To run predictions using the Random Forest model, use:
    >>> main()
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Random Forest Predictions")

    # Add a command-line argument for the model file path
    # Add a command-line argument for the input data file path
    parser.add_argument(
        'input_data_path',
        type=str,
        help='Path to the CSV file containing input data for predictions',
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # You can call your train_test function here with the data object
    startup_predictor = StartupPredictor(
        'data/best_model.joblib'
    )  # Replace with your actual model path
    preprocessed_data = startup_predictor.preprocess_input(args.input_data_path)
    # pylint: disable=protected-access
    predictions = startup_predictor._predict(preprocessed_data)
    print(predictions)


if __name__ == '__main__':
    main()
