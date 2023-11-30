"""
This module provides a command-line interface for running the
Ramdom Forest Model on a specified file,
allowing users to analyze and classify data with the trained model.
"""
import os
import sys
import argparse
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

# Import the StartupPredictor class from mymodule.class
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
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the saved Random Forest model weights (joblib file)',
    )

    # Add a command-line argument for the input data file path
    parser.add_argument(
        'input_data_path',
        type=str,
        help='Path to the CSV file containing input data for predictions',
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create an instance of the StartupPredictor class
    startup_predictor = StartupPredictor(args.model_path)

    # Preprocess the input data
    preprocessed_data: pd.DataFrame = startup_predictor.preprocess_input(args.input_data_path)

    # Make predictions
    predictions: pd.DataFrame = startup_predictor.predict(preprocessed_data)

    # Display predictions
    print("Predictions using Random Forest model:")
    print(predictions)


if __name__ == '__main__':
    main()
