"""
mymodule/class.py

Module Description:
This module defines the StartupPredictor class for making predictions using a trained model.
"""

# Import Statements: You're importing modules from startup_prediction.preprocessing,
# which indicates that your script relies on external modules.
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

import argparse
from typing import Any
import pandas as pd
import joblib
from startup_prediction.preprocessing import preprocess_data


def load_model(model_path: str) -> Any:
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        Any: The loaded model.
    """
    return joblib.load(model_path)


def preprocess_input(data_path: str) -> pd.DataFrame:
    """
    Preprocess the input data.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed input data.
    """
    return preprocess_data(data_path)


def make_predictions(model: Any, processed_data: pd.DataFrame) -> Any:
    """
    Make predictions using the loaded model and preprocessed input data.

    Args:
        model (Any): Loaded model.
        processed_data (pd.DataFrame): Preprocessed input data.

    Returns:
        Any: Predictions made by the model.
    """
    return model.predict(processed_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest Predictions")

    parser.add_argument(
        '--model_path',
        type=str,
        default='saved_model/best_model.joblib',
        help="The path to a file to load the model.",
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/startup.csv',
        help="The path to a csv file containing the input to predict one.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.model_path)

    # Preprocess the input data
    processed_data = preprocess_input(args.input_csv)

    # Make predictions
    predictions = make_predictions(model, processed_data)

    print("Predictions:")
    print(predictions)


if __name__ == '__main__':
    main()
