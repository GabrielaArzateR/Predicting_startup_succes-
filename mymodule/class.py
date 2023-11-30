"""
mymodule/class.py

Module Description:
This module defines the StartupPredictor class for making predictions using a trained model.
"""
import os
import sys
from typing import Any
import pandas as pd
import joblib

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

# Import preprocess_data from mymodule.preprocessing
from mymodule.preprocessing import (  # pylint: disable=wrong-import-position
    preprocess_data,
)


class StartupPredictor:
    """
    StartupPredictor class for making predictions using a trained model.

    Args:
        model_path (str): Path to the saved model weights.

    Methods:
        preprocess_input(input_data): Preprocess input data.
        predict(input_data): Make predictions using the preprocessed input data.
    """

    def __init__(self, model_path: str):
        """
        Initialize the StartupPredictor.

        Args:
            model_path (str): Path to the saved model.
        """

        # Load the saved model
        self.model = joblib.load(model_path)

    def preprocess_input(self, file_path: str) -> pd.DataFrame:
        """
        Preprocess the input data.

        Args:
            input_data (pd.DataFrame): Input data to be preprocessed.

        Returns:
            pd.DataFrame: Preprocessed input data.
        """
        # Use the imported preprocess_data function
        processed_data = preprocess_data(file_path)

        return processed_data

    def predict(self, input_data: pd.DataFrame) -> Any:
        """
        Make predictions using the preprocessed input data.

        Args:
            input_data (pd.DataFrame): Preprocessed input data.

        Returns:
            Any: Predictions made by the model.
        """
        # Preprocess the input data
        processed_data = self.preprocess_input(input_data)

        # Make predictions
        predictions = self.model.predict(processed_data)

        return predictions


# Example of usage
if __name__ == "__main__":
    # Replace 'your_model.joblib' with the actual path to your saved model weights
    model_file_path: str = (
        '/Users/gabrielaarzate/Desktop/predicting_startup_succes/data/best_model.joblib'
    )

    # Create an instance of the StartupPredictor class
    startup_predictor = StartupPredictor(model_file_path)

    # Replace 'input_data.csv' with the actual path to your input data file
    input_data_path: str = (
        '/Users/gabrielaarzate/Desktop/predicting_startup_succes/data/startup.csv'
    )
    # Preprocess the input data
    preprocessed_data: pd.DataFrame = startup_predictor.preprocess_input(input_data_path)

    # Make predictions
    startup_predictions: Any = startup_predictor.predict(preprocessed_data)

    # Display predictions
    print(startup_predictions)
