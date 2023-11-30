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

    def preprocess_input(self, data_path: str) -> pd.DataFrame:
        """
        Preprocess the input data.

        Args:
            data_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Preprocessed input data.
        """
        # Use the imported preprocess_data function
        preprocessedd_data = preprocess_data(data_path)

        return preprocessedd_data

    def predict(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the preprocessed input data.

        Args:
            processed_data (pd.DataFrame): Preprocessed input data.

        Returns:
            Any: Predictions made by the model.
        """
        # Make predictionss
        predictions = self.model.predict(processed_data)

        return predictions

    def _predict(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Define the reverse mapping for status labels
        reverse_status_mapping = {1: 'acquired', 0: 'closed'}

        # Step 2: Use the model to make numeric predictions
        numeric_predictions = self.model.predict(processed_data)

        # Step 3: Map numeric predictions to labels
        label_predictionss = pd.Series(numeric_predictions).map(reverse_status_mapping)

        # Step 4: Create a DataFrame with the label predictions
        result_df = pd.DataFrame({'prediction': label_predictionss})

        # Step 5: Concatenate the label predictions with the original processed_data
        result_df = pd.concat([processed_data, result_df], axis=1)

        return result_df


# Example of usages
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

    # Display predictions from the 'predict' method
    print("Predictions using 'predict' method:")
    print(startup_predictions)

    # Make label predictions using the protected '_predict' method
    # pylint: disable=protected-access
    label_predictions: pd.DataFrame = startup_predictor._predict(preprocessed_data)

    # Display label predictions from the '_predict' method
    print("\nLabel Predictions using '_predict' method:")
    print(label_predictions)
