"""
This module contains test cases for the Random Forest Model.
"""

from startup_prediction.script import StartupPredictor


def test_startup_predictor() -> None:
    """Test the basic functionality of StartupPredictor class"""

    # Replace 'your_model.joblib' with the actual path to your saved model weights
    model_file_path: str = (
        '/Users/gabrielaarzate/Desktop/predicting_startup_succes/data/best_model.joblib'
    )

    # Replace 'input_data.csv' with the actual path to your input data file
    input_data_path: str = (
        '/Users/gabrielaarzate/Desktop/predicting_startup_succes/data/startup.csv'
    )

    # Create an instance of the StartupPredictor class
    startup_predictor = StartupPredictor(model_file_path)

    # Preprocess the input data and make predictions
    predictions = startup_predictor.predict(startup_predictor.preprocess_input(input_data_path))

    # Ensure that predictions are not None or empty, depending on your expectations
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) > 0, "Predictions should not be empty"


if __name__ == '__main__':
    test_startup_predictor()
