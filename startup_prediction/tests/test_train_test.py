"""
This module contains test cases for the Random Forest Model.
"""
# When you want to run this file change the debugging to pytest otherwise you will have
# an error.
import sklearn
from startup_prediction.script import StartupPredictor


def test_startup_predictor() -> None:
    """Test the basic functionality of StartupPredictor class"""

    # Replace 'your_model.joblib' with the actual path to your saved model weights
    model_file_path: str = 'data/best_model.joblib'

    # Replace 'input_data.csv' with the actual path to your input data file
    input_data_path: str = 'data/startup.csv'

    # Create an instance of the StartupPredictor class
    startup_predictor = StartupPredictor(model_file_path)

    # Preprocess the input data and make predictions
    predictions = startup_predictor.predict(startup_predictor.preprocess_input(input_data_path))

    assert predictions is not None, "Predictions should not be None"  # nosec
    assert len(predictions) > 0, "Predictions should not be empty"  # nosec


if __name__ == '__main__':
    test_startup_predictor()
    print("Test completed.")


print(sklearn.__version__)
