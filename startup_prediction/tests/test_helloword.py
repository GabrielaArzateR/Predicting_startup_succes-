"""
Dummy module description
"""

from startup_prediction.script import StartupPredictor


def test_startup_predictor() -> None:
    """Test the StartupPredictor class with a specified data path"""
    data_path = "data/startup.csv"
    StartupPredictor(data_path)


if __name__ == '__main__':
    test_startup_predictor()
