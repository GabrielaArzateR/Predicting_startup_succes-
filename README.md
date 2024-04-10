
# Startup Success Predictor

Welcome to the Startup Success Predictor! This project uses data to predict how likely a startup is to succeed. Here, we train models, evaluate them, and make predictions to give you insights into a startup's potential success.

## Installation

Follow these steps to set up the Startup Success Predictor on your local machine:

### Prerequisites

Make sure you have the following installed:
- Python (version 3.x)

### Clone the Repository

```bash
git clone https://github.com/GabrielaArzateR/startup-success-predictor.git
cd startup-success-predictor

Create a new environment and install the Python requirements:
```
pip install -r requirements.txt

### Demonstration Notebook
>>>>>>> main

This Jupyter notebook contains detailed code examples, along with visualizations, to showcase the functionality and features of the project. It serves as a useful guide for understanding and interacting with the project's components.

## Project Structure
You can find the structure of the project below:
```bash
.
├── README.md
├── data
│   ├── best_model.joblib
│   └── startup.csv
├── notebook
│   └── experimentation.ipynb
├── pyproject.toml
├── requirements.txt
├── scripts
│   └── train.py
├── setup.py
└── startup_prediction
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── __init__.cpython-312.pyc
    │   ├── preprocessing.cpython-310.pyc
    │   ├── preprocessing.cpython-312.pyc
    │   ├── script.cpython-310.pyc
    │   ├── script.cpython-312.pyc
    │   └── test_preprocessing.cpython-312-pytest-7.4.3.pyc
    ├── helloword.py
    ├── preprocessing.py
    ├── script.py
    ├── test_preprocessing.py
    └── tests
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-312.pyc
        │   └── test_train_test.cpython-312-pytest-7.4.3.pyc
        └── test_train_test.py
```
The most important components are the following:
- `data`: Contains the training and test data, organized into subdirectories for different classes.
- `scripts`: Contains standalone Python scripts for training, evaluation, relabeling, and image similarity search.
- `service.py`: Code for defining the BentoML service and APIs.
- `startup_prediction`: The main module, encompassing core functionality, data processing, and machine learning components.
- `startup_prediction/tests`: Includes test scripts for automated testing as part of Continuous Integration (CI).

## Testing
The repository employs Continuous Integration (CI) to ensure code quality and functionality. The CI is defined in `.github/workflows/main-ci.yml` and triggers on pull requests to the main branch. It runs various checks including:
- unit tests with Pytest
- code quality assessment with Pylint
- Black for code style
- Bandit for security vulnerabilities
- Mypy for static type checking.

## Features











### Model Training
The model training is facilitated by the train.py script, which trains a ResNet50 model on image classification data. To execute the training script, use the following command:
```bash
python scripts/train.py --input_dir data \
    --output_dir train_logs \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --val_ratio 0.2
```

You can run the API version with (after serving the Bento or the Docker container):
```bash
curl -X POST \
    -H "content-type: application/json" \
    --data '{"epochs": 5, "batch_size": 32, "val_ratio": 0.2, "learning_rate": 1e-4}' \
    http://localhost:3000/train
```

### Model Evaluation
To run the model evaluation script with explicit arguments, you can use the following bash command:
```bash
python scripts/evaluate.py --weights_path train_logs/best_weights.pth \
    --input_dir data \
    --batch_size 32
```

You can run the API version with (after serving the Bento or the Docker container):
```bash
curl -X POST \
    -H "content-type: application/json" \
    --data '{"weights_path": "train_logs/best_weights.pth", "batch_size": 32}' \
    http://localhost:3000/evaluate

```




