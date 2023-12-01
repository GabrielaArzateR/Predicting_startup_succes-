

STARTUP SUCCED PREDICTION

This repository solves the STARTUP SUCCED PREDICTION . It showcases a model training, evaluation.

Installation

Tested with:

Create a new environment and install the Python requirements:
pip install -r requirements.txt

Demonstration Notebook

This Jupyter notebook contains detailed code examples, along with visualizations, to showcase the functionality and features of the project. It serves as a useful guide for understanding and interacting with the project's components.

Project Structure

You can find the structure of the project below:
















The most important components are the following:

data: Contains the training and test data, organized into subdirectories for different classes.
scripts: Contains standalone Python scripts for training, evaluation, relabeling, and image similarity search.
service.py: Code for defining the BentoML service and APIs.
terminator_classifier: The main module, encompassing core functionality, data processing, and machine learning components.
terminator_classifier/tests: Includes test scripts for automated testing as part of Continuous Integration (CI).


Testing

The repository employs Continuous Integration (CI) to ensure code quality and functionality. The CI is defined in .github/workflows/main-ci.yml and triggers on pull requests to the main branch. It runs various checks including:

unit tests with Pytest
code quality assessment with Pylint
Black for code style
Bandit for security vulnerabilities
Mypy for static type checking.


Features

Model Training
The model training is facilitated by the train.py script, which trains a ResNet50 model on image classification data. To execute the training script, use the following command:


Model Evaluation


Conclusion



