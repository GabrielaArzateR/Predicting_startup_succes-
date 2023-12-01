# Terminator Challenge
This repository solves the Terminator Challenge. It showcases a model training, evaluation, image similarity search and relabelling as well as many extra bonuses ;)

## Installation
Tested with:
- Ubuntu 22.04
- Docker 24.0.4 (mandatory to containerize the code)
- Python >= 3.10.12 (mandatory for typing)
- Cuda Toolkit 12.1
- SLI 2080 Ti


Create a new environment and install the Python requirements:
```
pip install -r requirements.txt
```

Make sure to install Docker

## Demonstration Notebook
For a practical demonstration of the project's capabilities, please refer to the `demo.ipynb` notebook.

This Jupyter notebook contains detailed code examples, along with visualizations, to showcase the functionality and features of the project. It serves as a useful guide for understanding and interacting with the project's components.

## Project Structure
You can find the structure of the project below:
```bash
.
├── data
│   ├── test
│   │   ├── 0
│   │   └── 1
│   └── train
│       ├── 0
│       └── 1
├── scripts
│   ├── evaluate.py
│   ├── image_similarity.py
│   ├── relabel.py
│   └── train.py
├── service.py
├── setup.py
└── terminator_classifier
    ├── __init__.py
    ├── arguments.py
    ├── constants.py
    ├── data_loader.py
    ├── dataset.py
    ├── image_similarity_search.py
    ├── parse_training_data.py
    ├── relabel.py
    ├── scripts.py
    ├── test.py
    ├── tests
    │   ├── __init__.py
    │   ├── test_image_similarity.py
    │   ├── test_relabel.py
    │   └── test_train.py
    ├── train.py
    └── visualization.py
```
The most important components are the following:
- `data`: Contains the training and test data, organized into subdirectories for different classes.
- `scripts`: Contains standalone Python scripts for training, evaluation, relabeling, and image similarity search.
- `service.py`: Code for defining the BentoML service and APIs.
- `terminator_classifier`: The main module, encompassing core functionality, data processing, and machine learning components.
- `terminator_classifier/tests`: Includes test scripts for automated testing as part of Continuous Integration (CI).

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






## Conclusion
Overall, I've really enjoyed your challenge !
I hope that you appreciated all these cool features.
See you soon ;)



