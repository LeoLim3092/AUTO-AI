# AUTO AI

## Description
- An automatic porgram for training ML models with sklearn package with numerical data.


## Installation
### Prerequisites
- scikit-learn==1.1.3
- xgboost==1.7.4
- lightgbm==3.4.0
- mlxtend==0.22.0
- joblib==1.2.0
- numpy==1.25.0
- matplotlib==3.7.0
- seaborn==0.12.0
- pandas==1.6.0

### Installing
after creating a vitual enviroments
- pip install -r ./requirements.txt

## Usage
This program is designed to handle inputs in the form of 2D matrices utilizing either Pandas or NumPy libraries. Each row in the matrix represents a sample, and each column represents a feature. It is important to note that the program does not support time-series data.


## Features
- Hyperparameter Tuning: The program includes a built-in grid search method to optimize hyperparameters. Ensure that the search ranges in the grid_search_dt.json file are correctly updated to reflect the parameters you wish to tune.
- Feature Selection: Implements a sequential forward feature selection method to identify the most significant features for model training.
- Model Training Options: Users can choose among several model validation methods, including leave-one-out, cross-validation, or a simple train-test split, to train the model effectively.


## Contributing
Contribution guidelines

## Tests
How to run tests

## Deployment

## Built With

## Authors

## License

## Acknowledgments

## Contact Information

## Project Status
