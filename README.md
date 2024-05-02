# AUTO AI

## Description
- An automatic porgram for training ML models with sklearn package with numerical data.


## Installation
### Prerequisites
- python==3.11.5

- scikit-learn==1.2.2
- xgboost==1.7.3
- lightgbm==4.1.0
- mlxtend==0.23.0
- joblib==1.2.0
- numpy==1.26.0
- matplotlib==3.8.1
- seaborn==0.13.0
- pandas==2.1.1


### Installing
after creating a vitual enviroments
- conda create -n ENV_Name python==3.11.5
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
- Setup the package
- create an output folder 
- prepare the datasets
- run example.py or create your own
- remember to create correct pth for output and datasets
- change the traning parameter in example.py train_params_dt

## Deployment

## Built With

## Authors

## License

## Acknowledgments

## Contact Information

## Project Status
