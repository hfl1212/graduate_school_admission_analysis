"""
Implements all functions needed for building a Lasso regression
model and for analyzing the weight of each feature.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import math


def standardize(v):
    """
    Takes a single column of a DataFrame and returns a new column
    with the data standardized (mean 0, std deviation 1)
    """
    std = v.std()
    if std == 0:
        return np.zeros(len(v))
    else:
        return (v - v.mean()) / std


def print_coefficients(model, features):
    """
    This function takes in a model column and a features column
    and prints the coefficient along with its feature name.
    """
    feats = list(zip(model.coef_, features))
    print(*feats, sep="\n")


def lasso_regression(origin_data):
    """
    Performs lasso regression on given data set and extracts the weight of
    features.
    """
    data_process = origin_data.copy()

    # Show the total number of data points
    print(f'Total Number of data points: {len(origin_data)} \n')

    # All of the features of interest
    features = [
        'GRE Score',
        'TOEFL Score',
        'University Rating',
        'SOP',
        'LOR',
        'CGPA',
        'Research'
    ]

    # The target feature
    target = 'Chance of Admit'

    # Standardize input features (mean 0, std. dev. 1) and center the
    # target values (mean 0)
    # Standardize each of the features
    for feature in features:
        data_process[feature] = standardize(data_process[feature])

    # Make the chance of admit have mean 0
    mean_admit = data_process[target].mean()
    data_process[target] -= mean_admit

    # Visualize preprocessed data
    print('Preprocessed data:')
    print(data_process.head())
    print('\n')

    # Split the data set into training, validation, and test sets.
    # Use 70% of the data to train, 15% for validation, and 15% to test.
    train_and_validation, test = train_test_split(data_process,
                                                  test_size=0.15)
    train, validation = train_test_split(train_and_validation,
                                         test_size=0.15)
    print(f'Number of training points: {len(train)}')
    print(f'Number of validation points: {len(validation)}')
    print(f'Number of testing points: {len(test)} \n')

    # Apply grid search to find the best hyper-parameter for lasso regression
    lasso_params = {'alpha': [i/100 for i in range(1, 10)]}
    search = GridSearchCV(Lasso(), param_grid=lasso_params, cv=6,
                          return_train_score=True)
    search.fit(train[features], train[target])
    print('GridSearch result:')
    print(search.best_params_)
    print('\n')
    cv_results = search.cv_results_
    scores = cv_results['mean_test_score']
    print('GridSearch test scores: ' + str(scores) + '\n')

    # Evaluate LASSO Regression with various L1 penalties by hand
    penalties = [i/100 for i in range(1, 10)]
    data = []

    for a in penalties:
        lasso_model = Lasso(alpha=a)
        model = lasso_model.fit(train[features], train[target])
        train_rmse = math.sqrt(mean_squared_error(train[target],
                               model.predict(train[features])))
        validation_rmse = math.sqrt(mean_squared_error(validation[target],
                                    model.predict(validation[features])))
        data.append({'penalty': a, 'model': model, 'train_rmse': train_rmse,
                     'validation_rmse': validation_rmse})

    lasso_data = pd.DataFrame(data)
    print(lasso_data)
    print('\n')

    # Train a lasso model with the best hyper-parameter and calculate the
    # mean squared test error
    lasso_model = Lasso(alpha=0.01)
    model = lasso_model.fit(train[features], train[target])
    test_rmse = math.sqrt(mean_squared_error(test[target],
                                             model.predict(test[features])))
    print('mean squared test error: ' + str(test_rmse) + '\n')

    # Visualize the weight of each feature
    print('Weight of each feature:')
    print_coefficients(model, features)
