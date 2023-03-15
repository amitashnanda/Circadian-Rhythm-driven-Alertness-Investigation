import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum
from math import pi
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler 
from sklearn.metrics import mean_absolute_error
import sklearn
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load the data from the file path specified.

    Parameters:
    file_path (str): The path to the data file.

    Returns:
    DataFrame: The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    data['dst'] = data['dst'].astype('object')
    data['weekend'] = data['weekend'].astype('object')
    data = data.drop(['family_id', 'username'], axis = 1)
    return data

def normalize_data(data, categorical_attributes, numerical_attributes):
    """
    Normalize the given data by applying one-hot encoding to categorical columns
    and standard scaling to numerical columns.

    Parameters:
    data (DataFrame): The data to be normalized.
    categorical_attributes (list): A list of categorical column names.
    numerical_attributes (list): A list of numerical column names.

    Returns:
    DataFrame: The normalized data as a pandas DataFrame.
    """
    # transform non-numeric parameter to one-hot encoding 
    data = pd.get_dummies(data, dummy_na=False)
    data_skew = data.drop(categorical_attributes, axis=1)

    # Normalize numerical columns
    ct = make_column_transformer(
        (StandardScaler(), numerical_attributes)
    )
    data_normalized = pd.DataFrame(ct.fit_transform(data_skew))

    # Rename columns to match original data
    col_dict = dict(zip(data_normalized.columns, numerical_attributes))
    data_normalized = data_normalized.rename(columns=col_dict)

    # Concatenate categorical and numerical columns
    data_normalized = pd.concat([data[categorical_attributes], data_normalized], axis=1)

    return data_normalized

def train_models(X_train, y_train):
    """
    Train several regression models on the given training data.

    Parameters:
    X_train (DataFrame): The input features for training.
    y_train (Series): The target variable for training.

    Returns:
    list: A list of tuples containing model names and corresponding trained models.
    """
    # Make dictionary of models
    models = {
        'SVR': SVR(),
        'XGBRegressor': XGBRegressor(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'SGDRegressor': SGDRegressor(),
        'BayesianRidge': BayesianRidge(),
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor()
    }

    # Train models and store results
    trained_models = []
    for name, model in models.items():
        trained_model = model.fit(X_train, y_train)
        trained_models.append((name, trained_model))

    return trained_models

def predict_data(X_test, trained_models):
    """
    Use a list of trained models to make predictions on new test data.

    Parameters:
    X_test (DataFrame): The input features for prediction.
    trained_models (list): A list of tuples containing (model_type, model_instance).

    Returns:
    y_pred (array-like): Predicted labels for the input data.
    """
    # Create a list to store predictions from each model
    model_predictions = []
    
    # Make predictions for each trained model
    for model_type, model_instance in trained_models:
        # Use the appropriate method to make predictions based on the model type
        if model_type == 'SVR':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'XGBRegressor':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'Ridge':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'ElasticNet':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'SGDRegressor':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'BayesianRidge':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'LinearRegression':
            y_pred = model_instance.predict(X_test)
        elif model_type == 'RandomForestRegressor':
            y_pred = model_instance.predict(X_test)
        else:
            raise ValueError(f"Invalid model type '{model_type}'")
        
        # Append the predictions to the list of model predictions
        model_predictions.append(y_pred)
    
    # Take the mean of the predictions across all models to get the final prediction
    y_pred = np.mean(model_predictions, axis=0)
    
    return y_pred
