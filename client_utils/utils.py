import pandas as pd
import numpy as np
from scipy.stats import linregress
import flwr as fl
from client_utils.file_controller import FileController
from client_utils.ModelEnum import ModelEnum
import xgboost as xgb
import json
def detect_timeseries_type(data, feature):
    # Extracting time index
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data = data[feature]
    time_index = np.arange(len(data))

    # Performing linear regression
    slope, intercept, _, _, _ = linregress(time_index, data)

    # Compute residuals
    residuals = data - (slope * time_index + intercept)

    # Calculate the variance of residuals
    residual_var = np.var(residuals)

    # Compute the mean of the time series data
    data_mean = np.mean(data)

    # Check if the residuals show a pattern (indicating multiplicative seasonality)
    if residual_var > data_mean:
        return "multiplicative"
    else:
        return "additive"


def log_transform(X, feature):
        """
        Apply natural logarithm transformation to the specified column of the input DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame.

            feature: feature used

        Returns:
            pd.DataFrame: The input DataFrame with logarithmically transformed column replacing the original one.
        """

        X_transformed = X.copy()
        min_val= min(X_transformed[feature])
        #shift the data to the positive side if column contains -ve values
        if min_val <= 0:
            X_transformed[feature] = X_transformed[feature]+(abs(min_val)+1)

        X_transformed[feature] = np.log(X_transformed[feature])
        X_transformed.dropna(inplace=True)
        return X_transformed


def get_model_weights(model):
    if hasattr(model, 'coef_'):
        weights = [model.coef_, model.intercept_]
        weights = [w.astype(np.float32).tobytes() for w in weights]
        return weights
    elif hasattr(model, 'coefs_'):
        weights =  model.coefs_ + model.intercepts_
        weights = [w.astype(np.float32).tobytes() for w in weights]
        return weights
    else:
        booster = model.get_booster()
        booster_json = booster.save_raw("json")
        print(type(booster_json))
        return [bytes(booster_json)]
# Define a function to set model weights
def set_model_weights(model, weights):
    if hasattr(model, 'coef_'):
        model.coef_ = weights[0]
        model.intercept_ = weights[1]
    elif hasattr(model, "coefs_"):
        coefs = weights[:len(model.coefs_)]
        intercepts = weights[len(model.coefs_):]
        model.coefs_ = coefs
        model.intercepts_ = intercepts
    elif hasattr(model, 'feature_importances_'):
        booster = xgb.Booster()
        booster.load_model(weights)
        model._Booster = booster
    return model
def weights_to_parameters(weights):
    return fl.common.Parameters(tensors= weights, tensor_type="numpy")

# Convert Parameters object to model weights
def parameters_to_weights(parameters):
    tensors = parameters.parameters.tensors
    return [np.frombuffer(t, dtype=np.float32) for t in tensors]

def get_best_model():
    best_model = FileController().get_file(file_name="best_model")
    model_name = best_model['model_name']
    model_params = best_model['model_parameters']
    model_class,_ = ModelEnum.get_model_data(model_name)
    model = model_class.__class__(**model_params)
    return model
