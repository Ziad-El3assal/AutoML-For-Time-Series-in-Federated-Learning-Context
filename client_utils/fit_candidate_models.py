from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Lasso, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from client_utils.file_controller import FileController
from client_utils.TrainTestSplits import TrainTestSplit
from client_utils.ModelEnum import ModelEnum
class FitCandidateModels:
    """
    Fits different regression models on the given dataset and selected features,
    then computes the RMSE for each model on the test set.

    Parameters:
    regression_data (DataFrame): The regression dataset.
    selected_features (list): List of selected features to be used in the models.
    models_to_fit (dict): Dictionary containing models to be fitted with their respective hyperparameters for grid search.
    """

    def __init__(self, regression_data, selected_features, models_to_fit, target_column):
        self.regression_data = regression_data.dropna(axis=0)
        self.selected_features = selected_features
        self.models_to_fit = models_to_fit
        self.file_controller = FileController()
        self.target_column = target_column
        # self.regression_data.set_index('Timestamp', inplace=True)
        self.time_series_split = TrainTestSplit(data= self.regression_data,train_freq=0.8,
                                                 selected_features=self.selected_features,
                                                 target_column=self.target_column)

        self.modelEnum = ModelEnum
    def fit_models(self):
        """
        Fits different regression models on the data and computes RMSE for each model.

        Returns:
        dict: A dictionary containing RMSE for each model.
        """

        X_train, y_train, X_test, y_test = self.time_series_split.train_test_split(test=True)

        # Fit models and compute RMSE
        out_put = {}
        hyperparameters_result = {}
        tscv = TimeSeriesSplit(n_splits=5)
        for name in self.models_to_fit:
            model, params = self.modelEnum.get_model_data(name)
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=tscv, verbose=1,n_jobs=-1)
            #how to svae the outputs of grid serac in csv   
            
            grid_search.fit(X_train, y_train)
            hyperparameters_result[name] = grid_search.best_params_
            y_pred = grid_search.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            out_put[name] = rmse
        self.file_controller.save_file(hyperparameters_result,"hyperParameters")
        
        # results=grid_search.cv_results_
        # results = pd.DataFrame(results)
        # results.to_csv("grid_search_results.csv")
        return {"rmse_results": out_put}


# Example usage:
# if __name__ == "__main__":
#     from sklearn.datasets import make_regression
#
#     # Generate dummy regression dataset
#     X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
#     my_regression_dataset = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(1, 11)])
#     my_regression_dataset['Target'] = y
#
#     # List of selected features
#     my_selected_features = ['feature_1', 'feature_3', 'feature_5', 'feature_7', 'feature_8']
#
#     # Dictionary of models with hyperparameters for grid search
#     models_to_fit = ['Lasso', 'SVR', 'ElasticNetCV', 'XGBoostRegressor', 'MLPRegressor']
#
#     # Initialize the class with your regression dataset, selected features, and models to fit
#     fit_different_models = FitCandidateModels(my_regression_dataset, my_selected_features, models_to_fit)
#
#     # Call the fit_models method to get the RMSE results for each model
#     rmse_results = fit_different_models.fit_models()
#     print(rmse_results)
