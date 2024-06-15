from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from client_utils.file_controller import FileController
from client_utils.split_data import SplitData
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

    def __init__(self, train_data, test_data, selected_features, models_to_fit, target_column):
        self.train_data = train_data
        self.test_data = test_data
        self.selected_features = selected_features
        self.models_to_fit = models_to_fit
        self.file_controller = FileController()
        self.target_column = target_column
        self.modelEnum = ModelEnum

    def fit_models(self):
        """
        Fits different regression models on the data and computes RMSE for each model.

        Returns:
        dict: A dictionary containing RMSE for each model.
        """

        X_train, y_train = SplitData(data=self.train_data,
                                     selected_features=self.selected_features,
                                     target_column=self.target_column).x_y_split()
        X_test, y_test = SplitData(data=self.test_data,
                                   selected_features=self.selected_features,
                                   target_column=self.target_column).x_y_split()

        # Fit models and compute RMSE
        out_put = {}
        hyperparameters_result = {}
        tscv = TimeSeriesSplit(n_splits=5)
        for name in self.models_to_fit:
            model, params = self.modelEnum.get_model_data(name)
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=tscv, verbose=1)
            grid_search.fit(X_train, y_train)
            hyperparameters_result[name] = grid_search.best_params_
            y_pred = grid_search.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            out_put[name] = rmse
        self.file_controller.save_file(hyperparameters_result, "hyperParameters")
        return {"rmse_results": out_put}

