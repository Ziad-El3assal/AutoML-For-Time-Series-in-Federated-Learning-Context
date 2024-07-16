from sklearn.metrics import mean_squared_error
import numpy as np
from client_utils.split_data import SplitData  # Assuming SplitData is implemented elsewhere
from sklearn.linear_model import Lasso, ElasticNetCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import csv
import sys
import pandas as pd
from client_utils.file_controller import FileController
import time
import os

class FitModelsFromCSV:
    """
    Fits different regression models on the given dataset and selected features,
    then computes the RMSE for each model on the test set.

    Parameters:
    train_data (DataFrame): The training dataset.
    test_data (DataFrame): The test dataset.
    selected_features (list): List of selected features to be used in the models.
    models_csv (str): Path to the CSV file containing models and parameters.
    target_column (str): Name of the target column in the dataset.
    """

    def __init__(self, train_data, test_data, selected_features, models_csv, target_column,dataset_name):
        self.train_data = train_data
        self.test_data = test_data
        self.selected_features = selected_features
        self.models_csv = models_csv
        self.target_column = target_column
        self.dataset_name = dataset_name
        self.curr_client = sys.argv[1]
        self.file_name = "results"
        self.file_controller = FileController()
        self.num_Clients =int(os.getenv('number_clients'))

    @staticmethod
    def create_model_from_string(model_string):
        """
        Create a regression model from a string representation.

        Parameters:
        model_string (str): String representation of the model with parameters.

        Returns:
        model: Instantiated regression model object.
        """
        # Split the model_string into model_name and parameters
        model_name, params_string = model_string.split(',', 1)
        
        # Clean up model_name and remove surrounding whitespaces
        model_name = model_name.strip()
        
        # Extract parameters and convert them into a dictionary
        params_list = [param.strip() for param in params_string.split(',')]
        params = {}

        # Set of keys we are interested in (parameters that need conversion to float)
        float_params = {'alpha', 'C', 'epsilon', 'l1_ratio',
                        'reg_lambda', 'gamma', 'subsample', 'learning_rate_init'}
        
        # Set of keys we are interested in (parameters that need conversion to bool)
        bool_params = {'early_stopping'}

        # Set of keys we are interested in (parameters that need conversion to int)
        int_params = {'max_iter'}

        for param in params_list:
            key, value = param.split('=')
            cleanKey = key.strip()
            cleanValue = value.strip()
            if cleanKey in float_params:
                params[cleanKey] = float(cleanValue) 
            elif cleanKey in int_params:
                # print(cleanValue)
                params[cleanKey] = int(cleanValue)    
            elif cleanKey in bool_params:
                params[cleanKey] =cleanValue.lower() == 'true'
            elif cleanKey == 'learning_rate':
                if cleanValue in ['constant', 'adaptive']:
                    params[cleanKey] = cleanValue
                else:
                    params[cleanKey] = float(cleanValue)      
            else:
                params[cleanKey] = cleanValue
            
        
        # Initialize the corresponding model with the extracted parameters
        if model_name == "Lasso":
            model = Lasso(random_state=42, **params)
        elif model_name == "SVR":
            model = SVR(**params)
        elif model_name == "ElasticNetCV":
            model = ElasticNetCV(random_state=42, **params)
        elif model_name == "XGBRegressor":
            model = XGBRegressor(random_state=42, **params)
        elif model_name == "MLPRegressor":
            model = MLPRegressor(random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model    

    # def fit_models(self):
    #     """
    #     Fits different regression models on the data and computes RMSE for each model.

    #     Returns:
    #     dict: A dictionary containing RMSE for each model.
    #     """
    #     X_train, y_train = SplitData(data=self.train_data,
    #                                  selected_features=self.selected_features,
    #                                  target_column=self.target_column).x_y_split()
    #     X_test, y_test = SplitData(data=self.test_data,
    #                                selected_features=self.selected_features,
    #                                target_column=self.target_column).x_y_split()

    #     # Fit models one by one and compute RMSE
    #     out_put = {}

    #     print("hfffffffffffffffff")

    #     with open(self.models_csv, 'r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             model_name = row['Model']
    #             params_str = row['Parameters']
    #             model_string = f"{model_name},{params_str}"
    #             model = self.create_model_from_string(model_string)

    #             # Fit the model
    #             model.fit(X_train, y_train)

    #             # Predict on test set and compute RMSE
    #             y_pred = model.predict(X_test)
    #             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
    #             # Store RMSE in the output dictionary
    #             out_put[model_name] = rmse

    #             #Print model details and RMSE
    #             print(f"Model: {model_name}")
    #             print(f"Parameters: {model.get_params()}")
    #             print(f"RMSE: {rmse}")
    #             print()
             

    #             # Optionally, delete the model instance to release memory
    #             del model
        
    #     # return out_put 

    def fit_models(self):
        X_train, y_train = SplitData(data=self.train_data,
                                     selected_features=self.selected_features,
                                     target_column=self.target_column).x_y_split()
        X_test, y_test = SplitData(data=self.test_data,
                                   selected_features=self.selected_features,
                                   target_column=self.target_column).x_y_split()

        results = []

        with open(self.models_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model_name = row['Model']
                params_str = row['Parameters']
                model_string = f"{model_name},{params_str}"
                
                model = self.create_model_from_string(model_string)
                record_exists = self.file_controller.check_record_exists(self.dataset_name,self.curr_client, self.num_Clients, model_name,model.get_params(), self.file_name)
                # print(record_exists)
                if record_exists:
                  continue  
                start_time = time.time()
                model.fit(X_train, y_train)
                elapsed_time = time.time() - start_time

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                result = {
                    'model': model_name,
                    'hyperparameters': model.get_params(),
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'time_taken': elapsed_time,
                    'dataset_name': self.dataset_name,
                    'num_clients': self.num_Clients,
                    'id': f"{self.dataset_name}-{self.curr_client}"
                }
                results.append(result)

                # print(f"Model: {model_name}")
                # print(f"Parameters: {model.get_params()}")
                # print(f"Train RMSE: {train_rmse}")
                # print(f"Test RMSE: {test_rmse}")
                # print(f"Time Taken: {elapsed_time}")
                # print()
        
        results_df = pd.DataFrame(results)
        self.file_controller.save_file_append(results_df, self.file_name, type="csv")
        del model

        # return results_df
