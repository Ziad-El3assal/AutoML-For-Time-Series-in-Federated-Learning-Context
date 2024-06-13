import json
from client_utils.features_extraction import features_extraction_pipeline
from client_utils.features_engineering import features_engineering_pipeline
from client_utils.extract_features_importance import FeatureImportanceExtraction
from client_utils.fit_candidate_models import FitCandidateModels
from client_utils.file_controller import FileController
from client_utils.read_preprocess_data import ReadPreprocessData
from client_utils.TrainTestSplits import TrainTestSplit
from client_utils.ModelEnum import ModelEnum
from sklearn.metrics import mean_squared_error, mean_absolute_error
from client_utils.utils import get_model_weights, set_model_weights, get_best_model
class ParametersHandler:
    def __init__(self, raw_data, columns_types, dataset_type):
        # self.dataset_path = dataset_path
        self.raw_data, self.columns_types, self.dataset_type = raw_data, columns_types, dataset_type
        self.regression_data = None
        self.file_controller = FileController()
        self.data_length = None
        self.selected_features = []
        self.modelEnum = ModelEnum

    def get_output(self, parameters,data_list):
        server_round = data_list[0]['server_round']
        output = {}
        if server_round == 1:
            #metafeatureextrcation(self.raw_data, self.columns_types)
            print(f"Round {server_round} started: Extract time series features")
            meta_features, self.data_length = features_extraction_pipeline(self.raw_data, self.columns_types)
            self.file_controller.save_file(data=meta_features, file_name="TimeSeriesFeatures")
            output = meta_features
            print(f"Round {server_round} Done: Extracted time series features and returned to the server")
            
        elif server_round == 2:
            print(
                f"Round {server_round} started: Feature engineering on selected time series features and Extract feature importance")
            del data_list[0]['server_round']
            self.file_controller.save_file(data_list[0], "SelectedTimeSeriesFeatures")
            self.regression_data = features_engineering_pipeline(data=self.raw_data, features=data_list[0], columns_types=self.columns_types)
            self.regression_data.set_index('Timestamp', inplace=True)
            train_test_split = TrainTestSplit(data=self.regression_data,train_freq=0.9,
                                              target_column=self.columns_types['target'])
            
            
            self.train_data, self.test_data = train_test_split.time_series_split()
            self.file_controller.save_file(self.train_data, "train_data","csv")
            self.file_controller.save_file(self.test_data, "test_data","csv")
            feature_importance = FeatureImportanceExtraction(self.train_data, target_column=self.columns_types['target'])
            output = feature_importance.extract_feature_importance()
            print(
                f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
        elif server_round == 3:
            print(f"Round {server_round} started: Hyperparameter tuning on candidate models")
            del data_list[0]['server_round']
            models = ['lasso', 'ELASTICNETCV']
            #####################################################################################
            self.selected_features = data_list[0]['selected_features']
            self.file_controller.save_file(self.selected_features, "FinalSelectedFeatures")
            output = FitCandidateModels(self.train_data, self.selected_features, models,
                                        target_column=self.columns_types['target']).fit_models()
            #####################################################################################
            print(f"Round {server_round} Done: returned best performance of candidate models to the server")
        elif server_round == 4:
            print(f"Round {server_round} started: Receive the best model over all clients and start to train the model")
            del data_list[0]['server_round']
            hyper_parameters = self.file_controller.get_file("hyperParameters")
            best_model = str(data_list[0]['best_model'])
            best_model_parameters = hyper_parameters[best_model]
            print(f"Best model: {best_model}, Best_model_parameters: {best_model_parameters}")
            choosed_model = {}
            choosed_model['model_name'] = best_model
            choosed_model['model_parameters'] = best_model_parameters
            self.file_controller.save_file(choosed_model,"best_model")
            X,y = TrainTestSplit(data=self.train_data,selected_features=self.selected_features,
                                 target_column=self.columns_types['target']).train_test_split()
            model_class,_ = self.modelEnum.get_model_data(best_model)
            model = model_class.__class__(**best_model_parameters)
            model.fit(X,y)
            output = get_model_weights(model)
        return output
