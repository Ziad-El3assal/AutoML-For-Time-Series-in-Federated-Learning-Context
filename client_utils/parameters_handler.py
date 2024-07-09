import json
from client_utils.features_extraction import features_extraction_pipeline
from client_utils.features_engineering import FeaturesEngineeringPipeline
from client_utils.extract_features_importance import FeatureImportanceExtraction
from client_utils.fit_candidate_models import FitCandidateModels
from client_utils.file_controller import FileController
from client_utils.split_data import SplitData
from client_utils.ModelEnum import ModelEnum
from client_utils.utils import get_model_weights
from client_utils.fitModelFromCSV import FitModelsFromCSV
from client_utils.aggFunc import aggCSV
import os

class ParametersHandler:
    def __init__(self, preprocessed_train_data, preprocessed_test_data, columns_types, dataset_type, client_id):
        self.preprocessed_train_data = preprocessed_train_data
        self.preprocessed_test_data = preprocessed_test_data
        self.test_data = None
        self.train_data = None
        self.columns_types, self.dataset_type = columns_types, dataset_type
        self.file_controller = FileController()
        self.data_length = None
        self.selected_features = []
        self.modelEnum = ModelEnum
        self.data_name=os.path.splitext(os.path.basename(os.getenv('DataPath')))[0]
        self.client_id=str(client_id)

    def get_output(self, parameters, data_list):
        server_round = data_list[0]['server_round']
        output = {}
        if server_round == 1:
            # if not os.path.exists("./output/TimeSeriesFeatures_"+self.client_id+".json"):
            if True:
                print(f"Round {server_round} started: Extract time series features")
                time_series_features, self.data_length = features_extraction_pipeline(self.preprocessed_train_data, self.columns_types)
                self.file_controller.save_file(data=self.data_length, file_name="DataLength_"+self.client_id)
                self.file_controller.save_file(data=time_series_features, file_name="TimeSeriesFeatures_"+self.client_id)
                output = self.file_controller.get_file("TimeSeriesFeatures_"+self.client_id)
                print(f"Round {server_round} Done: Extracted time series features and returned to the server")
            else:
                print(f"Round {server_round} started: Extract time series features")
                self.data_length = self.file_controller.get_file("DataLength_"+self.client_id)
                output=self.file_controller.get_file("TimeSeriesFeatures_"+self.client_id)
                print(f"Round {server_round} Done: Extracted time series features and returned to the server")
                
        elif server_round == 2:

            # if not os.path.exists("./output/SelectedTimeSeriesFeatures.json") or not os.path.exists("./output/train_data_"+self.client_id+".csv") or not os.path.exists("./output/test_data_"+self.client_id+".csv") or not os.path.exists("./output/feature_importance.json"):
            if True:

                print(
                    f"Round {server_round} started: Feature engineering on selected time series features and Extract feature importance")
                del data_list[0]['server_round']
                self.file_controller.save_file(data_list[0], "SelectedTimeSeriesFeatures")
                pipeline = FeaturesEngineeringPipeline(features=data_list[0], columns_types=self.columns_types)
                # Fit and transform the train data
                self.train_data = pipeline.fit_transform(self.preprocessed_train_data)
                # Transform the test data
                self.test_data = pipeline.transform(self.preprocessed_test_data)
                self.file_controller.save_file(self.train_data, "train_data_"+self.client_id, "csv")
                self.file_controller.save_file(self.test_data, "test_data_"+self.client_id, "csv")
                feature_importance = FeatureImportanceExtraction(self.train_data,
                                                                target_column=self.columns_types['target'])
                self.file_controller.save_file(feature_importance.extract_feature_importance(), "feature_importance")
                output = feature_importance.extract_feature_importance()
                print(
                    f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
            else:
                print(
                    f"Round {server_round} started: Feature engineering on selected time series features and Extract feature importance")
                self.file_controller.save_file(data_list[0], "SelectedTimeSeriesFeatures")
                self.train_data = self.file_controller.get_file("train_data_"+self.client_id, "csv")
                self.test_data = self.file_controller.get_file("test_data_"+self.client_id, "csv")
                output=self.file_controller.get_file("feature_importance",'json')
                print(
                    f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
        elif server_round == 3:
            print(f"Round {server_round} started: Hyperparameter tuning on candidate models")
            del data_list[0]['server_round']
            models = ['lasso'] 
            self.selected_features = data_list[0]['selected_features']
            self.file_controller.save_file(self.selected_features, "FinalSelectedFeatures")
            # output = FitCandidateModels(self.train_data, self.test_data, self.selected_features, models,
            #                             target_column=self.columns_types['target']).fit_models()
            output={"rmse_results": {'lasso':20000}}
            # print("output = ")
            # print(output)
            # print(f"Round {server_round} Done: returned best performance of candidate models to the server")
            # FitModelsFromCSV(self.train_data, self.test_data, self.selected_features,"models_params.csv", target_column=self.columns_types['target'],dataset_name=self.data_name).fit_models()    
            print(self.data_name)
        elif server_round == 4:
            print(f"Round {server_round} started: Receive the best model over all clients and start to train the model")
            # aggCSV("output/results.csv","resultsAgg").fit()
            del data_list[0]['server_round']
            Data = self.file_controller.get_file("hyperParameters")
            #best_model =  #str(data_list[0]['best_model'])
            best_model = Data['model_name']
            best_model_parameters = Data['HP']
            print(f"Best model: {best_model}, Best_model_parameters: {best_model_parameters}")
            chosen_model = {'model_name': best_model, 'model_parameters': best_model_parameters}
            self.file_controller.save_file(chosen_model, "best_model")
            X, y = SplitData(data=self.train_data, selected_features=self.selected_features,
                             target_column=self.columns_types['target']).x_y_split()
            model_class, _ = self.modelEnum.get_model_data(best_model)
            model = model_class.__class__(**best_model_parameters)
            model.fit(X, y)
            output = get_model_weights(model)
            print(output)
        return output
