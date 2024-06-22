import json
import numpy as np
from client_utils.features_extraction import features_extraction_pipeline
from client_utils.features_engineering import FeaturesEngineeringPipeline
from client_utils.extract_features_importance import FeatureImportanceExtraction
from client_utils.fit_candidate_models import FitCandidateModels
from client_utils.file_controller import FileController
from client_utils.split_data import SplitData
from client_utils.ModelEnum import ModelEnum
from client_utils.meta_feature_extraction import meta_feature_extraction
from client_utils.utils import get_model_weights
from client_utils.meta_after_featuers import FEX_pipeline

class ParametersHandler:
    def __init__(self,raw_train_data, preprocessed_train_data, preprocessed_test_data, columns_types, dataset_type):
        self.preprocessed_train_data = preprocessed_train_data
        self.preprocessed_test_data = preprocessed_test_data
        self.raw_train_data=raw_train_data
        self.test_data = None
        self.train_data = None
        self.meta_features_before= {}
        self.meta_features_after={}
        self.columns_types, self.dataset_type = columns_types, dataset_type
        self.file_controller = FileController()
        self.data_length = None
        self.selected_features = []
        self.modelEnum = ModelEnum

    def get_output(self, parameters, data_list):
        server_round = data_list[0]['server_round']
        output = {}
        if server_round == 1:
            print(f"Round {server_round} started: Extract time series features")
            time_series_features, self.data_length = features_extraction_pipeline(self.preprocessed_train_data, self.columns_types)
            self.file_controller.save_file(data=time_series_features, file_name="TimeSeriesFeatures")
            output = time_series_features
            print("parameter handler",output)
            print(f"Round {server_round} Done: Extracted time series features and returned to the server")
        elif server_round == 2:
            print(
                f"Round {server_round} started: Feature engineering on selected time series features and Extract feature importance")
            del data_list[0]['server_round']
            self.file_controller.save_file(data_list[0], "SelectedTimeSeriesFeatures")
            pipeline = FeaturesEngineeringPipeline(features=data_list[0], columns_types=self.columns_types)
            # Fit and transform the train data
            self.train_data = pipeline.fit_transform(self.preprocessed_train_data)
            # Transform the test data
            self.test_data = pipeline.transform(self.preprocessed_test_data)
            self.file_controller.save_file(self.train_data, "train_data", "csv")
            self.file_controller.save_file(self.test_data, "test_data", "csv")
            feature_importance = FeatureImportanceExtraction(self.train_data,
                                                             target_column=self.columns_types['target'])
            output = feature_importance.extract_feature_importance()
            print(
                f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
        elif server_round == 3:      
            del data_list[0]['server_round']
            self.selected_features = data_list[0]['selected_features']
            self.file_controller.save_file(self.selected_features, "FinalSelectedFeatures")

            self.meta_features_before=meta_feature_extraction(self.raw_train_data) 
            # creat a selected featuers dataframe
            # Reset the index and add it as a column to the DataFrame
            self.train_data = self.train_data.reset_index()
            columns_to_select = self.selected_features + ['Target', 'Timestamp']
            selected_featuers_df=self.train_data[columns_to_select].copy()
            print(selected_featuers_df)
            self.meta_features_after= FEX_pipeline(selected_featuers_df) 
            self.file_controller.save_file(self.meta_features_before, "MetaFeatuerBefore")
            self.file_controller.save_file(self.meta_features_after, "MetaFeatuerAfter")
            combined_meta_features = self.meta_features_before.copy()  # Start with a copy of the first dictionary
            combined_meta_features.update(self.meta_features_after)    # Update with the second dictionary
            self.file_controller.save_file(combined_meta_features, "allMetaFeatuers")
            output=self._convert_to_serializable(combined_meta_features)
            # output=self.meta_features
            print("meta output",output)
            print(
                f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
        elif server_round == 4:
            print(f"Round {server_round} started: Hyper-parameter tuning on candidate models")
            del data_list[0]['server_round']
            models = ['lasso']
            self.selected_features = data_list[0]['selected_features']
            self.file_controller.save_file(self.selected_features, "FinalSelectedFeatures")
            output = FitCandidateModels(self.train_data, self.test_data, self.selected_features, models,
                                        target_column=self.columns_types['target']).fit_models()
            print(f"Round {server_round} Done: returned best performance of candidate models to the server")
        elif server_round == 5:
            print(f"Round {server_round} started: Receive the best model over all clients and start to train the model")
            del data_list[0]['server_round']
            hyper_parameters = self.file_controller.get_file("hyperParameters")
            best_model = str(data_list[0]['best_model'])
            best_model_parameters = hyper_parameters[best_model]
            print(f"Best model: {best_model}, Best_model_parameters: {best_model_parameters}")
            chosen_model = {'model_name': best_model, 'model_parameters': best_model_parameters}
            self.file_controller.save_file(chosen_model, "best_model")
            X, y = SplitData(data=self.train_data, selected_features=self.selected_features,
                             target_column=self.columns_types['target']).x_y_split()
            model_class, _ = self.modelEnum.get_model_data(best_model)
            model = model_class.__class__(**best_model_parameters)
            model.fit(X, y)
            output = get_model_weights(model)
        return output
 
    def _convert_to_serializable(self, obj):
        """
        Recursively convert non-serializable types to serializable types.
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.int32):  # Add other types here if needed
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj