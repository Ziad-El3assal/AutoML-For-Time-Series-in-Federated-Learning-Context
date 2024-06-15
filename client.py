import flwr as fl
import json
from client_utils.parameters_handler import ParametersHandler
from client_utils.file_controller import FileController
from client_utils.read_preprocess_data import ReadPreprocessData
from client_utils.split_data import SplitData
from client_utils.ModelEnum import ModelEnum
import pickle
from sklearn.metrics import mean_absolute_error
from client_utils.utils import (get_model_weights, set_model_weights, parameters_to_weights,
                                weights_to_parameters, get_best_model)
import pandas as pd
from client_utils.split_data import SplitData
import sys

class FlowerClient(fl.client.Client):
    def __init__(self, cid, server_address, server_port, dataset_path):
        self.cid = cid
        self.server_address = server_address
        self.server_port = server_port
        self.raw_data = pd.read_csv(dataset_path)
        
        self.raw_data.rename({'timestamp':'Timestamp', 'value':'Target'},inplace=True, axis=1)
        print(self.raw_data.columns)
        split_data = SplitData(data=self.raw_data, train_freq=0.9)
        self.raw_train_data, self.raw_test_data = split_data.train_test_split()
        read_preprocess_data = ReadPreprocessData()
        self.preprocessed_train_data, self.columns_types, self.dataset_type = read_preprocess_data.fit_transform(self.raw_train_data)
        self.preprocessed_test_data = read_preprocess_data.transform(self.raw_test_data)
        self.parameters_handler = ParametersHandler(preprocessed_train_data=self.preprocessed_train_data,
                                                    preprocessed_test_data=self.preprocessed_test_data,
                                                    columns_types=self.columns_types, dataset_type=self.dataset_type)
        self.file_controller = FileController()
        self.modelEnum = ModelEnum
        self.selected_features = []
        self.model = None
        super().__init__()  # Initialize the parent class

    def get_parameters(self, config):
        features_bytes = json.dumps({'server_round': 1}).encode("utf-8")
        parameters = fl.common.Parameters(tensors=[features_bytes], tensor_type="features_weights")
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.typing.GetParametersRes(status=status, parameters=parameters)

    def fit(self, parameters):
        # Perform feature extraction
        metrics = {}
        if parameters.parameters.tensor_type == "features_weights":
            data_list = [json.loads(tensor.decode("utf-8")) for tensor in parameters.parameters.tensors]
            if data_list[0]['server_round'] <= 4:
                output = self.parameters_handler.get_output(parameters, data_list)
                if isinstance(output, list):
                    model = get_best_model()
                    metrics["model"] = str(model.__class__.__name__)
                    parameters = fl.common.Parameters(tensors=output,
                                                      tensor_type="weights")

                else:
                    features_bytes = json.dumps(output).encode("utf-8")
                    # Create a Parameters object with features as bytes
                    parameters = fl.common.Parameters(tensors=[features_bytes], tensor_type="dict")

        elif parameters.parameters.tensor_type == "xgboost_weights":
            self.model = get_best_model()
            metrics["model"] = str(self.model.__class__.__name__)
            global_model = []
            for item in parameters.parameters.tensors:
                global_model = bytearray(item)
            self.model.load_model(global_model)
            self.model = set_model_weights(self.model, global_model)
            with open(f'model_{self.cid}.pkl', 'wb') as model_file:
                pickle.dump(self.model, model_file)
            output = get_model_weights(self.model)
            parameters = fl.common.Parameters(tensors=[output],
                                              tensor_type="weights")
        else:
            weights = parameters_to_weights(parameters)
            selected_features = self.file_controller.get_file(file_name="FinalSelectedFeatures")
            train_data = self.file_controller.get_file("train_data", "csv")
            X, y = SplitData(data=train_data,
                             selected_features=selected_features,
                             target_column=self.columns_types['target']).x_y_split()
            self.model = get_best_model()
            metrics["model"] = str(self.model.__class__.__name__)
            self.model = set_model_weights(self.model, weights)
            self.model.fit(X, y)
            with open(f'model_{self.cid}.pkl', 'wb') as model_file:
                pickle.dump(self.model, model_file)
            parameters = get_model_weights(self.model)
            parameters = weights_to_parameters(parameters)
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        fit_res = fl.common.FitRes(
            parameters=parameters,
            num_examples=self.parameters_handler.data_length,
            metrics=metrics,
            # Set metrics to an empty dictionary since you don't want to return any metrics
            status=status
        )
        return fit_res

    def evaluate(self, parameters):
        tensor_type = parameters.parameters.tensor_type
        if tensor_type == "xgboost_weights":
            self.model = get_best_model()
            global_model = []
            for item in parameters.parameters.tensors:
                global_model = bytearray(item)
            self.model.load_model(global_model)
            self.model = set_model_weights(self.model, global_model)
            test_data = self.file_controller.get_file("test_data", "csv")
            selected_features = self.file_controller.get_file(file_name="FinalSelectedFeatures")
            X_test, y_test = SplitData(data=test_data,
                                       selected_features=selected_features,
                                       target_column=self.columns_types['target']).x_y_split()

            # Make predictions
            y_pred = self.model.predict(X_test)
            loss = mean_absolute_error(y_test, y_pred)
            print(loss)
        else:
            test_data = self.file_controller.get_file("test_data", "csv")
            train_data = self.file_controller.get_file("train_data", "csv")
            selected_features = self.file_controller.get_file(file_name="FinalSelectedFeatures")
            X, y = SplitData(data=train_data,
                             selected_features=selected_features,
                             target_column=self.columns_types['target']).x_y_split()
            X_test, y_test = SplitData(data=test_data,
                                       selected_features=selected_features,
                                       target_column=self.columns_types['target']).x_y_split()
            weights = parameters_to_weights(parameters)
            model = get_best_model()
            model.fit(X, y)
            model = set_model_weights(model, weights)
            y_pred = model.predict(X_test)
            loss = mean_absolute_error(y_test, y_pred)
            print(loss)
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.EvaluateRes(loss=loss, num_examples=len(X_test), metrics={}, status=status)

if __name__ == "__main__":
    #--------- server configs --------------
    #get the data path from the bat file
    if len (sys.argv) != 3:
        print("Usage: python client.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[2]
    print(data_path)
    #----------------------------------------
    cid="clinet_"+str(sys.argv[1])
    print(cid)
    
    client_server_address = "localhost"  # Change to actual server address
    client_server_port = 5555  # Change to actual server port

    # # Create an instance of the client
    client = FlowerClient(cid=cid, server_address=client_server_address, server_port=client_server_port,
                          dataset_path=data_path)
    # # # Connect the client to the server
    fl.client.start_client(server_address="localhost:5555", client=client)
