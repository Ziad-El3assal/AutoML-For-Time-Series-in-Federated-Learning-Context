import pandas as pd
from client_utils.file_controller import FileController


class aggCSV:
    def __init__(self,file_path,outFileName,agg_features =['model', 'hyperparameters', 'dataset_name', 'num_clients'] ):
        self.file_Path = file_path
        self.agg_features = agg_features
        self.file_controller = FileController()
        self.out_file_name = outFileName
        self.best_file_name = "BestFile"

    def fit(self):
        df = pd.read_csv(self.file_Path)
        # Group by specified columns and calculate the average for train_rmse, test_rmse, and time_taken
        agg_df = df.groupby(self.agg_features).agg({
            'train_rmse': 'mean',
            'test_rmse': 'mean',
            'time_taken': 'mean'
        }).reset_index()

        
        best_models = agg_df.loc[agg_df.groupby(['dataset_name', 'num_clients'])['test_rmse'].idxmin()]


        self.file_controller.save_file(best_models, self.best_file_name, type="csv")

        self.file_controller.save_file(agg_df, self.out_file_name, type="csv")