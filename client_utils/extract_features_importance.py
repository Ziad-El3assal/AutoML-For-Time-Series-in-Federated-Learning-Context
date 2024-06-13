from sklearn.ensemble import RandomForestRegressor
from client_utils.TrainTestSplits import TrainTestSplit

class FeatureImportanceExtraction:
    """
    Class to extract feature importance using RandomForestRegressor.

    Attributes:
        regression_dataset (pandas.DataFrame): The regression dataset.
        target_column (list): column used as target
    """

    def __init__(self, regression_dataset, target_column):
        """
        Initializes FeatureImportanceExtraction with the provided dataset and selected features.

        Args:
            regression_dataset (pandas.DataFrame): The regression dataset.
            target_column (list): column used as target
        """
        self.regression_dataset = regression_dataset.dropna(axis=0)
        self.target_column = target_column
        self.feature_importance = None


    def extract_feature_importance(self):
        """
        Extracts feature importance using RandomForestRegressor.

        Returns:
            dict: A dictionary containing feature importance scores.
        """
        # Select features from the dataset
        # self.regression_dataset.set_index('Timestamp', inplace=True)
        print(self.regression_dataset)
        print(self.target_column)
        X ,y = TrainTestSplit(data=self.regression_dataset,target_column=self.target_column).train_test_split()
        # Train Random Forest on the data
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)

        # Get feature importance from the model
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        return {"feature_importance": self.feature_importance}

#
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
#     # Define the target column name
#     target_column_name = 'Target'
#
#     # Create an instance of FeatureImportanceExtraction
#     data = pd.read_csv("D:/federatedLearning/federatedLearning/client1/output/regression_data.csv",index_col=0)
#     feature_importance_extractor = FeatureImportanceExtraction(data, 'value')
#
#     # Extract feature importance
#     importance_dict = feature_importance_extractor.extract_feature_importance()
#
#     # Print the dictionary of feature importance
#     print(importance_dict)
