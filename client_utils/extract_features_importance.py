from sklearn.ensemble import RandomForestRegressor
from client_utils.split_data import SplitData

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
        X ,y = SplitData(data=self.regression_dataset, target_column=self.target_column).x_y_split()
        # Train Random Forest on the data
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)
        # Get feature importance from the model
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        return {"feature_importance": self.feature_importance}

