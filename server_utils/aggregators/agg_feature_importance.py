import numpy as np
from server_utils.aggregators.base_aggregator import Aggregator


class FeatureImportanceAggregator(Aggregator):
    """
    Class to aggregate feature importance dictionaries.
    """

    def __init__(self):
        self.importance_threshold = 0.05

    def aggregate(self, parameters, data_sizes=[]):
        """
        Aggregate features importance and select top N features.

        Returns:
        - list: List of selected features.
        """
        features_importance = parameters["feature_importance"]

        aggregated_importance = {}

        for feature, total_importance in features_importance.items():
            aggregated_importance[feature] = np.mean(total_importance)

        # Return only feature names
        return {"selected_features": [feature for feature, v in aggregated_importance.items() if
                                      v > self.importance_threshold]}

# Example usage:
# if __name__ == "__main__":
#     # Sample feature importance dictionaries
#     features_importance_list = {"feature1": [0.01, .03, .01],
#                                 "feature2": [0.2, 3, .4],
#                                 "feature3": [0.22, .33, .4]}
#
#     # Create an instance of FeatureImportanceAggregator
#     aggregator = FeatureImportanceAggregator()
#
#     # Aggregate and print the top feature names
#     top_features = aggregator.aggregate(features_importance_list)
#     print("Top Features:")
#     for feature in top_features:
#         print(feature)
