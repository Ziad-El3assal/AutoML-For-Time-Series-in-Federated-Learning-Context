import numpy as np
from server_utils.aggregators.base_aggregator import Aggregator


class ModelsAggregator(Aggregator):
    """
    A class to choose the best model based on evaluations.

    """

    def aggregate(self, parameters, data_sizes=[]):
        """
        Selects the best model based on average performance across evaluations.

        Returns:
            str: The name of the best model.
        """
        models_evaluations = parameters['rmse_results']

        average_performance = {}

        # Take average performance for each model
        for model, result in models_evaluations.items():
            average_performance[model] = np.mean(result)

        # Find the model with the lowest average performance
        best_model = min(average_performance, key=average_performance.get)
        return {"best_model": best_model}

# Example usage:
# if __name__ == "__main__":
#     model_evaluations = {'Lasso': [132, 121],
#                          'SVR': [132, 121],
#                          'ElasticNetCV': [100, 121],
#                          'MLPRegressor': [134, 121],
#                          'XGBoostRegressor': [112, 121]}
#     cbm = ModelsAggregator()
#     best_model = cbm.aggregate(model_evaluations)
#     print("Best model selected:", best_model)
