from server_utils.aggregators.agg_models import ModelsAggregator
from server_utils.aggregators.agg_feature_importance import FeatureImportanceAggregator
from server_utils.aggregators.agg_feature_extraction import FeatureExtractionAggregator
from server_utils.aggregators.agg_model_weights import ModelWeightsAggregator
from server_utils.aggregators.agg_xgboost_weights import XGBoostWeightsAggregator


class CreateAggregator():

    def createAggregator(self, server_round,metrics={}):
        if server_round == 1:
            return FeatureExtractionAggregator()
        elif server_round == 2:
            return FeatureImportanceAggregator()
        elif server_round == 3:
            return ModelsAggregator()
        else:
            if "model" in metrics:
                if metrics['model'] == "XGBRegressor":
                    return XGBoostWeightsAggregator()
                else:
                    return ModelWeightsAggregator()
        return None
