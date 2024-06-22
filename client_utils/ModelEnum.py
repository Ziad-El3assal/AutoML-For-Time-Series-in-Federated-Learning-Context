from enum import Enum
from sklearn.linear_model import Lasso, ElasticNetCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

class ModelEnum(Enum):
    LASSO = (
        Lasso(random_state=42),
        {'alpha': np.logspace(np.log10(1e-5), np.log10(2), num=30), 'selection': ['cyclic', 'random']}
    )
    SVR = (
        SVR(),
        {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [1, 2, 3, 5, 10], 'epsilon': [0.01, 0.05, 0.1]}
    )
    ELASTICNETCV = (
        ElasticNetCV(random_state=42),
        {'l1_ratio': np.linspace(0.3, 1, 10), 'selection': ['cyclic', 'random']}
    )
    XGBOOST_REGRESSOR = (
        XGBRegressor(random_state=42),
        {
            # 'n_estimators': [20, 200],
            # 'max_depth': [2, 7],
            'learning_rate': [0.1, 1],
            'reg_lambda': [0.8, 10],
            'gamma': [0.9, 1.16467595, 2.248149123539492, 3.9963209507789],
            # 'colsample_bytree': [0.5, 1.0],
            'subsample': [0.1, 1]
        }
    )
    MLP_REGRESSOR = (
        MLPRegressor(random_state=42),
        {
            # 'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
            'learning_rate': ['constant', 'adaptive'],
            'early_stopping': [True, False],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [25, 50, 100]
        }
    )

    @staticmethod
    def get_model_data(model_name: str):
        try:
            model_enum = ModelEnum[model_name.upper()]
            return model_enum.value
        except KeyError:
            raise ValueError(f"Model '{model_name}' is not a valid model name. Choose from: {', '.join([model.name for model in ModelEnum])}")
