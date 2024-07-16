import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from client_utils.utils import detect_timeseries_type


class TrendExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, seasonality_type, y=None):
        self.seasonality_type = seasonality_type
        return self

    def get_trend_type(self, series_df) -> str:
        """
         check trend type (Linear or logistic) using adfuller test
        :param series_df: series to check the type of its trend
        :returns: str defines series type (linear or logistic)
        """
        # Check if all values in the series are the same
        if series_df.y.nunique() == 1:
            return "flat"

        # Perform the Augmented Dickey-Fuller test
        adf_result = adfuller(series_df)
        p_value = adf_result[1]

        # Determine the trend type based on the p-value
        if p_value < 0.05:
            return "linear"
        else:
            return "logistic"

    def transform(self, X):
        # Assuming X is a DataFrame with 'timestamp' and 'value' columns

        newX = X.copy()
        newX.rename(columns={'Timestamp': 'ds', self.feature: 'y'}, inplace=True)
        self.trend_type = self.get_trend_type(newX.set_index('ds'))
        if self.trend_type == "logistic":
            # with a logistic growth trend, we need to provide information about the capacity (maximum limit) that the time series is expected to approach as it grows
            self.train_cap = max(newX.y.values)
            newX['cap'] = self.train_cap
            # print(self.train_cap)
        self.trend_model = Prophet(seasonality_mode=self.seasonality_type,
                                   growth=self.trend_type)
        self.trend_model.fit(newX)
        trend = self.trend_model.predict(newX)['trend']
        X[f'{self.feature}_trend'] = trend.values
        return X


class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Extract time features
        time_features = pd.to_datetime(X['Timestamp'])
        if self.feature == "Target":
            X['minutes'] = time_features.dt.minute
            X['hours'] = time_features.dt.hour
            X['dayofweek'] = time_features.dt.dayofweek
            X['month'] = time_features.dt.month
            X['quarter'] = time_features.dt.quarter
        return X


class LagFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature, optimal_lags=24):
        self.feature = feature
        self.optimal_lags = optimal_lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute autocorrelation
        for i in range(1, self.optimal_lags + 1):
            X[f'lag{i}_{self.feature}'] = X[self.feature].shift(i)
        return X


class SeasonalityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature, freq):
        self.feature = feature
        self.freq = freq

    def fit(self, X, y=None):
        return self

    def get_seasonality_features(self, peak_frequencies, ind):
        """
        Generate the seasonality features to merge them with the dataframe.
        :param: spectrum CalendarFourier object containing the seasonality components
        """
        seasonal_features = pd.DataFrame()

        # Convert timestamp column to epoch timestamps
        time_regressor = (ind - np.datetime64('1970-01-01T00:00:00')) // np.timedelta64(1, 's')
        for f in peak_frequencies:
            # Create a mask to retain the dominant frequencies
            seasonal_features[f"cos_{int(1 / f)}"] = np.cos(2 * np.pi * time_regressor * f)
            seasonal_features[f"sin_{int(1 / f)}"] = np.sin(2 * np.pi * time_regressor * f)
        seasonal_features['Timestamp'] = ind
        seasonal_features = seasonal_features.set_index('Timestamp')

        return seasonal_features

    def transform(self, X):
        data = X.copy()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
        ind = data.index.to_numpy()
        df_added = self.get_seasonality_features(self.freq, ind)
        df_added.columns = [f'{self.feature}_' + col for col in df_added.columns]
        all_df = pd.concat([data, df_added], axis=1)
        return all_df

class FeaturesEngineeringPipeline:
    def __init__(self, features, columns_types):
        self.features = features
        self.columns_types = columns_types
        self.fitted_pipelines = {}

    def fit_transform(self, train_data):
        self.train_df = []
        for feature, params in self.features.items():
            seasonality = params['seasonality']
            optimal_lags = params['optimal_lags']
            # Define the pipeline
            pipeline = Pipeline([
                ('trend', TrendExtractor(feature)),
                ('time_features', TimeFeaturesExtractor(feature)),
                ('lags', LagFeaturesExtractor(feature, optimal_lags=optimal_lags)),
                ('seasonality', SeasonalityExtractor(feature, freq=seasonality))
            ])
            # Fit and transform the data
            train_data_copy = train_data[[self.columns_types["timestamp"], feature]].copy()
            seasonality_type = detect_timeseries_type(train_data_copy, feature)
            train_reg = pipeline.fit_transform(train_data_copy, seasonality_type)
            train_reg.reset_index(inplace=True)
            self.train_df.append(train_reg)
            # Store the fitted pipeline
            self.fitted_pipelines[feature] = pipeline
        # Append categorical columns
        train_cat_data = train_data[self.columns_types['categorical']].copy().reset_index(drop=True)
        self.train_df.append(train_cat_data)
        self.train_df = pd.concat(self.train_df, axis=1)
        self.train_df = self.train_df.loc[:, ~self.train_df.columns.duplicated()].copy().dropna().set_index('Timestamp')
        return self.train_df

    def transform(self, test_data):
        self.test_df = []
        for feature in self.features.keys():
            # Retrieve the fitted pipeline
            pipeline = self.fitted_pipelines[feature]
            # Transform the test data using the fitted pipeline
            test_data_copy = test_data[[self.columns_types["timestamp"], feature]].copy()
            test_reg = pipeline.transform(test_data_copy)
            test_reg.reset_index(inplace=True)
            self.test_df.append(test_reg)
        # Append categorical columns
        test_cat_data = test_data[self.columns_types['categorical']].copy().reset_index(drop=True)
        self.test_df.append(test_cat_data)
        self.test_df = pd.concat(self.test_df, axis=1)
        self.test_df = self.test_df.loc[:, ~self.test_df.columns.duplicated()].copy().dropna().set_index('Timestamp')
        return self.test_df

