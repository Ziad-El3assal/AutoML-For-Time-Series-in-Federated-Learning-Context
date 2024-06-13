import numpy as np
import pandas as pd
from scipy.signal import periodogram, find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller, pacf
from client_utils.utils import log_transform, detect_timeseries_type


class TrendExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        trend = X.rolling(window=7).mean()
        return {'trend': trend.values}


class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, data):
        self.X = data

    def fit(self, X, y=None):
        return self

    def transform(self, d):
        d['minutes'] = True
        d['hours'] = True
        d['day'] = True
        d['week'] = True
        d['month'] = True
        d['year'] = True
        return d


class SeasonalityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature, period=24):
        self.feature = feature
        self.period = period

    def fit(self, X, y=None):
        return self

    def __get_seasonality_components(self, spectrum, frequencies):
        """
        Generate the seasonality features to merge them with the dataframe.
        :param: spectrum
        """
        peaks_inds, _ = find_peaks(spectrum)
        peak_frequencies = []
        freq_spectrum = {}
        threshold = np.mean(spectrum[peaks_inds]) + 2 * np.std(spectrum[peaks_inds])
        for index in peaks_inds:
            if spectrum[index] >= threshold:
                peak_frequencies.append(frequencies[index])
                freq_spectrum[frequencies[index]] = spectrum[index]

        return peak_frequencies, freq_spectrum

    def transform(self, data):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
        time_series_type = detect_timeseries_type(data, self.feature)
        ts_values = data[self.feature].to_numpy()
        if time_series_type == "additive":
            freq, spectrum = periodogram(ts_values, fs=1)
        else:
            data_transformed = log_transform(data, self.feature)
            ts_values = data_transformed[self.feature].to_numpy()
            freq, spectrum = periodogram(ts_values, fs=1)

        peak_frequencies, freq_spectrum = self.__get_seasonality_components(spectrum, freq)
        generator_peak_freqs = pd.DataFrame(freq_spectrum.items(), columns=['freq', 'spectrum'])
        return {"seasonality": {"freq": generator_peak_freqs['freq'].tolist(),
                                'spectrum': generator_peak_freqs['spectrum'].tolist()}}


class LagsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_lags=24):
        self.max_lags = max_lags

    def fit(self, data, y=None):
        return self

    def transform(self, features_extraction):
        pacf_values = pacf(features_extraction[1], nlags=self.max_lags)
        significant_lags = [lag for lag, pacf_value in enumerate(pacf_values) if abs(pacf_value) > 0.1 * max(pacf_values)]
        optimal_lags = significant_lags[1:] if significant_lags else []
        if optimal_lags:
            features_extraction[0]['optimal_lags'] = optimal_lags[-1]
        else:
            features_extraction[0]['optimal_lags'] = 0
        return features_extraction[0]

class StationarityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, data, max_diff=2):
        self.data = data
        self.max_diff = max_diff

    def fit(self, data, y=None):
        return self

    def check_stationarity(self, series, significant_value=0.05):
        adf_result = adfuller(series)
        return adf_result[1] < significant_value

    def transform(self, features_extraction):
        diff_data = self.data.copy()
        diff_count = 0
        while diff_count <= self.max_diff:
            if self.check_stationarity(diff_data):
                break
            diff_data = diff_data.diff().dropna()
            diff_count += 1
        features_extraction['stationary'] = diff_count
        return features_extraction, diff_data


# Create feature extraction pipeline
def features_extraction_pipeline(data, columns_types):
    print("Feature extraction started.............................")
    print("--"*50)
    extracted_feature = {}
    for feature in [columns_types['target']] + columns_types['numerical']:
        input_data = data[[columns_types["timestamp"], feature]].copy()
        feature_extraction_pipeline = Pipeline([
        ('seasonality_extractor', SeasonalityExtractor(feature)),
        ('stationarity_extractor', StationarityExtractor(input_data)),
        ('lags_extractor', LagsExtractor())
        ])
        extracted_feature[feature] = feature_extraction_pipeline.fit_transform(input_data)
    return extracted_feature, len(data)
