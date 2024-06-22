import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, pacf
from scipy.stats import linregress

class Feature_Extraction:
    def __init__(self, df,target=None, timestamp=None):
        self.df = df
        self.target = target 
        self.timestamp = timestamp 
        self.df[self.timestamp] = pd.to_datetime(self.df[self.timestamp])
        self.df = self.df.sort_values(by=self.timestamp).reset_index(drop=True)
        self.series = self.df[self.target]
   
    
    def get_number_of_stationary_features(self):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp:
                series = self.df[col].dropna()
                if adfuller(series)[1] < 0.05:
                    stationary_count += 1
        return stationary_count

    def get_number_of_stationary_features_1_dif(self):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp:
                series = self.df[col].dropna().diff().dropna()
                if adfuller(series)[1] < 0.05:
                    stationary_count += 1
        return stationary_count

    def get_number_of_stationary_features_2_dif(self):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp:
                series = self.df[col].dropna().diff().diff().dropna()
                if adfuller(series)[1] < 0.05:
                    stationary_count += 1
        return stationary_count

    def get_target_stationarity(self):
        p_value = adfuller(self.series)[1]
        return p_value < 0.05

    def get_significant_lags(self, nlags=20):
        pacf_values = pacf(self.series, nlags=nlags)
        conf_interval = 1.96 / np.sqrt(len(self.series))
        significant_lags_count = sum(abs(value) > conf_interval for value in pacf_values)
        return significant_lags_count

    def get_sampling_rate(self):
        X = pd.DataFrame({'timestamp': pd.to_datetime(self.df[self.timestamp])})
        time_diff = X['timestamp'].diff()
        sampling_period = time_diff.dt.total_seconds() / 3600  # sampling rate per hour
        return sampling_period.iloc[1]

    def get_seasonality_components(self):
        decomposition = sm.tsa.seasonal_decompose(self.series, period=1)
        seasonal = decomposition.seasonal
        unique_seasons = len(set(seasonal))
        return unique_seasons

    def get_max_seasonality_period(self, max_period=365):
        best_period = 1
        max_strength = 0
        for period in range(2, max_period + 1):
            decomposition = sm.tsa.seasonal_decompose(self.series, period=period)
            seasonal_strength = decomposition.seasonal.var()
            if seasonal_strength > max_strength:
                max_strength = seasonal_strength
                best_period = period
        return best_period

    def get_min_seasonality_period(self, max_period=365):
        best_period = 1
        min_strength = float('inf')
        for period in range(2, max_period + 1):
            decomposition = sm.tsa.seasonal_decompose(self.series, period=period)
            seasonal_strength = decomposition.seasonal.var()
            if seasonal_strength < min_strength:
                min_strength = seasonal_strength
                best_period = period
        return best_period

    def get_fractal_dimension_analysis(self):
        n = len(self.series)
        max_k = int(np.log2(n))
        k_vals = np.arange(1, max_k + 1)
        L = np.zeros(len(k_vals))
        for i, k in enumerate(k_vals):
            n_k = n // k
            Lk = np.zeros(k)
            for m in range(k):
                idx = np.arange(m, n, k)
                Lk[m] = np.sum(np.abs(np.diff(self.series[idx])))
            L[i] = np.sum(Lk) * (n - 1) / (k * n_k)
        log_k = np.log(k_vals)
        log_L = np.log(L)
        coeffs = np.polyfit(log_k, log_L, 1)
        return coeffs[0]

    def get_insignificant_lags_between_significant(self, nlags=20):
        pacf_values = pacf(self.series, nlags=nlags)
        conf_interval = 1.96 / np.sqrt(len(self.series))
        significant_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > conf_interval]
        if len(significant_lags) < 2:
            return 0
        first_significant = significant_lags[0]
        last_significant = significant_lags[-1]
        insignificant_lags = sum(abs(pacf_values[lag]) < conf_interval for lag in range(first_significant + 1, last_significant))
        return insignificant_lags

    def extract_features(self):
        features = {}
        features['No. Of Stationary Features'] = self.get_number_of_stationary_features()
        features['No. Of Stationary Features after 1st order diff'] = self.get_number_of_stationary_features_1_dif()
        features['No. Of Stationary Features after 2nd order diff'] = self.get_number_of_stationary_features_2_dif()
        features['Target Stationarity'] = self.get_target_stationarity()
        features['Sampling Rate'] = self.get_sampling_rate()
        features['Significant Lags using pACF in Target'] = self.get_significant_lags()
        features['No. Of Insignificant Lags between 1st and last significant ones in Target '] = self.get_insignificant_lags_between_significant()
        features['No. Of Seasonality Components in Target'] = self.get_seasonality_components()
        features['Maximum Period of Seasonality Components in Target'] = self.get_max_seasonality_period()
        features['Minimum Period of Seasonality Components in Target'] = self.get_min_seasonality_period()
        features['Fractal Dimension Analysis of Target'] = self.get_fractal_dimension_analysis()
        return features

def FEX_pipeline(df):
    TARGET_KEYWORDS = ['Close', 'close', 'Value', 'value', 'target', 'Target']
    TIMESTAMP_KEYWORDS = ['timestamp', 'Timestamp']
    target_col, timestamp_col = detect_target_timestamp(df, TARGET_KEYWORDS ,  TIMESTAMP_KEYWORDS)
    ext = Feature_Extraction(df,target_col,timestamp_col)
    return ext.extract_features()

def detect_target_timestamp(df, target_names, timestamp_names):
        target_col = None
        timestamp_col = None

        for col in df.columns:
            if col in target_names:
                target_col = col
            if col in timestamp_names:
                timestamp_col = col

        if target_col is None or timestamp_col is None:
            raise ValueError("Could not find the target or timestamp columns in the DataFrame")

        return target_col, timestamp_col