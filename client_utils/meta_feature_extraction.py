import pandas as pd
import numpy as np
from math import log
import sys

class MetaFeatures:
    def __init__(self, df):
        self.df = df
        self.categorical_columns = []
        self.numerical_columns = []
        

    def get_num_features(self):
        target_keywords = ['Close', 'close', 'value', 'Value']
        for col in self.df.columns:
            if any(keyword in col for keyword in target_keywords):
                self.df.rename(columns={col: 'Target'}, inplace=True)
                break
        return len(self.df.columns) - 2 if 'Timestamp' and 'Target' in self.df.columns else len(self.df.columns)

    def get_num_instances(self):
        return len(self.df)

    def get_num_missing_vals(self):
        return self.df.isnull().sum().sum()

    def get_target_missing_vals(self):
        return self.df['Target'].isnull().sum() if 'Target' in self.df.columns else 0

    def columns_types(self):
        num_samples = len(self.df)
        log_num_samples = log(num_samples)
        for column in self.df.columns:
            if column in ['Target', 'Timestamp']:
                continue
            try:
                unique_values = self.df[column].nunique()
                if unique_values < log_num_samples or isinstance(self.df[column].dtype, pd.CategoricalDtype):
                    self.categorical_columns.append(column)
                else:
                    self.df[column] = pd.to_numeric(self.df[column], errors='raise')
                    self.numerical_columns.append(column)
            except:
                self.categorical_columns.append(column)

    def get_num_numerical_features(self):
        return len(self.numerical_columns)

    def get_num_categorical_features(self):
        return len(self.categorical_columns)

    def get_skewness_features(self):
        skewness = self.df[self.numerical_columns].skew(axis=0).abs()
        return {
            "mean": skewness.mean(),
            "min": skewness.min(),
            "max": skewness.max(),
            "std": skewness.std()
        }

    def get_kurtosis_features(self):
        kurtosis = self.df[self.numerical_columns].kurtosis(axis=0).abs()
        return {
            "mean": kurtosis.mean(),
            "min": kurtosis.min(),
            "max": kurtosis.max(),
            "std": kurtosis.std()
        }
    
    def get_symbol_counts(self):
        symbol_stats = {}
        if self.categorical_columns:
            for col in self.categorical_columns:
                unique_counts = self.df[col].astype(str).apply(lambda x: len(set(x)))
                symbol_stats[col] = unique_counts
            symbol_means = np.mean([stats.mean() for stats in symbol_stats.values()])
            symbol_mins = np.min([stats.min() for stats in symbol_stats.values()])
            symbol_maxs = np.max([stats.max() for stats in symbol_stats.values()])
            symbol_stds = np.std([stats.std() for stats in symbol_stats.values()])
        else:
            symbol_means, symbol_mins, symbol_maxs, symbol_stds = 0, 0, 0, 0
        return symbol_means, symbol_mins, symbol_maxs, symbol_stds

    def get_sampling_rate(self):
        if 'Timestamp' in self.df.columns:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            time_diffs = self.df['Timestamp'].diff().dropna().dt.total_seconds()
            if not time_diffs.empty:
                return time_diffs.mode()[0]  # most common time difference
        return None

def meta_feature_extraction(df):
    mf = MetaFeatures(df)
    mf.columns_types()
    num_instances = mf.get_num_instances()
    num_features = mf.get_num_features()
    num_numerical_features = mf.get_num_numerical_features()
    num_categorical_features = mf.get_num_categorical_features()
    sampling_rate = mf.get_sampling_rate()
    missing_vals = mf.get_num_missing_vals()
    target_missing_vals = mf.get_target_missing_vals()
    skewness_features = mf.get_skewness_features()
    kurtosis_features = mf.get_kurtosis_features()
    symbol_means, symbol_mins, symbol_maxs, symbol_stds = mf.get_symbol_counts()

    results = {
        "No. Of Instances": num_instances,
        "No. Of Features": num_features,
        "No. Of Numerical Features": num_numerical_features,
        "No. Of Categorical Features": num_categorical_features,
        "Sampling Rate": sampling_rate if sampling_rate else "N/A",
        "Dataset Missing Values %": (missing_vals / (num_instances * num_features)) * 100,
        "Target Missing Values %": (target_missing_vals / num_instances) * 100,
        "Average Skewness of Numerical Features": skewness_features["mean"],
        "Minimum Skewness of Numerical Features": skewness_features["min"],
        "Maximum Skewness of Numerical Features": skewness_features["max"],
        "Stddev Skewness of Numerical Features": skewness_features["std"],
        "Average Kurtosis of Numerical Features": kurtosis_features["mean"],
        "Minimum Kurtosis of Numerical Features": kurtosis_features["min"],
        "Maximum Kurtosis of Numerical Features": kurtosis_features["max"],
        "Stddev Kurtosis of Numerical Features": kurtosis_features["std"],
        "Avg No. of Symbols per Categorical Features": symbol_means,
        "Min. No. Of Symbols per Categorical Features": symbol_mins,
        "Max. No. Of Symbols per Categorical Features": symbol_maxs,
        "Stddev No. Of Symbols per Categorical Features": symbol_stds
    }
    return results
# if __name__ == "__main__":
#     data_path = sys.argv[1]
#     output_path = sys.argv[2]
#     meta_feature_extraction(data_path, output_path)
