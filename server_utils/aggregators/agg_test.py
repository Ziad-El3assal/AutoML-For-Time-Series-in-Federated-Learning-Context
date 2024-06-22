import pandas as pd
import numpy as np
from statistics import mean, mode , stdev,  StatisticsError
from scipy.stats import entropy
from server_utils.aggregators.base_aggregator import Aggregator

class MetaFeatureExtractionAggregator(Aggregator):
    def aggregate(self, parameters, data_sizes=[]):
        print(parameters)
        out_parameters = {}
        for feature, feature_parameters in parameters.items():
            out_parameters[feature] = {}
            """Meta Feature Extraction Before preprocessing""" 
            out_parameters[feature]['No. Of Clients'] = self._aggregate_len(feature_parameters['No. Of Clients'])
            #for Instances in Clients
            out_parameters[feature]['Sum of Instances in Clients'] = self._aggregate_sum(feature_parameters['No. Of Instances'])
            out_parameters[feature]['Max. Of Instances in Clients'] = self._aggregate_max(feature_parameters['No. Of Instances'])
            out_parameters[feature]['Min. Of Instances in Clients'] = self._aggregate_min(feature_parameters['No. Of Instances'])
            out_parameters[feature]['Stddev of Instances in Clients'] = self._aggregate_std(feature_parameters['No. Of Instances'])
            # for dataset missing values 
            out_parameters[feature]['Average Dataset Missing Values %'] = self._aggregate_average(feature_parameters['Average Dataset Missing Values %'])
            out_parameters[feature]['Min Dataset Missing Values %'] = self._aggregate_min(feature_parameters[' Dataset Missing Values %'])
            out_parameters[feature]['Max Dataset Missing Values %'] = self._aggregate_max(feature_parameters[' Dataset Missing Values %'])
            out_parameters[feature]['Stddev Dataset Missing Values %'] = self._aggregate_std(feature_parameters['Dataset Missing Values %'])
            # for target missing values
            out_parameters[feature]['Average Target Missing Values %'] = self._aggregate_average(feature_parameters[' Target Missing Values %'])
            out_parameters[feature]['Min Target Missing Values %'] = self._aggregate_min(feature_parameters['Target Missing Values %'])
            out_parameters[feature]['Max Target Missing Values %'] = self._aggregate_max(feature_parameters[' Target Missing Values %'])
            out_parameters[feature]['Stddev Target Missing Values %'] = self._aggregate_std(feature_parameters[' Target Missing Values %'])
            # for No. Of Features
            out_parameters[feature]['No. Of Features'] = self._aggregate_mode(feature_parameters['No. Of Features'])
            # for No. Of Numerical Features
            out_parameters[feature]['No. Of Numerical Features'] = self._aggregate_mode(feature_parameters['No. Of Numerical Features'])
            # for No. Of Categorical Features
            out_parameters[feature]['No. Of Categorical Features'] = self._aggregate_mode(feature_parameters['No. Of Categorical Features'])
            # for Sampling Rate
            out_parameters[feature]['Sampling Rate'] = self._aggregate_mode(feature_parameters['Sampling Rate'])
            # for Skewness of Numerical Features
            out_parameters[feature]['Average Skewness of Numerical Features'] = self._aggregate_average(feature_parameters['Average Skewness of Numerical Features'])
            out_parameters[feature]['Minimum Skewness of Numerical Features'] = self._aggregate_min(feature_parameters['Minimum Skewness of Numerical Features'])
            out_parameters[feature]['Maximum Skewness of Numerical Features'] = self._aggregate_max(feature_parameters['Maximum Skewness of Numerical Features'])
            out_parameters[feature]['Stddev Skewness of Numerical Features'] = self._aggregate_std(feature_parameters['Stddev Skewness of Numerical Features'])
            # for Kurtosis of Numerical Features
            out_parameters[feature]['Average Kurtosis of Numerical Features'] = self._aggregate_average(feature_parameters['Average Kurtosis of Numerical Features'])
            out_parameters[feature]['Minimum Kurtosis of Numerical Features'] = self._aggregate_min(feature_parameters['Minimum Kurtosis of Numerical Features'])
            out_parameters[feature]['Maximum Kurtosis of Numerical Features'] = self._aggregate_max(feature_parameters['Maximum Kurtosis of Numerical Features'])
            out_parameters[feature]['Stddev Kurtosis of Numerical Features'] = self._aggregate_std(feature_parameters['Stddev Kurtosis of Numerical Features'])
            # for No. Of Symbols per Categorical Features
            out_parameters[feature]['Avg No. of Symbols per Categorical Features'] = self._aggregate_average(feature_parameters['Avg No. of Symbols per Categorical Features'])
            out_parameters[feature]['Min. No. Of Symbols per Categorical Features'] = self._aggregate_min(feature_parameters['Min. No. Of Symbols per Categorical Features'])
            out_parameters[feature]['Max. No. Of Symbols per Categorical Features'] = self._aggregate_max(feature_parameters['Max. No. Of Symbols per Categorical Features'])
            out_parameters[feature]['Stddev No. Of Symbols per Categorical Features'] = self._aggregate_std(feature_parameters['Stddev No. Of Symbols per Categorical Features'])

            """Meta Time Series Feature Extraction"""
            # No. Of Stationary Features
            out_parameters[feature]['Avg No. Of Stationary Features'] = self._aggregate_average(feature_parameters['No. Of Stationary Features'])
            out_parameters[feature]['Min No. Of Stationary Features'] = self._aggregate_min(feature_parameters[' No. Of Stationary Features'])
            out_parameters[feature]['Max No. Of Stationary Features'] = self._aggregate_max(feature_parameters[' No. Of Stationary Features'])
            out_parameters[feature]['Min No. Of Stationary Features'] = self._aggregate_max(feature_parameters['No. Of Stationary Features'])
            # No. Of Stationary Features after 1st order diff
            out_parameters[feature]['Avg No. Of Stationary Features after 1st order'] = self._aggregate_average(feature_parameters['No. Of Stationary Features after 1st order'])
            out_parameters[feature]['Min No. Of Stationary Features after 1st order'] = self._aggregate_min(feature_parameters[' No. Of Stationary Features after 1st order'])
            out_parameters[feature]['Max No. Of Stationary Features after 1st order'] = self._aggregate_max(feature_parameters[' No. Of Stationary Features after 1st order'])
            out_parameters[feature]['Stddev No. Of Stationary Features after 1st order'] = self._aggregate_std(feature_parameters[' No. Of Stationary Features after 1st order'])
            # No. Of Stationary Features after 2nd order diff
            out_parameters[feature]['Avg No. Of Stationary Features after 2nd order'] = self._aggregate_average(feature_parameters[' No. Of Stationary Features after 2nd order'])
            out_parameters[feature]['Min No. Of Stationary Features after 2nd order'] = self._aggregate_min(feature_parameters[' No. Of Stationary Features after 2nd order'])
            out_parameters[feature]['Max No. Of Stationary Features after 2nd order'] = self._aggregate_max(feature_parameters['No. Of Stationary Features after 2nd order'])
            out_parameters[feature]['Stddev No. Of Stationary Features after 2nd order'] = self._aggregate_std(feature_parameters[' No. Of Stationary Features after 2nd order'])
            # Significant Lags using pACF in Target
            out_parameters[feature]['Avg No. Of Significant Lags in Target'] = self._aggregate_average(feature_parameters[' No. Of Significant Lags in Target'])
            out_parameters[feature]['Min No. Of Significant Lags in Target'] = self._aggregate_min(feature_parameters['No. Of Significant Lags in Target'])
            out_parameters[feature]['Max No. Of Significant Lags in Target'] = self._aggregate_max(feature_parameters[' No. Of Significant Lags in Target'])
            out_parameters[feature]['Stddev No. Of Significant Lags in Target'] = self._aggregate_std(feature_parameters['No. Of Significant Lags in Target'])
            # No. Of Insignificant Lags between 1st and last significant ones in Target
            out_parameters[feature]['Avg No. Of Insignificant Lags in Target'] = self._aggregate_average(feature_parameters[' No. Of Insignificant Lags in Target'])
            out_parameters[feature]['Max No. Of Insignificant Lags in Target'] = self._aggregate_max(feature_parameters[' No. Of Insignificant Lags in Target'])
            out_parameters[feature]['Min No. Of Insignificant Lags in Target'] = self._aggregate_min(feature_parameters[' No. Of Insignificant Lags in Target'])
            out_parameters[feature]['Stddev No. Of Insignificant Lags in Target'] = self._aggregate_std(feature_parameters[' No. Of Insignificant Lags in Target'])
            # No. Of Seasonality Components in Target
            out_parameters[feature]['Avg. No. Of Seasonality Components in Target'] = self._aggregate_average(feature_parameters['. No. Of Seasonality Components in Target'])
            out_parameters[feature]['Max No. Of Seasonality Components in Target'] = self._aggregate_max(feature_parameters[' No. Of Seasonality Components in Target'])
            out_parameters[feature]['Min No. Of Seasonality Components in Target'] = self._aggregate_min(feature_parameters[' No. Of Seasonality Components in Target'])
            out_parameters[feature]['Stddev No. Of Seasonality Components in Target'] = self._aggregate_std(feature_parameters[' No. Of Seasonality Components in Target'])
            # Fractal Dimension Analysis of Target
            out_parameters[feature]['Average Fractal Dimensionality Across Clients of Target'] = self._aggregate_average(feature_parameters['Fractal Dimensionality Across Clients of Target'])
            # Period of Seasonality Components in Target
            out_parameters[feature]['Maximum Period of Seasonality Components in Target Across Clients'] = self._aggregate_max(feature_parameters['Maximum Period of Seasonality Components in Target Across Clients'])
            out_parameters[feature]['Minimum Period of Seasonality Components in Target Across Clients'] = self._aggregate_min(feature_parameters['Minimum Period of Seasonality Components in Target Across Clients'])
            # Target Stationarity Entropy
            out_parameters[feature]['Entropy of Target Stationarity'] = self._aggregate_entropy(feature_parameters['Target Stationarity'])
            print(out_parameters)
            
        return out_parameters
    
    # Sum
    def _aggregate_sum(self , values): 
        return sum(values)
    # max
    def _aggregate_max(self, values):
        return max(values)
    # min
    def _aggregate_min(self, values):
        return min(values)
    # mean 
    def _aggregate_mean(self , values): 
        return mean(values)
    # mode 
    def _aggregate_mode(self, values): 
        try:
            return mode(values)
        except StatisticsError:
            return "No unique mode"
    # std 
    def _aggregate_std(self , values): 
        return stdev(values)
    # len
    def _aggregate_len(self , values): 
        return len(values)
    # average 
    def _aggregate_average(self , values):
        return sum(values) / len(values)
    # entropy
    def _aggregate_entropy(self, values):
        # Count the occurrences of each value in the list
        value_counts = [values.count(v) for v in set(values)]
        # Calculate entropy using counts directly
        return entropy(value_counts, base=None)  # Using natural log
    

    
    
    