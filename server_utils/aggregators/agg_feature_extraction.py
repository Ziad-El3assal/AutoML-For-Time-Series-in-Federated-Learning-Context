import pandas as pd
from server_utils.aggregators.base_aggregator import Aggregator


class FeatureExtractionAggregator(Aggregator):
    def aggregate(self, parameters, data_sizes=[]):
        out_parameters = {}
        print("Feature_extraction",parameters)
        for feature, feature_parameters in parameters.items():
            out_parameters[feature] = {}
            out_parameters[feature]['optimal_lags'] = self._aggregate_lags(feature_parameters['optimal_lags'])
            out_parameters[feature]['stationary'] = self._aggregate_stationary(feature_parameters['stationary'])
            out_parameters[feature]['seasonality'] = self._aggregate_seasonality(feature_parameters['seasonality'])
           

        print(" after agg_extraction",out_parameters)
        return out_parameters 

    def _aggregate_lags(self, lags):
        return max(lags)
    
    def _aggregate_stationary(self, stationary):
        total_0 = sum(1 for x in stationary if x == 0)
        total_1 = sum(1 for x in stationary if x == 1)
        total_2 = sum(1 for x in stationary if x == 2)
        if total_0 > max(total_1, total_2):
            return 0
        return 1 if total_1 > total_2 else 2
        
    def _aggregate_seasonality(self, seasonalities):
        all_peak_freqs_df = pd.DataFrame(columns=["freq", "spectrum"])
        for freq_spectrum in seasonalities:
            generator_peak_freqs = pd.DataFrame({'freq': freq_spectrum['freq'],
                                                 'spectrum': freq_spectrum['spectrum']})
            all_peak_freqs_df = pd.concat([all_peak_freqs_df, generator_peak_freqs]).reset_index(drop=True)
        avg_spectrum_df = all_peak_freqs_df.groupby('freq').mean().reset_index()
        top_10_freq = avg_spectrum_df.sort_values(by='spectrum', ascending=False).head(10)['freq'].to_list()
        return top_10_freq
