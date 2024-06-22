from abc import ABC, abstractmethod
import json


class Aggregator(ABC):

    def aggregate_keys(self, results):
        # Placeholder for aggregated parameters
        aggregated_keys = {}
        print(results)
        # Aggregate parameters from all clients
        for i in range(len(results)):
            client, fit_res = results[i]
            parameters = fit_res.parameters
            decoded_features = json.loads(parameters.tensors[0].decode("utf-8"))
            for feature in decoded_features.keys():
                if feature not in aggregated_keys:
                    aggregated_keys[feature] = {}
                for k, v in decoded_features[feature].items():
                    if k in aggregated_keys[feature]:
                        aggregated_keys[feature][k].append(v)
                    else:
                        aggregated_keys[feature][k] = [v] 
            print("base aggregator",aggregated_keys)            
        return aggregated_keys 
    def aggregate_size(self, results):
        aggregated_length = []
        # Aggregate parameters from all clients
        for i in range(len(results)):
            client, fit_res = results[i]
            parameters = fit_res.num_examples
            aggregated_length.append(parameters)
        return aggregated_length
    @abstractmethod
    def aggregate(self, parameters, data_sizes=[]):
        pass
