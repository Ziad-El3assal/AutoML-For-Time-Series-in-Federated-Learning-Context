from collections import Counter

from server_utils.aggregators.base_aggregator import Aggregator
import numpy as np
import flwr as fl
class ModelWeightsAggregator(Aggregator):

    def parameters_to_weights(self,parameters):
        tensors = parameters.tensors
        return [np.frombuffer(t, dtype=np.float32) for t in tensors]

# Define a function to convert weights to Parameters
    def weights_to_parameters(self,weights):
        return fl.common.Parameters(tensors=[w.astype(np.float32).tobytes() for w in weights],
                                    tensor_type="model_weights")

    def aggregate(self, parameters, data_sizes=[]):
        weights_results = [self.parameters_to_weights(parameters_res.parameters) for _,parameters_res in parameters]
        num_examples = [num_examples.num_examples for _, num_examples in parameters]
        if len(weights_results) == 0:
            raise ValueError("No weights to aggregate")

        # Ensure all clients have the same number of layers and weights
        num_layers = len(weights_results[0])
        for client_weights in weights_results:
            assert len(client_weights) == num_layers, "Mismatch in number of layers among clients"

        # Perform a weighted average of the weights
        weighted_weights = []
        aggregated_parameters = []
        for idx in range(num_layers):
            # Collect the weights for the current layer from all clients
            layer_weights = [weights[idx] for weights in weights_results]

            # Perform a weighted average of these weights
            averaged_layer_weights = np.average(layer_weights, axis=0, weights=num_examples)

            # Append the averaged weights for the current layer to the list
            weighted_weights.append(averaged_layer_weights)

            # Convert the aggregated weights back to parameters
        aggregated_parameters = self.weights_to_parameters(weighted_weights)
        return aggregated_parameters


