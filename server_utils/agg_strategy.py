# Import aggregator
from server_utils.aggregators.create_aggregator import CreateAggregator
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import numpy as np
from datetime import datetime
import json
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy import fedavg

import os 
from client_utils.file_controller import FileController
import time
import pandas as pd

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class CustomStrategy(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
            self,
            round_number,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 3,
            min_evaluate_clients: int = 3,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.aggregator = CreateAggregator()
        self.num_client_available = 0
        self.weights = {}
        self.best_round = -1
        self.best_loss = 99999999
        self.max_round = round_number
        ###### my editss
        print('-'*10)
        print(os.getenv('DataSetPath'))
        print(os.getenv('nClients'))
        print('-'*10)

        self.data_name=os.path.splitext(os.path.basename(os.getenv('DataSetPath')))[0]
        
        self.file_name = "FLresults"
        self.file_controller = FileController()
        self.num_Clients =int(os.getenv('nClients'))
        self.FLResult={}

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CustomAgg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        self.startTime = time.time()
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        import time
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        num_clients_connected = client_manager.num_available()
        while num_clients_connected < min_num_clients :
            num_clients_connected = client_manager.num_available()
            if num_clients_connected == min_num_clients:
                time.sleep(20)
        config = {}
        if self.num_client_available < client_manager.num_available():
            server_round = 1
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self.num_client_available = num_clients_connected
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0 or server_round < 4:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        agg_parameters = fl.common.Parameters(tensors=[], tensor_type="dict")
        #print(agg_parameters)

        client, fit_res = results[0]
        #print(fit_res)
        #print()
        metrics = fit_res.metrics
        #print(metrics)
        #print()
        aggregator = self.aggregator.createAggregator(server_round=server_round,metrics=metrics)
        if aggregator:
            if server_round >= 4:
                agg_parameters = aggregator.aggregate(results)
                self.weights[server_round] = agg_parameters
            else:
                agg_features = aggregator.aggregate_keys(results=results)
                agg_size = aggregator.aggregate_size(results=results)
                agg_parameters = aggregator.aggregate(agg_features, agg_size)
                agg_parameters['server_round'] = server_round + 1
                agg_features = json.dumps(agg_parameters).encode("utf-8")
                agg_parameters = fl.common.Parameters(tensors=[agg_features], tensor_type="features_weights")

        if server_round == self.max_round:
            #print(1)
            agg_parameters = self.weights[self.best_round]
            #print()
            #print(results)
            #print()
            self.FLResult["model"] = fit_res.metrics["model"]
            self.FLResult["hyperparameters"] = fit_res.metrics["model_hyperparameters"]
        return agg_parameters, {}

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        #print(results)
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if self.best_loss > loss_aggregated:
            self.best_loss=loss_aggregated
            self.best_round = server_round
            self.train_aggregated = np.mean([r.metrics["train_loss"] for _, r in results if "train_loss" in r.metrics])
            self.train_times_aggregated = np.mean([r.metrics["train_time"] for _, r in results if "train_time" in r.metrics])
        if server_round == self.max_round:
            #print(2)
            self.FLResult['dataset_name']=self.data_name
            self.FLResult['num_clients']=self.num_Clients
            self.FLResult['best_loss']=self.best_loss
            self.FLResult["train_rmse"]= np.mean([r.metrics["train_loss"] for _, r in results if "train_loss" in r.metrics])
            print("---------",self.FLResult["train_rmse"])
            # self.FLResult["time_taken"]=self.train_times_aggregated
            self.FLResult["time_taken"] = time.time()-self.startTime
            # FLresult = {
                    
            #         # 'hyperparameters': model.get_params(),
            #         # 'train_rmse': train_rmse,
            #         # 'test_rmse': test_rmse,
            #         # 'time_taken': elapsed_time,
            #         ,
            #         'num_clients': self.num_Clients,
            #         'best_loss':self.best_loss
            #     }
            # FLresults.append(FLresult)
            self.file_controller.save_file_append(pd.DataFrame([self.FLResult]), self.file_name, type="csv")
   
        return loss_aggregated, metrics_aggregated