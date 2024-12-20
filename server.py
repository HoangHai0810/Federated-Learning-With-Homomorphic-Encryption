from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from typing import Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.server import ServerConfig, ServerAppComponents
from flwr.common import Parameters, NDArrays
from blockchain import Blockchain  # Import blockchain class
import numpy as np
from model import Net
import pickle

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

def sparse_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Chuyển đổi Parameters sang NDArrays (dạng numpy)."""
    return [pickle.loads(tensor) for tensor in parameters.tensors]

def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
    """Chuyển đổi NDArrays (dạng numpy) sang Parameters."""
    return ndarrays_to_parameters([pickle.dumps(ndarray) for ndarray in ndarrays])

class FedCustom(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
        )
        self.blockchain = Blockchain()
        self.history = {"round": [], "loss": [], "accuracy": []}

    def aggregate_fit(self, rnd: int, results: List[Tuple], failures: List[BaseException]):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_result is not None:
            aggregated_weights, _ = aggregated_result
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_weights)

            self.blockchain.add_block(rnd, aggregated_ndarrays)
            print(f"Round {rnd}: Aggregated weights added to blockchain.")

        return aggregated_result

    def aggregate_evaluate(
        self, rnd: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated = np.mean([res.loss for _, res in results])
        accuracy_aggregated = np.mean([res.metrics["accuracy"] for _, res in results])

        print(f"Round {rnd}: Aggregated Loss: {loss_aggregated:.4f}, Accuracy: {accuracy_aggregated:.4f}")

        self.history["round"].append(rnd)
        self.history["loss"].append(loss_aggregated)
        self.history["accuracy"].append(accuracy_aggregated)

        return loss_aggregated, {"accuracy": accuracy_aggregated}

def server_fn(context: Context):
    strategy = FedCustom()
    config = ServerConfig(num_rounds=10)
    return ServerAppComponents(config=config, strategy=strategy)
