import pickle
from pathlib import Path
from blockchain import Blockchain

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evalulate_fn, get_on_fit_config
import time

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )
    
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)
    
    #Strategy
    blockchain = Blockchain()
    def save_parameters(parameters):
        blockchain.add_transaction(str(hash(str(parameters))))
    class CustomFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            aggregated_fit = super().aggregate_fit(server_round, results, failures)
            if aggregated_fit:
                aggregated_parameters, _ = aggregated_fit
                save_parameters(aggregated_parameters)
            return aggregated_fit
    strategy = CustomFedAvg()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )
    
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
