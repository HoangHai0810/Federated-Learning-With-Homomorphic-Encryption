import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import pickle
from pathlib import Path
import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from client import client_fn
from server import server_fn
from flwr.client import ClientApp
from flwr.server import ServerApp
from dataset import prepare_dataset
from attack import attack_fn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logging.getLogger("flwr").propagate = False
from SegCKKS import *


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    backend_config = {
        "client_resources": {"num_cpus": 16/cfg_dict['num_clients'], "num_gpus": 2/cfg_dict['num_clients']},
    }
    
    trainloaders, valloaders, _ = prepare_dataset(
        num_partitions=cfg_dict["num_clients"], batch_size=cfg_dict["batch_size"]
    )
    
    def client_fn_with_data(context):
        return client_fn(context, trainloaders, valloaders)
    
    client = ClientApp(client_fn=client_fn_with_data)
    server = ServerApp(server_fn=server_fn)
    

    attack_server = ServerApp(server_fn=attack_fn)

    history = fl.simulation.run_simulation(
        server_app=attack_server,
        client_app=client,
        num_supernodes=cfg_dict["num_clients"],
        backend_config=backend_config,
        verbose_logging=True,
    )

    results_path = Path(save_path) / "results.pkl"
    with open(str(results_path), "wb") as h:
        pickle.dump({"history": history}, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    strategy = server._strategy
    results_path = Path(save_path) / "history.json"
    with open(results_path, "w") as f:
        json.dump(strategy.history, f, indent=4)
    print(f"Saved history to {results_path}")


if __name__ == "__main__":
    main()
