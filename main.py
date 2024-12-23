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
import pandas as pd
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


def load_recovered_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_match_percentages(file_path, mode, num_clients, percentages):
    if Path(file_path).is_file():
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame({"Client": range(num_clients)})
        
    df[mode] = percentages
    df.to_csv(file_path, index=False)
    logging.info(f"Saved match percentages for mode '{mode}' to {file_path}")

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    results_file = "match.csv"

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    backend_config = {
        "client_resources": {"num_cpus": 16 / cfg_dict["num_clients"], "num_gpus": 2 / cfg_dict["num_clients"]},
    }
    
    mode = 'attack'
    encrypt = 'plain'  # Choose encryption type: 'plain', 'paillier', 'ckks'
    
    trainloaders, valloaders, _ = prepare_dataset(
        num_partitions=cfg_dict["num_clients"], batch_size=cfg_dict["batch_size"]
    )
    
    client = ClientApp(client_fn=lambda ctx: client_fn(ctx, trainloaders, valloaders))
    server = ServerApp(server_fn=attack_fn if mode == 'attack' else server_fn)

    history = fl.simulation.run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg_dict["num_clients"],
        backend_config=backend_config,
        verbose_logging=True,
    )

    percentages = []
    for i in range(cfg_dict["num_clients"]):
        first_batch = next(iter(trainloaders[i]))
        inputs, labels = first_batch    

        if encrypt == 'plain':
            recovered_data = load_recovered_data(f'recorvered_input_plain/recovered_input_client_plain_{i}.pkl')
        elif encrypt == 'paillier':
            recovered_data = load_recovered_data(f'recorvered_input_paillier/recovered_input_client_paillier_{i}.pkl')
        elif encrypt == 'ckks':
            recovered_data = load_recovered_data(f'recorvered_input_ckks/recovered_input_client_ckks_{i}.pkl')

        recovered_inputs = torch.tensor([item[0] for item in recovered_data], dtype=torch.float32)
        matches = torch.eq(inputs, recovered_inputs).sum().item()
        total_elements = inputs.numel()
        match_percentage = (matches / total_elements) * 100
        percentages.append(match_percentage)
        print(f"Feature {encrypt} match client {i} percentage: {match_percentage:.2f}%")

    save_match_percentages(results_file, encrypt, cfg_dict["num_clients"], percentages)

    results_path = "results.pkl"
    with open(results_path, "wb") as h:
        pickle.dump({"history": history}, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    strategy = server._strategy
    history_file = f"history_{encrypt}.json"
    with open(history_file, "w") as f:
        json.dump(strategy.history, f, indent=4)
    print(f"Saved history to {history_file}")

if __name__ == "__main__":
    main()
