import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import pickle
from pathlib import Path
import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
from client_mnist import client_fn
from server import server_fn
from flwr.client import ClientApp
from flwr.server import ServerApp
import pandas as pd
from attack import attack_fn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logging.getLogger("flwr").propagate = False
from SegCKKS import *

def prepare_dataset(num_partitions, batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Chuyển đổi sang tensor PyTorch với kích thước 4D (batch_size, channels, height, width)
    x_train = torch.tensor(x_train).permute(0, 3, 1, 2)  # Chuyển từ (batch_size, 28, 28, 1) -> (batch_size, 1, 28, 28)
    x_test = torch.tensor(x_test).permute(0, 3, 1, 2)    # (batch_size, 28, 28, 1) -> (batch_size, 1, 28, 28)

    # Phân chia dữ liệu cho các client
    train_data = torch.utils.data.random_split(
        TensorDataset(x_train, torch.tensor(y_train)),
        [len(x_train) // num_partitions] * num_partitions
    )
    val_data = torch.utils.data.random_split(
        TensorDataset(x_test, torch.tensor(y_test)),
        [len(x_test) // num_partitions] * num_partitions
    )

    trainloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_data]
    valloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in val_data]

    return trainloaders, valloaders, None 


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
    mode = 'normal'  # Choose mode: 'normal', 'attack'
    encrypt = 'paillier'  # Choose encryption type: 'plain', 'paillier', 'ckks'
    
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
    if mode == 'attack':
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
