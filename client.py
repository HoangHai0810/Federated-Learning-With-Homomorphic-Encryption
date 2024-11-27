from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from model import test, train
import time

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = instantiate(model_cfg)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config : Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = 0.01
        momentum = 0.9
        epochs = 1
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        train(self.model, self.trainloader, optimizer, epochs, self.device)
        
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, model_cfg):
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            model_cfg=model_cfg,
        ).to_client()

    return client_fn