import pickle
import time
import torch
import numpy as np
from flwr.common import FitIns, FitRes, Context
from flwr.client import Client
from flwr.common import FitIns, FitRes, Status, Code
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from server import FedCustom
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from typing import List, Optional, Tuple
from flwr.common import Parameters


class FedAttack(FedCustom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_data_map = {}
        self.recovered_data = []

    def aggregate_fit(self, rnd: int, results: List[Tuple], failures: List[BaseException]):
        print(f"Round {rnd} ended. Starting attack...")
        print(f"Results: {results[0][1].metrics}")
        for _, fit_res in results:
            client_id = fit_res.metrics.get("partition_id")
            assert isinstance(fit_res.parameters, Parameters), "Expected fit_res.parameters to be of type Parameters"
            self.intercept_fit(client_id, fit_res)
        return super().aggregate_fit(rnd, results, failures)


    def intercept_fit(self, client_id: str, fit_res: FitRes) -> FitRes:
        """Intercept parameters from a client and perform an attack or analysis"""
        parameters_original = fit_res.parameters
        print(f"Attacker intercepted parameters from client {client_id}.")

        recovered_data = self.model_inversion(parameters_original, client_id)

        if client_id not in self.client_data_map:
            self.client_data_map[client_id] = []

        self.client_data_map[client_id].append(recovered_data)
        print(f"Recovered data from client {client_id}: {recovered_data}")

        self.recovered_data.append(recovered_data)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_original,
            num_examples=fit_res.num_examples,
            metrics=fit_res.metrics,
        )

    def model_inversion(self, parameters: Parameters, client_id: str):
        """Hồi phục dữ liệu từ tham số."""
        ndarrays = parameters_to_ndarrays(parameters)

        model = self.build_model()
        state_dict = {f"layer_{i}": torch.tensor(value) for i, value in ndarrays.items()}
        model.load_state_dict(state_dict, strict=False)

        num_samples = 32
        recovered_data_list = []

        for sample_idx in range(num_samples):
            input_data = torch.randint(0, 10, (1, 405), dtype=torch.float32, requires_grad=True)
            target = torch.randint(0, 5, (1,), dtype=torch.long)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam([input_data], lr=0.001)

            for epoch in range(1000):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()

            recovered_data = input_data.detach().numpy().astype(int)
            recovered_data_list.append(recovered_data)
            print(f"Recovered data for sample {sample_idx} from client {client_id}")

        save_path = f"recorvered_input_plain/recovered_input_client_plain_{client_id}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(recovered_data_list, f)
        print(f"Recovered Input Data for client {client_id} saved to {save_path}")

        return np.vstack(recovered_data_list)


    def build_model(self):
        """Xây dựng mô hình với 405 đặc trưng đầu vào và 5 nhãn đầu ra"""
        model = nn.Sequential(
            nn.Linear(405, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        return model

    def save_recovered_data(self, save_path="recovered_data.pkl"):
        """Lưu lại dữ liệu đã phục hồi theo từng client"""
        with open(save_path, "wb") as f:
            pickle.dump(self.client_data_map, f)
        print(f"Recovered data saved to {save_path}")

def parameters_to_ndarrays(parameters: Parameters) -> dict:
    """Chuyển đổi tham số mô hình từ Parameters thành numpy array."""
    tensors = parameters.tensors  # Danh sách các tensor dưới dạng bytes
    return {f"layer_{i}": np.frombuffer(tensor, dtype=np.float32) for i, tensor in enumerate(tensors)}

def ndarrays_to_parameters(ndarrays: dict) -> Parameters:
    """Chuyển đổi numpy array thành Parameters."""
    tensors = [value.astype(np.float32).tobytes() for value in ndarrays.values()]
    return Parameters(tensors=tensors, tensor_type="numpy")

def attack_fn(context: Context):
    strategy = FedAttack()
    config = ServerConfig(num_rounds=1)
    return ServerAppComponents(config = config,strategy=strategy)

