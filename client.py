import sys
import time
import json
import torch
import copy
import numpy as np
import flwr as fl
import tenseal as ts
from glob import glob
from phe import paillier
from torch import nn
from torch.utils.data import DataLoader
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import List
from pathlib import Path
from model import Net, get_parameters, set_parameters
from SegCKKS import *

class Args:
    def __init__(self):
        self.device = 'cpu'
        self.lr = 0.01
        self.momentum = 0.9
        self.local_ep = 10
        self.mode = 'Plain'
        self.input_size = 405
        self.num_classes = 5
        self.ckks_sec_level = 256
        self.ckks_mul_depth = 1
        self.ckks_key_len = 8192
        self.phe_key_len = 256

class FlowerClient(Client):
    def __init__(self, args, partition_id, trainloaders, valloaders, w=None, FHE=None, PhePk=None, PheSk=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.partition_id = partition_id
        self.ldr_train = trainloaders[partition_id]  
        self.ldr_val = valloaders[partition_id]

        self.model = Net(input_size=args.input_size, num_classes=args.num_classes)
        self.model.to(args.device)
        if w is not None:
            self.model.load_state_dict(w)
        if self.args.mode == 'Paillier':
            self.pub_key = PhePk
            self.priv_key = PheSk
        elif self.args.mode == 'CKKS':
            self.HE = FHE
        
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        ndarrays: List[np.ndarray] = get_parameters(self.model)

        parameters = ndarrays_to_parameters(ndarrays)

        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.model, ndarrays_original)
        self.train()
        ndarrays_updated = get_parameters(self.model)

        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.ldr_train.dataset),
            metrics={"partition_id": self.partition_id},
        )

    def train(self):
        """Hàm train cục bộ"""
        start_t_train = time.time()
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        net.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (features, labels) in enumerate(self.ldr_train):
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                
                optimizer.zero_grad()
                log_probs = self.model(features)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                
            avg_loss = sum(batch_loss) / len(batch_loss)
            print(f"[Client {self.partition_id}] Epoch {iter + 1}/{self.args.local_ep}, Loss: {avg_loss:.4f}")

        w_new = self.model.state_dict()
        print("Client train time:", time.time() - start_t_train)

        update_w = self._compute_weight_update(w_new, w_old)
        
        return update_w

    def _compute_weight_update(self, w_new, w_old):
        update_w = {}
        if self.args.mode == 'Plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
        elif self.args.mode == 'Paillier':
            update_w = self._encrypt_paillier(w_new, w_old)
        elif self.args.mode == 'CKKS':
            update_w = self._encrypt_ckks(w_new, w_old)
        else:
            raise NotImplementedError("Unsupported encryption mode")
        return update_w

    def _encrypt_paillier(self, w_new, w_old):
        print('Paillier encrypting...')
        enc_start = time.time()
        update_w = {}
        for k in w_new.keys():
            update_w[k] = w_new[k] - w_old[k]
            list_w = update_w[k].view(-1).cpu().tolist()
            update_w[k] = [self.pub_key.encrypt(round(elem, 3)) for elem in list_w]
        print('Encryption time:', time.time() - enc_start)
        return update_w

    def _encrypt_ckks(self, w_new, w_old):
        print('CKKS encrypting...')
        enc_start = time.time()
        update_w = {}
        slot_count = self.HE.poly_modulus_degree // 2 
        
        for k in w_new.keys():
            update_w[k] = w_new[k] - w_old[k]
            list_w = update_w[k].view(-1).cpu().tolist()
            list_w = np.array(list_w)

            if len(list_w) <= slot_count:
                encrypted_vector = enc_vector(self.HE, list_w)
            else:
                encrypted_vector = seg_enc_vector(self.HE, list_w, len(list_w))

            update_w[k] = [enc_block.serialize() for enc_block in encrypted_vector] \
                if isinstance(encrypted_vector, list) else encrypted_vector.serialize()

        print('Encryption time:', (time.time() - enc_start))
        return update_w

    def update(self, w_glob):
        if self.args.mode == 'Plain':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Paillier':
            self._decrypt_paillier(w_glob)
        elif self.args.mode == 'CKKS':
            self._decrypt_ckks(w_glob)
        else:
            raise NotImplementedError("Unsupported encryption mode")

    def _decrypt_paillier(self, w_glob):
        print('Paillier decrypting...')
        dec_start = time.time()
        for k in w_glob.keys():
            decrypted = [self.priv_key.decrypt(elem) for elem in w_glob[k]]
            origin_shape = list(self.model.state_dict()[k].size())
            self.model.state_dict()[k] += torch.FloatTensor(decrypted).to(self.args.device).view(*origin_shape)
        print('Decryption time:', time.time() - dec_start)

    def _decrypt_ckks(self, w_glob):
        print('CKKS decrypting...')
        dec_start = time.time()
        for k in w_glob.keys():
            origin_shape = list(self.model.state_dict()[k].size())
            if isinstance(w_glob[k], list):
                dec_vec = seg_dec_vector(self.HE, w_glob[k])
            else:
                dec_vec = dec_vector(self.HE, w_glob[k])

            vlen = np.prod(origin_shape)
            dec_vec = dec_vec[:vlen]
            self.model.state_dict()[k] += torch.FloatTensor(dec_vec).to(self.args.device).view(*origin_shape)
        print('Decryption time:', time.time() - dec_start)
        
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = ins.parameters
        ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(self.model, ndarrays)

        self.model.eval()
        loss, correct = 0.0, 0
        total = len(self.ldr_val.dataset)
        with torch.no_grad():
            for features, labels in self.ldr_val:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                outputs = self.model(features)
                loss += self.loss_func(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss /= len(self.ldr_val)

        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=loss,
            num_examples=total,
            metrics={"accuracy": accuracy}
        )

def client_fn(context: fl.common.Context, trainloaders, valloaders) -> Client:
    partition_id = int(context.node_config["partition-id"])
    
    args = Args()

    if args.mode == "CKKS":
        HE = generate_ckks_key(args.ckks_sec_level, args.ckks_mul_depth, args.ckks_key_len)
    elif args.mode == "Paillier":
        phe_pk, phe_sk = paillier.generate_paillier_keypair(n_length=args.phe_key_len)
        
    print(f"Client {partition_id} is using {args.mode} encryption")

    model = Net(input_size=args.input_size, num_classes=args.num_classes)
    FHE=None
    PhePk = None
    PheSk = None
    if args.mode == "Plain":
        w=copy.deepcopy(model.state_dict())
    if args.mode == "CKKS":
        w=copy.deepcopy(model.state_dict())
        FHE=HE
    elif args.mode == "Paillier":
        w=copy.deepcopy(model.state_dict())
        PhePk=phe_pk
        PheSk=phe_sk
    
    return FlowerClient(
        args=args,
        partition_id=partition_id,
        trainloaders=trainloaders,
        valloaders=valloaders,
        w=w,
        FHE=FHE,
        PhePk=PhePk,
        PheSk=PheSk
    )
