import torch.nn as nn
import torch
import flwr as fl
import torch.optim as optim
import time


class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


def train(net, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad() 
            output = net(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    
    with torch.no_grad():  
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features) 
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset) 
    return loss, accuracy

def model_to_parameters(model):
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays


def parameters_to_model(model, parameters):
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict)
    return model