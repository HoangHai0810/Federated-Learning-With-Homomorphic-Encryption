import json
import matplotlib.pyplot as plt

# Đọc các file history_plain của MNIST, CIFAR-10 và CIC
datasets = ["history_mnist/history_ckks.json", "history_cifar/history_ckks.json", "history_cic/history_ckks.json"]
labels = ["MNIST", "CIFAR-10", "CIC"]
colors = ["red", "green", "blue"]

# Vẽ biểu đồ Loss và Accuracy cho từng dataset
plt.figure(figsize=(10, 6))

for dataset, label, color in zip(datasets, labels, colors):
    with open(dataset, "r") as f:
        history = json.load(f)
    
    rounds = history["round"]
    losses = history["loss"]
    accuracies = history["accuracy"]
    
    # Vẽ Loss và Accuracy cho mỗi dataset
    plt.plot(rounds, losses, marker="o", label=f"{label} Loss", color=color)
    plt.plot(rounds, accuracies, marker="o", label=f"{label} Accuracy", linestyle="--", color=color)

# Thêm các thông tin vào biểu đồ
plt.title("Training Loss and Validation Accuracy Across Rounds (CKKS)")
plt.xlabel("Round")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Hiển thị biểu đồ
plt.show()
