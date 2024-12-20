import json
import matplotlib.pyplot as plt

# Đọc file history.json
with open(".\\outputs\\2024-12-19\\01-21-00\\history.json", "r") as f:
    history = json.load(f)

rounds = history["round"]
losses = history["loss"]
accuracies = history["accuracy"]

# Vẽ biểu đồ Loss và Accuracy trên cùng 1 biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(rounds, losses, marker="o", label="Loss", color="red")
plt.plot(rounds, accuracies, marker="o", label="Accuracy", color="green")
plt.title("Training Loss and Validation Accuracy Across Rounds")
plt.xlabel("Round")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
