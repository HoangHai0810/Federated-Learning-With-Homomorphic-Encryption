import pandas as pd
import matplotlib.pyplot as plt

file_path = 'match.csv'
df = pd.read_csv(file_path)

averages = {
    "plainClient": df["plainClient"].mean(),
    "paillier": df["paillier"].mean(),
    "ckks": df["ckks"].mean()
}

plt.bar(averages.keys(), averages.values(), color=['blue', 'orange', 'green'])
plt.title("So sánh hiệu suất trung bình")
plt.ylabel("Giá trị trung bình")
plt.xlabel("Phương pháp")
plt.show()
