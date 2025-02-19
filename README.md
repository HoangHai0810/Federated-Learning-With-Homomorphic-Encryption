# Federated Learning with Homomorphic Encryption & Blockchain

## Introduction

The Federated Learning with Homomorphic Encryption & Blockchain project integrates three cutting-edge technologies:

- Federated Learning: Train machine learning models on decentralized client data without sharing raw data.
- Homomorphic Encryption: Encrypt data in a way that allows computation on the encrypted values without decryption, ensuring data privacy.
- Blockchain: Store the global model on a blockchain after each aggregation step, ensuring immutability, transparency, and a secure history of model updates.

By combining these technologies, the project aims to maximize data privacy while ensuring the integrity and traceability of the trained models.

## Goals
- Data Security: Use Homomorphic Encryption to encrypt client data, enabling computations on encrypted data without exposing sensitive information.
- Decentralized Training: Employ Federated Learning to train a global model from distributed data across multiple clients.
- Transparency and Integrity: Integrate blockchain to record and store the global model after every aggregation, ensuring a tamper-proof history of model updates.

## How It Works
This is the system deployment diagram:
![QuyTrinh](https://github.com/user-attachments/assets/33e00ed5-8fc0-45f7-96e8-8b1d8f5ec65e)
## Requirements
- Python: Version 3.7 or higher.
- Required Main Libraries:
  - flwr
  - numpy
  - pandas
  - scikit-learn
  - phe
  - tenseal
  - hashlib
## Installation
#### 1. Clone the repository:

```bash
git clone https://github.com/HoangHai0810/Federated-Learning-With-Homomorphic-Encryption.git
cd Federated-Learning-With-Homomorphic-Encryption
```
#### 2. Create and activate a virtual environment:
On Unix/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
#### 3.Install the required libraries:
```bash
pip install -r requirements.txt
```
## Usage
Run the training process:

```bash
python main.py
```

## Contact
For any questions or suggestions, please contact:

Author: Hoàng Hải Anh
Email: hoanghaianh0810@gmail.com
