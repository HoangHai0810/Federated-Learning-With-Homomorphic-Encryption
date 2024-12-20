import pickle

with open('recovered_input_client_unknown_client.pkl', "rb") as f:
    client_data_map = pickle.load(f)

for client_id, data in client_data_map.item():
    print(f"Client {client_id} data:")
    print(data)  # In ra dữ liệu phục hồi của client này
