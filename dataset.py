import torch
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
import time


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe.drop(columns=['Label']).values 
        self.labels = dataframe['Label'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def explode_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
        expanded_columns = pd.DataFrame(
            df[col].tolist(), 
            index=df.index
        ).fillna(0).astype(int)
        expanded_columns.columns = [f"{col}_{i}" for i in range(expanded_columns.shape[1])]

        df = pd.concat([df, expanded_columns], axis=1)
        df = df.drop(columns=[col])
    return df

def prepare_dataset(num_partitions, batch_size, val_ratio=0.1):
    df = pd.read_csv('./data/final_processed.csv')

    df = df.drop(columns=['index'])
    df = explode_columns(df, ['Permissions', 'Activities', 'Services', 'Receivers'])

    X = df.drop(columns=['Label'])
    y = df['Label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.reset_index(drop=True)
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = pd.DataFrame(y_test).reset_index(drop=True)
    
    trainset = pd.concat([X_train, y_train], axis=1)
    testset = pd.concat([X_test, y_test], axis=1)
    
    trainset.rename(columns={trainset.columns[-1]: 'Label'}, inplace=True)
    testset.rename(columns={testset.columns[-1]: 'Label'}, inplace=True)
    
    train_dataset = CustomDataset(trainset)
    test_dataset = CustomDataset(testset)

    num_rows = len(train_dataset) // num_partitions
    partition_len = [num_rows] * num_partitions
    partition_len[-1] += len(train_dataset) - sum(partition_len)

    trainsets = random_split(train_dataset, partition_len, torch.Generator().manual_seed(2024))

    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2024))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))
        
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloaders, valloaders, testloader