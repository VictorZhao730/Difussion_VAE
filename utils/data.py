import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def load_data(h5_path, key='data', split=None, train_ratio=0.8, random_seed=42):

    with h5py.File(h5_path, 'r') as f:
        data = f[key][:]
    data = np.squeeze(data, axis=1)  # (N, 72, 53)

    np.random.seed(random_seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * train_ratio)
    if split == 'train':
        return data[indices[:split_idx]]
    elif split == 'test':
        return data[indices[split_idx:]]
    else:
        return data
    
def load_cond_data(h5_path, key='data', split=None, train_ratio=0.8, random_seed=42):
    with h5py.File(h5_path, 'r') as f:
        data = f[key][:]
    data = np.squeeze(data, axis=1)  # (N, 72, 53)

    np.random.seed(random_seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * train_ratio)
    if split == 'train':
        return data[indices[:split_idx]], indices[:split_idx]
    elif split == 'test':
        return data[indices[split_idx:]], indices[split_idx:]
    else:
        return data, indices
    
def get_loader(np_data, batch_size=64, shuffle=False):
    tensor = torch.tensor(np_data, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)

def get_cond_loader(np_data, cond, batch_size=64, shuffle=False):
    """
    np_data: (N, 72, 53)
    cond: (N, C)
    """
    tensor_data = torch.tensor(np_data, dtype=torch.float32)
    tensor_cond = torch.tensor(cond, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_cond)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_cond(csv_path, indices=None):
    df = pd.read_csv(csv_path)
    cond_df = pd.get_dummies(df[['type', 'part']])
    cond = cond_df.values.astype(np.float32)
    if indices is not None:
        cond = cond[indices]
    return cond