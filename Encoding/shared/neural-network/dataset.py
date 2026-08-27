import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(path: str, width: int) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values.astype(np.float32)[:, 1:width + 1]

def build_dataloaders(cfg, seed: int, data_width: int):
    """
    Loads the dataset, splits into train/val.

    Returns: train_loader, val_loader
    """
    data = load_data(cfg.path, data_width)

    if data.ndim != 2 or data.shape[1] != data_width:
        raise ValueError(
            f"Expected a 2D array with {data_width} columns, got shape {data.shape}. "
            "Check the data path/format in conf/data/default.yaml."
        )

    n_val = max(1, int(len(data) * cfg.val_split))
    n_train = len(data) - n_val
    if n_train <= 0:
        raise ValueError("val_split too large for dataset size.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(data))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_arr, val_arr = data[train_idx].copy(), data[val_idx].copy()

    # Load existing normalization stats or create them
    if os.path.exists(cfg.stats_path):
        stats = np.load(cfg.stats_path)
        means = stats["means"]
        stds = stats["stds"]
    else:
        means = train_arr.mean(axis=0)
        stds = train_arr.std(axis=0)
        stds[stds == 0] = 1.0

        os.makedirs(os.path.dirname(cfg.stats_path), exist_ok=True)
        np.savez(
            cfg.stats_path,
            means=means,
            stds=stds,
        )

    train_arr = (train_arr - means) / stds
    val_arr = (val_arr - means) / stds

    train_loader = DataLoader(
        TabularDataset(train_arr),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        TabularDataset(val_arr),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
