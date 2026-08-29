import numpy as np

def load_standardizations(path, n):
    data = np.load(path)
    means = data["means"].astype(np.float32)[:n]
    stds = data["stds"].astype(np.float32)[:n]
    return means, stds
