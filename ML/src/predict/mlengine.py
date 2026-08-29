import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

class TabularDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.targets = torch.as_tensor(data[:, 0:1], dtype=torch.float32)
        self.features = torch.as_tensor(data[:, 1:], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class IMLEngine:
    STANDARDIZATIONS_PATH = "./data/standardization.npz"
    ROWS_PER_PROBLEM = 5

    def __init__(self, mode: str = "Predict"):
        self.mode = mode

    def predict(self, params) -> float:
        if self.mode == "Train":
            raise Exception("Bad Configuration Exception. The model is supposed to be trained, not to predict!")
        
        raise Exception("Not implemented")

    def train(self, cfg: DictConfig) -> float:
        if self.mode == "Predict":
            raise Exception("Bad Configuration Exception. The model is supposed to predict, not to be trained!")

        raise Exception("Not implemented")

    def _load_standardizations(self):
        data = np.load(self.STANDARDIZATIONS_PATH)
        self.means = data["means"].astype(np.float32)
        self.stds = data["stds"].astype(np.float32)

    def _load_training_data(self) -> np.ndarray:
        df = pd.read_csv( "./data/preprocessed.csv")
        return df.values.astype(np.float32)[:, :]

    def _get_data_splits(self, seed, split):
        data = self._load_training_data()
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got shape {data.shape}. "
                "Check the data path/format in conf/data/default.yaml."
            )

        n_rows = len(data)
        if n_rows % self.ROWS_PER_PROBLEM != 0:
            raise ValueError(
                f"Expected number of rows ({n_rows}) to be a multiple of "
                f"{self.ROWS_PER_PROBLEM} ({self.ROWS_PER_PROBLEM} tolerances per problem)."
            )

        n_problems = n_rows // self.ROWS_PER_PROBLEM

        n_val_problems = max(1, int(n_problems * split))
        n_train_problems = n_problems - n_val_problems
        if n_train_problems <= 0:
            raise ValueError("val_split too large for number of problems.")

        rng = np.random.RandomState(seed)
        problem_perm = rng.permutation(n_problems)
        train_problem_idx = problem_perm[:n_train_problems]
        val_problem_idx = problem_perm[n_train_problems:]

        def problems_to_row_indices(problem_idx):
            # each problem_idx p corresponds to rows [p*5, p*5+1, ..., p*5+4]
            row_idx = (
                problem_idx[:, None] * self.ROWS_PER_PROBLEM
                + np.arange(self.ROWS_PER_PROBLEM)[None, :]
            ).ravel()
            return row_idx

        train_row_idx = problems_to_row_indices(train_problem_idx)
        val_row_idx = problems_to_row_indices(val_problem_idx)

        train_arr = data[train_row_idx].copy()
        val_arr = data[val_row_idx].copy()

        return train_arr, val_arr

    def _build_dataloaders(self, cfg, seed: int):
        self.STANDARDIZATIONS_PATH = f"./data/standardization_{seed}.npz"
        train_arr, val_arr = self._get_data_splits(seed, cfg.val_split)

        # Load existing normalization stats or create them
        if os.path.exists(self.STANDARDIZATIONS_PATH):
            stats = np.load(self.STANDARDIZATIONS_PATH)
            means = stats["means"]
            stds = stats["stds"]
        else:
            means = train_arr.mean(axis=0)
            stds = train_arr.std(axis=0)
            stds[stds == 0] = 1.0

            os.makedirs(os.path.dirname(self.STANDARDIZATIONS_PATH), exist_ok=True)
            np.savez(
                self.STANDARDIZATIONS_PATH,
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

    def _save_loss_plot(
        self, train_history: list, val_history: list, run_dir: str
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        plt.figure(figsize=(10, 6))
        plt.plot(train_history, label="Train Loss", color="#ff8945")
        plt.plot(val_history, label="Validation Loss", color="#006064")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = os.path.join(run_dir, "loss_curves.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()  # Critical to avoid memory leaks over hundreds of sweep jobs
        