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
    PREPROCESSING_VERSION = 1

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
        train_arr, val_arr, preprocessing = self._prepare_standardized_splits(
            seed, cfg.val_split
        )

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

        return train_loader, val_loader, preprocessing

    def _prepare_standardized_splits(
        self, seed: int, val_split: float
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        train_arr, val_arr = self._get_data_splits(seed, val_split)

        # Column 0 is the tolerance target and column 1 is the target error.
        # Both quantities are strictly positive and span several orders of
        # magnitude, so transform them before fitting the scaler.
        train_arr = self._log_transform_target_and_error(train_arr)
        val_arr = self._log_transform_target_and_error(val_arr)

        # Statistics are always fitted on the training split and stored with
        # the trained model, never recovered from a shared sidecar file.
        means = train_arr.mean(axis=0, dtype=np.float64).astype(np.float32)
        stds = train_arr.std(axis=0, dtype=np.float64).astype(np.float32)
        stds[stds == 0] = 1.0

        preprocessing = {
            "version": self.PREPROCESSING_VERSION,
            "transform": "log10_target_and_error",
            "means": means,
            "stds": stds,
            "error_feature_index": 0,
            "error_feature_name": "err",
        }
        return (
            (train_arr - means) / stds,
            (val_arr - means) / stds,
            preprocessing,
        )

    @staticmethod
    def _log_transform_target_and_error(data: np.ndarray) -> np.ndarray:
        transformed = np.asarray(data, dtype=np.float32).copy()
        if transformed.ndim != 2 or transformed.shape[1] < 2:
            raise ValueError(
                "Expected a 2D array containing tolerance and error columns."
            )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Training data must contain only finite values.")
        if np.any(transformed[:, 0] <= 0):
            raise ValueError("Tolerance values must be positive for log10 scaling.")
        if np.any(transformed[:, 1] <= 0):
            raise ValueError("Error values must be positive for log10 scaling.")

        transformed[:, 0] = np.log10(transformed[:, 0])
        transformed[:, 1] = np.log10(transformed[:, 1])
        return transformed

    def _standardize_prediction_inputs(
        self, inputs: np.ndarray, preprocessing: dict
    ) -> np.ndarray:
        if (
            preprocessing.get("version") != self.PREPROCESSING_VERSION
            or preprocessing.get("transform") != "log10_target_and_error"
        ):
            raise ValueError(
                "Unsupported checkpoint preprocessing version: "
                f"{preprocessing.get('version')!r}."
            )

        means = np.asarray(preprocessing["means"], dtype=np.float32)
        stds = np.asarray(preprocessing["stds"], dtype=np.float32)
        transformed = np.asarray(inputs, dtype=np.float32).copy()
        if transformed.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional feature vector, got {transformed.shape}."
            )
        if len(means) != transformed.size + 1 or len(stds) != len(means):
            raise ValueError(
                "Checkpoint preprocessing statistics do not match the model inputs."
            )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)) or np.any(stds <= 0):
            raise ValueError("Checkpoint preprocessing contains invalid scaling statistics.")
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Prediction inputs must contain only finite values.")

        error_index = int(preprocessing["error_feature_index"])
        if error_index < 0 or error_index >= transformed.size:
            raise ValueError("Checkpoint error-feature index is outside the input vector.")
        if transformed[error_index] <= 0:
            raise ValueError("Desired error must be positive for log10 scaling.")
        transformed[error_index] = np.log10(transformed[error_index])
        return (transformed - means[1:]) / stds[1:]

    @staticmethod
    def _inverse_standardized_log_target(
        predictions: np.ndarray | float, preprocessing: dict
    ) -> np.ndarray | float:
        means = np.asarray(preprocessing["means"], dtype=np.float32)
        stds = np.asarray(preprocessing["stds"], dtype=np.float32)
        standardized = np.asarray(predictions, dtype=np.float64)
        log_tolerance = standardized * float(stds[0]) + float(means[0])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            tolerances = np.power(10.0, log_tolerance)
        if np.any(~np.isfinite(tolerances)) or np.any(tolerances <= 0):
            raise ValueError("Predicted tolerance is invalid after inverse log scaling.")
        if tolerances.ndim == 0:
            return float(tolerances)
        return tolerances

    @staticmethod
    def _regression_loss(
        targets: np.ndarray,
        predictions: np.ndarray,
        loss_name: str,
        huber_beta: float,
    ) -> float:
        """Mean loss on standardized log10 targets, matching the NN metric.

        Sklearn estimators use this for reporting/model selection; their native
        fitting objectives are not replaced by this scoring function.
        """
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        if targets.shape != predictions.shape or targets.size == 0:
            raise ValueError("Targets and predictions must have the same non-empty shape.")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
            raise ValueError("Loss inputs must contain only finite values.")
        residual = np.abs(predictions - targets)
        if loss_name == "mse":
            return float(np.mean(np.square(residual)))
        if loss_name != "smooth_l1":
            raise ValueError(f"Unknown loss: {loss_name}")
        if not np.isfinite(huber_beta) or huber_beta < 0:
            raise ValueError("huber_beta must be finite and non-negative.")
        if huber_beta == 0:
            return float(np.mean(residual))

        losses = np.where(
            residual < huber_beta,
            0.5 * np.square(residual) / huber_beta,
            residual - 0.5 * huber_beta,
        )
        return float(np.mean(losses))

    @staticmethod
    def _load_regression_checkpoint(path: str) -> tuple[object, dict]:
        import joblib

        checkpoint = joblib.load(path)
        if not isinstance(checkpoint, dict) or not {
            "model", "preprocessing"
        }.issubset(checkpoint):
            raise ValueError(
                "Checkpoint does not contain log-space preprocessing statistics. "
                "Retrain the estimator with the current pipeline before prediction."
            )
        return checkpoint["model"], checkpoint["preprocessing"]

    def _save_loss_plot(
        self, train_history: list, val_history: list, run_dir: str
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        plt.figure(figsize=(10, 6))
        plt.plot(train_history, label="Train Loss", color="#ff8945")
        plt.plot(val_history, label="Validation Loss", color="#006064")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (standardized log10 tolerance)")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = os.path.join(run_dir, "loss_curves.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()  # Critical to avoid memory leaks over hundreds of sweep jobs
