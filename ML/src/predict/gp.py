from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from .mlengine import IMLEngine

log = logging.getLogger(__name__)

class GaussianProcess(IMLEngine):
    PREDICTION_MODEL_PATH = "./conf/tolerance-prediction/gp.joblib"

    def __init__(self, mode: str = "Predict"):
        super().__init__(mode)

    def _split_features_target(self, data: np.ndarray):
        return data[:, 1:], data[:, 0]

    def _build_kernel(self, cfg: DictConfig):
        # Constant * RBF + White noise is a solid, common default
        kernel = (
            ConstantKernel(cfg.model.constant_value, cfg.model.constant_bounds)
            # * RBF(cfg.model.length_scale, cfg.model.length_scale_bounds)
            * Matern(cfg.model.length_scale, cfg.model.length_scale_bounds, nu=1.5)
            + WhiteKernel(cfg.model.noise_level, cfg.model.noise_level_bounds)
        )
        return kernel

    def predict(self, inputs) -> float:
        self._load_standardizations()
        model = joblib.load(self.PREDICTION_MODEL_PATH)

        stdized = (inputs - self.means[1:]) / self.stds[1:]
        pred, std = model.predict(stdized.reshape(1, -1), return_std=True)

        mean = pred[0] * self.stds[0] + self.means[0]
        uncertainty = std[0] * self.stds[0]

        return mean, uncertainty

    def train(self, cfg: DictConfig) -> float:
        run_dir = HydraConfig.get().runtime.output_dir
        train_arr, val_arr = self._get_data_splits(cfg.seed, cfg.data.val_split)

        means = train_arr.mean(axis=0)
        stds = train_arr.std(axis=0)
        stds[stds == 0] = 1.0
        train_arr = (train_arr - means) / stds
        val_arr = (val_arr - means) / stds

        X_train, y_train = self._split_features_target(train_arr)
        X_val, y_val = self._split_features_target(val_arr)

        kernel = self._build_kernel(cfg)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=cfg.model.n_restarts_optimizer,
            normalize_y=False,
            random_state=cfg.seed,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, train_pred)

        val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, val_pred)

        ckpt_path = os.path.join(run_dir, "best_model.joblib")
        joblib.dump(model, ckpt_path)

        log.info(
            f"Done. train_loss={train_loss:.6f}, val_loss={val_loss:.6f}. "
            f"Learned kernel: {model.kernel_}. Model saved: {ckpt_path}"
        )

        return val_loss