from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
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

    def predict(self, inputs) -> tuple[float, float]:
        """Return the lognormal predictive mean and std in tolerance units."""
        model, preprocessing = self._load_regression_checkpoint(self.PREDICTION_MODEL_PATH)
        standardized = self._standardize_prediction_inputs(inputs, preprocessing)
        pred, std = model.predict(standardized.reshape(1, -1), return_std=True)

        # A Gaussian prediction in log space becomes lognormal in raw units.
        # Scaling std linearly (as with the old raw target) would be incorrect.
        target_mean = float(preprocessing["means"][0])
        target_std = float(preprocessing["stds"][0])
        natural_log_mean = np.log(10.0) * (float(pred[0]) * target_std + target_mean)
        natural_log_variance = (np.log(10.0) * float(std[0]) * target_std) ** 2
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mean = np.exp(natural_log_mean + 0.5 * natural_log_variance)
            uncertainty = mean * np.sqrt(np.expm1(natural_log_variance))
        if not np.isfinite(mean) or mean <= 0 or not np.isfinite(uncertainty):
            raise ValueError("GP prediction is invalid after inverse log scaling.")
        return float(mean), float(uncertainty)

    def train(self, cfg: DictConfig) -> float:
        run_dir = HydraConfig.get().runtime.output_dir
        train_arr, val_arr, preprocessing = self._prepare_standardized_splits(
            cfg.seed, cfg.data.val_split
        )

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
        train_loss = self._regression_loss(
            y_train, train_pred, cfg.training.loss, cfg.training.huber_beta
        )

        val_pred = model.predict(X_val)
        val_loss = self._regression_loss(
            y_val, val_pred, cfg.training.loss, cfg.training.huber_beta
        )

        ckpt_path = os.path.join(run_dir, "best_model.joblib")
        joblib.dump({
            "model": model,
            "preprocessing": preprocessing,
            "loss": cfg.training.loss,
            "huber_beta": cfg.training.huber_beta,
            "val_loss": val_loss,
        }, ckpt_path)

        log.info(
            f"Done. train_loss={train_loss:.6f}, val_loss={val_loss:.6f}. "
            f"metric={cfg.training.loss} on standardized log10(tol). "
            f"Learned kernel: {model.kernel_}. Model saved: {ckpt_path}"
        )

        return val_loss
