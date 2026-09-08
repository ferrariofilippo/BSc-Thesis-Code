from .mlengine import IMLEngine

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class ResponseSurfaceMethod(IMLEngine):
    PREDICTION_MODEL_PATH = "./conf/tolerance-prediction/rsm.joblib"

    def __init__(self, mode: str = "Predict"):
        super().__init__(mode)

    def _split_features_from_target(self, data):
        return data[:, 1:], data[:, 0]

    def predict(self, inputs) -> float:
        model, preprocessing = self._load_regression_checkpoint(self.PREDICTION_MODEL_PATH)
        standardized = self._standardize_prediction_inputs(inputs, preprocessing)
        predicted = model.predict(standardized.reshape(1, -1))[0]
        return self._inverse_standardized_log_target(predicted, preprocessing)

    def train(self, cfg: DictConfig) -> float:
        run_dir = HydraConfig.get().runtime.output_dir
        train_arr, val_arr, preprocessing = self._prepare_standardized_splits(
            cfg.seed, cfg.data.val_split
        )

        X_train, y_train = self._split_features_from_target(train_arr)
        X_val, y_val = self._split_features_from_target(val_arr)

        # classic RSM: degree-2 polynomial (linear + quadratic + interaction
        # terms). interaction_only=False keeps squared terms (x1^2, x2^2...),
        # which is the standard RSM formulation, not just pairwise interactions.
        model = Pipeline([
            ("poly", PolynomialFeatures(
                degree=cfg.model.degree,
                interaction_only=cfg.model.interaction_only,
                include_bias=False,
            )),
            # Ridge instead of plain LinearRegression: degree-2 expansion
            # with many original features creates a lot of correlated terms
            # (e.g. x1^2 and x1*x2 are often correlated), so some
            # regularization avoids an unstable/overfit fit, especially
            # with a small dataset.
            ("reg", Ridge(alpha=cfg.model.alpha)),
        ])
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

        n_terms = model.named_steps["poly"].n_output_features_
        log.info(
            f"Done. train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"metric={cfg.training.loss} on standardized log10(tol), "
            f"n_poly_terms={n_terms}. Model saved: {ckpt_path}"
        )

        return val_loss
