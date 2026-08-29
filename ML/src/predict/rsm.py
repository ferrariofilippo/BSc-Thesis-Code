from .mlengine import IMLEngine

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
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
        self._load_standardizations()
        model = joblib.load(self.PREDICTION_MODEL_PATH)

        stdized = (inputs - self.means[1:]) / self.stds[1:]
        predicted = model.predict(stdized.reshape(1, -1))[0]

        return predicted * self.stds[0] + self.means[0]

    def train(self, cfg: DictConfig) -> float:
        run_dir = HydraConfig.get().runtime.output_dir
        train_arr, val_arr = self._get_data_splits(cfg.seed, cfg.data.val_split)

        means = train_arr.mean(axis=0)
        stds = train_arr.std(axis=0)
        stds[stds == 0] = 1.0

        train_arr = (train_arr - means) / stds
        val_arr = (val_arr - means) / stds

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
        train_loss = mean_squared_error(y_train, train_pred)

        val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, val_pred)

        ckpt_path = os.path.join(run_dir, "best_model.joblib")
        joblib.dump(model, ckpt_path)

        n_terms = model.named_steps["poly"].n_output_features_
        log.info(
            f"Done. train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"n_poly_terms={n_terms}. Model saved: {ckpt_path}"
        )

        return val_loss