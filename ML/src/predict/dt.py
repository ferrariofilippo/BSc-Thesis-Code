from sklearn.tree import DecisionTreeRegressor
import joblib
import os 
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .mlengine import IMLEngine

log = logging.getLogger(__name__)

class DecisionTree(IMLEngine):
    PREDICTION_MODEL_PATH = "./conf/tolerance-prediction/dt.joblib"

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

        model = DecisionTreeRegressor(
            max_depth=cfg.model.max_depth,
            min_samples_split=cfg.model.min_samples_split,
            min_samples_leaf=cfg.model.min_samples_leaf,
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
            f"metric={cfg.training.loss} on standardized log10(tol). Model saved: {ckpt_path}"
        )

        return val_loss
