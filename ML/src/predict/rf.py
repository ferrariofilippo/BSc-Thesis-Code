from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
import joblib
import os 
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from .mlengine import IMLEngine

log = logging.getLogger(__name__)

class RandomForest(IMLEngine):
    PREDICTION_MODEL_PATH = "./conf/tolerance-prediction/rf.joblib"

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
        stds[stds==0] = 1.0

        train_arr = (train_arr - means) / stds
        val_arr = (val_arr - means) / stds

        X_train, y_train = self._split_features_from_target(train_arr)
        X_val, y_val = self._split_features_from_target(val_arr)

        model = RandomForestRegressor(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            min_samples_split=cfg.model.min_samples_split,
            min_samples_leaf=cfg.model.min_samples_leaf,
            max_features=cfg.model.max_features,
            n_jobs=cfg.model.n_jobs,
            random_state=cfg.seed,
            oob_score=True,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, train_pred)

        val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, val_pred)

        ckpt_path = os.path.join(run_dir, "best_model.joblib")
        joblib.dump(model, ckpt_path)

        log.info(
            f"Done. train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, oob_score={model.oob_score_:.4f}. "
            f"Model saved: {ckpt_path}"
        )

        return val_loss
        