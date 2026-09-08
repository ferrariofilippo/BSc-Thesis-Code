import hydra
from omegaconf import DictConfig

from src.predict.mlengine import IMLEngine
from src.predict.nn import NeuralNetwork
from src.predict.dt import DecisionTree
from src.predict.rf import RandomForest
from src.predict.gp import GaussianProcess
from src.predict.rsm import ResponseSurfaceMethod

MODEL_NAME = "dt"  # One of the below

MODEL_REGISTRY = {
    "nn": NeuralNetwork,
    "rf": RandomForest,
    "dt": DecisionTree,
    "gp": GaussianProcess,
    "rsm": ResponseSurfaceMethod
}

def build_model(name: str) -> IMLEngine:
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown MODEL_NAME {name!r}; expected one of: {valid}") from exc
    return model_cls("Train")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model = build_model(MODEL_NAME)
    model.train(cfg)

if __name__ == "__main__":
    main()
