"""Predicts the best tolerance for a PDE problem given its coefficients.

Edit the PROBLEM_CONFIG / MODEL_NAME constants below to change the problem
being solved.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.encode.encoder1d import Encoder1D
from src.encode.encoder2d import Encoder2D
from src.math.functions import Functions
from src.predict.nn import NeuralNetwork
from src.predict.rf import RandomForest

##############################   PROBLEM CONFIG   ##############################
# Each entry is (function_kind, magnitude). Edit these to change the problem.
PROBLEM_CONFIG: dict[str, tuple[str, float]] = {
    "mu": ("Costante", 1e-1),
    "b1": ("Costante", 1.0),
    "b2": ("Costante", 0.0),
    "sigma": ("Costante", 0.0),
    "forcing": ("Funzione 8", 1.0),
    "bc1": ("Costante", 0.0),
    "bc2": ("Costante", 0.0),
    "bc3": ("Costante", 0.0),
    "bc4": ("Costante", 0.0),
}

ALPHA = 1e-1
DESIRED_ERROR = 1e-3

# Which PROBLEM_CONFIG entries are 2D functions vs 1D boundary functions
# (1D ones also need their border index, 1-4).
FIELDS_2D: Sequence[str] = ("mu", "b1", "b2", "sigma", "forcing")
FIELDS_1D_BORDERS: dict[str, int] = {"bc1": 1, "bc2": 2, "bc3": 3, "bc4": 4}

##############################   ML MODEL CONFIG   #############################
MODEL_NAME = "nn"  # "nn" or "rf"

MODEL_REGISTRY = {
    "nn": NeuralNetwork,
    "rf": RandomForest,
}

##############################   BUSINESS LOGIC   ##############################
def build_pde_params(config: dict[str, tuple[str, float]], alpha: float) -> dict[str, float]:
    return {
        "mu": config["mu"][1],
        "b1": config["b1"][1],
        "b2": config["b2"][1],
        "sigma": config["sigma"][1],
        "alpha": alpha,
    }


def build_model(name: str):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown MODEL_NAME {name!r}; expected one of: {valid}") from exc
    return model_cls()

def get_encodings(
    config: dict[str, tuple[str, float]],
    math_helper: Functions,
    encoder_1d: Encoder1D,
    encoder_2d: Encoder2D,
) -> np.ndarray:
    chunks: list[np.ndarray] = []

    for field in FIELDS_2D:
        kind, magnitude = config[field]
        func = math_helper.get_by_name_2d(kind, magnitude)
        chunks.append(encoder_2d.get_encoding(func))

    for field, border in FIELDS_1D_BORDERS.items():
        kind, magnitude = config[field]
        func = math_helper.get_by_name_1d(kind, magnitude, border)
        chunks.append(encoder_1d.get_encoding(func))

    return np.concatenate(chunks)

def main() -> None:
    ml_engine = build_model(MODEL_NAME)

    math_helper = Functions()
    math_helper.update_params(build_pde_params(PROBLEM_CONFIG, ALPHA))

    encoder_1d = Encoder1D()
    encoder_2d = Encoder2D()

    encodings = get_encodings(PROBLEM_CONFIG, math_helper, encoder_1d, encoder_2d)
    features = np.insert(encodings, 0, DESIRED_ERROR)

    tolerance = ml_engine.predict(features)
    print(f"Best predicted tolerance: {tolerance:.6f}")

if __name__ == "__main__":
    main()
