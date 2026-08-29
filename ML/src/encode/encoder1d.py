"""One-dimensional Chebyshev autoencoder."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .encoder import Array, IEncoder, chebyshev_dct_axis
from .load_standardizations import load_standardizations


MODEL = "./conf/1d/best_model.pt"
STANDARDIZATION = "./conf/1d/standardization.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder1D(IEncoder):
    """Encode degree-15 Chebyshev expansions sampled at 128 nodes."""

    def __init__(self) -> None:
        sample_count = 128
        max_chebyshev_degree = 15
        coefficient_count = max_chebyshev_degree + 1

        checkpoint = torch.load(MODEL, map_location=DEVICE, weights_only=False)
        means, stds = load_standardizations(STANDARDIZATION, coefficient_count)

        super().__init__(
            sample_count,
            max_chebyshev_degree,
            0.0,
            1.0,
            DEVICE,
            means,
            stds,
            checkpoint,
            0.0,
            0.1,
        )

    def _get_evaluations(self, function: Callable[[float], float]) -> Array:
        evaluations = np.asarray(
            [function(float(x)) for x in self.chebyshev_nodes], dtype=float
        )
        if evaluations.shape != (self.sample_count,):
            raise ValueError(
                f"the 1D function must produce {self.sample_count} scalar values"
            )
        if not np.all(np.isfinite(evaluations)):
            raise ValueError("the 1D function produced non-finite values")
        return evaluations

    def _get_chebyshev_coefficients(self, evaluations: Array) -> Array:
        coefficients = chebyshev_dct_axis(evaluations, axis=0)
        retained_count = self.max_chebyshev_degree + 1
        return coefficients[:retained_count].copy()
