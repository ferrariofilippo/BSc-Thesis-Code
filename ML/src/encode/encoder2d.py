"""Two-dimensional tensor-product Chebyshev autoencoder."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .encoder import Array, IEncoder, chebyshev_dct_axis
from .load_standardizations import load_standardizations


MODEL = "./conf/2d/best_model.pt"
STANDARDIZATION = "./conf/2d/standardization.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder2D(IEncoder):
    """Encode degree-15-by-15 Chebyshev expansions sampled on a 64x64 grid."""

    def __init__(self) -> None:
        sample_count = 64
        max_chebyshev_degree = 15
        coefficient_count = (max_chebyshev_degree + 1) ** 2

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
            0.0,
        )

    def _get_evaluations(
        self, function: Callable[[float, float], float]
    ) -> Array:
        nodes = self.chebyshev_nodes
        evaluations = np.asarray(
            [
                [function(float(x), float(y)) for y in nodes]
                for x in nodes
            ],
            dtype=float,
        )
        expected_shape = (self.sample_count, self.sample_count)
        if evaluations.shape != expected_shape:
            raise ValueError(
                f"the 2D function must produce a {self.sample_count}x"
                f"{self.sample_count} grid of scalar values"
            )
        if not np.all(np.isfinite(evaluations)):
            raise ValueError("the 2D function produced non-finite values")
        return evaluations

    def _get_chebyshev_coefficients(self, evaluations: Array) -> Array:
        coefficients = chebyshev_dct_axis(evaluations, axis=0)
        coefficients = chebyshev_dct_axis(coefficients, axis=1)
        retained_count = self.max_chebyshev_degree + 1
        retained = coefficients[:retained_count, :retained_count]
        return retained.ravel(order="C").copy()
