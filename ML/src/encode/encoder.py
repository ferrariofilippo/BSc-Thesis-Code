"""Shared functionality for Chebyshev-coefficient autoencoders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.fft import dct

from .model import Autoencoder


Array = np.ndarray


def chebyshev_gauss_lobatto_nodes(
    sample_count: int, range_start: float, range_end: float
) -> Array:
    """Return Chebyshev--Gauss--Lobatto nodes mapped to the target interval."""

    if sample_count < 2:
        raise ValueError("a DCT-I requires at least two samples")
    if range_end <= range_start:
        raise ValueError("range_end must be greater than range_start")

    indices = np.arange(sample_count, dtype=float)
    standard_nodes = np.cos(np.pi * indices / (sample_count - 1))
    midpoint = 0.5 * (range_start + range_end)
    half_width = 0.5 * (range_end - range_start)
    return midpoint + half_width * standard_nodes


def chebyshev_dct_axis(values: Array, axis: int) -> Array:
    """Compute correctly normalized Chebyshev coefficients along one axis."""

    values = np.asarray(values, dtype=float)
    sample_count = values.shape[axis]
    if sample_count < 2:
        raise ValueError("a DCT-I requires at least two samples")

    max_transform_degree = sample_count - 1
    coefficients = dct(values, type=1, axis=axis) / max_transform_degree

    first = [slice(None)] * coefficients.ndim
    last = [slice(None)] * coefficients.ndim
    first[axis] = 0
    last[axis] = -1
    coefficients[tuple(first)] *= 0.5
    coefficients[tuple(last)] *= 0.5
    return coefficients


class IEncoder:
    """Base class for autoencoders operating on Chebyshev coefficients."""

    def __init__(
        self,
        sample_count: int,
        max_chebyshev_degree: int,
        range_start: float,
        range_end: float,
        device: str,
        means: Array,
        stds: Array,
        ckpt: dict,
        dropout1: float,
        dropout2: float,
    ) -> None:
        if max_chebyshev_degree < 0 or max_chebyshev_degree >= sample_count:
            raise ValueError(
                "max_chebyshev_degree must be between 0 and sample_count - 1"
            )

        self.sample_count = sample_count
        self.max_chebyshev_degree = max_chebyshev_degree
        self.range_start = range_start
        self.range_end = range_end
        self.device = device
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)

        if self.means.shape != self.stds.shape:
            raise ValueError("means and standard deviations must have the same shape")
        if np.any(self.stds == 0.0):
            raise ValueError("standard deviations must be nonzero")

        model_cfg = ckpt["model_cfg"]
        self.model = Autoencoder(
            input_dim=model_cfg.input_dim,
            hidden_1_dim=model_cfg.hidden_1_dim,
            hidden_2_dim=model_cfg.hidden_2_dim,
            latent_dim=model_cfg.latent_dim,
            hidden_activation=model_cfg.hidden_activation,
            latent_activation=model_cfg.latent_activation,
            output_activation=model_cfg.output_activation,
            dropout1=dropout1,
            dropout2=dropout2,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        if self.means.size != model_cfg.input_dim:
            raise ValueError(
                "standardization size does not match the autoencoder input dimension"
            )

    @property
    def chebyshev_nodes(self) -> Array:
        """Nodes at which the input function is sampled."""

        return chebyshev_gauss_lobatto_nodes(
            self.sample_count, self.range_start, self.range_end
        )

    def get_chebyshev_coefficients(self, function: Callable) -> Array:
        """Return the exact truncated discrete Chebyshev transform."""

        evaluations = self._get_evaluations(function)
        coefficients = np.asarray(
            self._get_chebyshev_coefficients(evaluations), dtype=np.float32
        )
        if coefficients.shape != self.means.shape:
            raise ValueError(
                "Chebyshev coefficient shape does not match the autoencoder input: "
                f"{coefficients.shape} != {self.means.shape}"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Chebyshev coefficients contain non-finite values")
        return coefficients

    def get_encoding(self, function: Callable) -> Array:
        """Encode a function's truncated Chebyshev coefficients."""

        return self._encode(self.get_chebyshev_coefficients(function))

    def get_reconstructed(self, function: Callable) -> Array:
        """Predict Chebyshev coefficients after an autoencoder round trip."""

        coefficients = self.get_chebyshev_coefficients(function)
        standardized = (coefficients - self.means) / self.stds
        with torch.no_grad():
            inputs = torch.as_tensor(
                standardized, dtype=torch.float32, device=self.device
            )
            reconstructed, _ = self.model(inputs)

        return reconstructed.cpu().numpy() * self.stds + self.means

    def _get_evaluations(self, function: Callable) -> Array:
        raise NotImplementedError

    def _get_chebyshev_coefficients(self, evaluations: Array) -> Array:
        raise NotImplementedError

    def _encode(self, coefficients: Array) -> Array:
        standardized = (coefficients - self.means) / self.stds
        with torch.no_grad():
            inputs = torch.as_tensor(
                standardized, dtype=torch.float32, device=self.device
            )
            _, encoding = self.model(inputs)

        return encoding.cpu().numpy()
