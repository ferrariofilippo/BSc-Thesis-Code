"""Compare real 1D functions with Chebyshev and Fourier reconstructions.

For each target, the script plots the original function, its degree-15
Chebyshev transform, the autoencoder-predicted Chebyshev expansion, and a
truncated real Fourier expansion.  The Fourier expansion is computed from 128
uniform samples and retains order 12: one DC coefficient plus 12 cosine and 12
sine coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.encode.encoder1d import Encoder1D


Array = np.ndarray
Function1D = Callable[[Array], Array]


@dataclass(frozen=True)
class Config:
    """Plotting and Fourier-transform parameters."""

    n_plot: int = 2_000
    fft_sample_count: int = 128
    fft_order: int = 12
    output_dir: str = "./plots/1d"
    figure_dpi: int = 200


@dataclass(frozen=True)
class Target1D:
    slug: str
    label: str
    function: Function1D


@dataclass(frozen=True)
class RealFourierCoefficients:
    """Coefficients for dc + sum(a_k cos(2 pi k x) + b_k sin(2 pi k x))."""

    dc: float
    cosines: Array
    sines: Array

    def __post_init__(self) -> None:
        if self.cosines.ndim != 1 or self.sines.ndim != 1:
            raise ValueError("Fourier cosine and sine coefficients must be flat")
        if self.cosines.shape != self.sines.shape:
            raise ValueError("Fourier cosine and sine arrays must have equal length")

    @property
    def order(self) -> int:
        return int(self.cosines.size)

    @property
    def count(self) -> int:
        return 1 + 2 * self.order


def _real_finite(values: Array, shape: tuple[int, ...], context: str) -> Array:
    """Return a finite real array with the expected shape."""

    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{context}: expected shape {shape}, got {array.shape}")
    if np.iscomplexobj(array):
        max_imaginary = float(np.max(np.abs(array.imag)))
        if max_imaginary > 1e-13:
            raise ValueError(f"{context}: imaginary residual {max_imaginary:.3e}")
        array = array.real

    result = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context}: non-finite values")
    return result


def evaluate_chebyshev(x: Array, coefficients: Array) -> Array:
    """Evaluate a Chebyshev expansion on [0, 1]."""

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1:
        raise ValueError("Chebyshev coefficients must be a flat array")
    values = np.polynomial.chebyshev.chebval(2.0 * x - 1.0, coefficients)
    return _real_finite(values, x.shape, "Chebyshev reconstruction")


def get_real_fourier_coefficients(
    function: Function1D,
    sample_count: int,
    order: int,
) -> RealFourierCoefficients:
    """Return a real order-``order`` Fourier expansion from uniform samples.

    NumPy's FFT produces coefficients for both positive and negative
    frequencies.  For a real input these coefficients are conjugate pairs.
    Each +k/-k pair is combined into one real cosine coefficient and one real
    sine coefficient, so an order-12 expansion contains exactly 25 real
    coefficients.
    """

    if sample_count <= 0:
        raise ValueError("FFT sample count must be positive")
    if order < 0 or order >= (sample_count + 1) // 2:
        raise ValueError(
            "FFT order must be non-negative and below the Nyquist frequency"
        )

    sample_x = np.arange(sample_count, dtype=float) / sample_count
    samples = _real_finite(
        function(sample_x), sample_x.shape, "uniform FFT samples"
    )
    spectrum = np.fft.fft(samples) / sample_count

    if order == 0:
        empty = np.empty(0, dtype=float)
        return RealFourierCoefficients(float(spectrum[0].real), empty, empty)

    positive = spectrum[1 : order + 1]
    negative = spectrum[-1 : -order - 1 : -1]
    conjugacy_error = float(np.max(np.abs(negative - np.conj(positive))))
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(spectrum))))
    if conjugacy_error > tolerance:
        raise ValueError(
            "real FFT samples did not produce conjugate frequency pairs: "
            f"maximum mismatch {conjugacy_error:.3e}"
        )

    # c_k exp(i theta) + c_-k exp(-i theta)
    # = (c_k + c_-k) cos(theta) + i(c_k - c_-k) sin(theta).
    cosines = np.asarray((positive + negative).real, dtype=float)
    sines = np.asarray((1j * (positive - negative)).real, dtype=float)
    return RealFourierCoefficients(
        dc=float(spectrum[0].real),
        cosines=cosines,
        sines=sines,
    )


def evaluate_real_fourier(x: Array, coefficients: RealFourierCoefficients) -> Array:
    """Evaluate a real Fourier expansion of period one."""

    result = np.full(x.shape, coefficients.dc, dtype=float)
    if coefficients.order:
        harmonics = np.arange(1, coefficients.order + 1, dtype=float)
        phases = 2.0 * np.pi * np.multiply.outer(x, harmonics)
        result += np.cos(phases) @ coefficients.cosines
        result += np.sin(phases) @ coefficients.sines
    return _real_finite(result, x.shape, "Fourier reconstruction")


def compute_error_metrics(exact: Array, approximation: Array) -> tuple[float, float]:
    """Return maximum absolute error and root-mean-square error."""

    error = np.abs(exact - approximation)
    return float(np.max(error)), float(np.sqrt(np.mean(np.square(error))))


def print_results(
    target: Target1D,
    actual: Array,
    exact_chebyshev: Array,
    autoencoder_prediction: Array,
    fourier: Array,
    fourier_coefficients: RealFourierCoefficients,
) -> None:
    """Print reconstruction errors and the retained real FFT coefficients."""

    print(f"\n1D target: {target.label}")
    for label, values in (
        ("Exact Chebyshev transform", exact_chebyshev),
        ("Autoencoder prediction", autoencoder_prediction),
        (f"Order-{fourier_coefficients.order} Fourier transform", fourier),
    ):
        max_error, rms_error = compute_error_metrics(actual, values)
        print(f"  {label:<35} max={max_error:.6e}  rms={rms_error:.6e}")

    print(f"  Fourier coefficients ({fourier_coefficients.count} real values):")
    print(f"    DC       {fourier_coefficients.dc:+.9e}")
    for harmonic, (cosine, sine) in enumerate(
        zip(fourier_coefficients.cosines, fourier_coefficients.sines), start=1
    ):
        print(f"    k={harmonic:<2} cos={cosine:+.9e}  sin={sine:+.9e}")


def plot_comparison(
    target: Target1D,
    x: Array,
    actual: Array,
    exact_chebyshev: Array,
    autoencoder_prediction: Array,
    fourier: Array,
    fourier_order: int,
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Plot the function and all three approximations."""

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, actual, label="Actual function", color="#ff8945", linewidth=1.8)
    axis.plot(
        x,
        exact_chebyshev,
        label="Exact Chebyshev transform",
        color="#324b4c",
        linestyle="--",
        linewidth=1.5,
    )
    axis.plot(
        x,
        autoencoder_prediction,
        label="Autoencoder prediction",
        color="#006064",
        linewidth=1.5,
    )
    axis.plot(
        x,
        fourier,
        label=f"Exact FFT (order {fourier_order})",
        color="#7e57c2",
        linestyle=":",
        linewidth=1.8,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_title(f"1D transform comparison: {target.label}")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.6)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=figure_dpi)
    plt.close(figure)


def get_targets() -> tuple[Target1D, ...]:
    """Return the one-dimensional functions to compare."""

    return (
        Target1D("zero", "f(x) = 0", lambda x: np.zeros_like(x)),
        Target1D("linear", "f(x) = x", lambda x: x),
        Target1D("sin_pi_x", "f(x) = sin(pi x)", lambda x: np.sin(np.pi * x)),
        Target1D("exp_minus_x", "f(x) = exp(-x)", lambda x: np.exp(-x)),
        Target1D(
            "exp_cos_2pi_x",
            "f(x) = exp(cos(2 pi x))",
            lambda x: np.exp(np.cos(2.0 * np.pi * x)),
        ),
        Target1D("tanh_x", "f(x) = tanh(x)", lambda x: np.tanh(x)),
        Target1D("exp_x", "f(x) = exp(x)", lambda x: np.exp(x)),
        Target1D("log_1_plus_x", "f(x) = log(1 + x)", lambda x: np.log1p(x)),
        Target1D(
            "one_over_1_plus_x2",
            "f(x) = 1 / (1 + (x - 0.5)^2)",
            lambda x: 1.0 / (1.0 + (x - 0.5) ** 2),
        ),
    )


def main() -> None:
    """Generate 1D Chebyshev, autoencoder, and Fourier comparison plots."""

    config = Config()
    expected_coefficient_count = 1 + 2 * config.fft_order
    if expected_coefficient_count != 25:
        raise ValueError(
            "this comparison is configured for exactly 25 Fourier coefficients"
        )

    output_directory = Path(config.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    encoder = Encoder1D()
    if encoder.sample_count != config.fft_sample_count:
        raise ValueError(
            "the encoder and FFT must both use 128 samples for this comparison"
        )

    generated_files: list[Path] = []
    x_plot = np.linspace(0.0, 1.0, config.n_plot, endpoint=True)
    for index, target in enumerate(get_targets(), start=1):
        actual = _real_finite(target.function(x_plot), x_plot.shape, target.label)
        exact_coefficients = encoder.get_chebyshev_coefficients(target.function)
        predicted_coefficients = encoder.get_reconstructed(target.function)
        exact_chebyshev = evaluate_chebyshev(x_plot, exact_coefficients)
        autoencoder_prediction = evaluate_chebyshev(
            x_plot, predicted_coefficients
        )

        fourier_coefficients = get_real_fourier_coefficients(
            target.function,
            sample_count=config.fft_sample_count,
            order=config.fft_order,
        )
        fourier = evaluate_real_fourier(x_plot, fourier_coefficients)

        print_results(
            target,
            actual,
            exact_chebyshev,
            autoencoder_prediction,
            fourier,
            fourier_coefficients,
        )
        output_path = output_directory / f"plot_1d_{index:02d}_{target.slug}.png"
        plot_comparison(
            target,
            x_plot,
            actual,
            exact_chebyshev,
            autoencoder_prediction,
            fourier,
            config.fft_order,
            output_path,
            config.figure_dpi,
        )
        generated_files.append(output_path)

    missing_files = [path for path in generated_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"expected plot files were not created: {missing_files}")

    print(f"\nGenerated {len(generated_files)} plots in {output_directory.resolve()}:")
    for path in generated_files:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
