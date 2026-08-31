"""Compare functions with exact and autoencoder-predicted Chebyshev expansions.

The exact coefficients and the coefficients passed through each autoencoder use
the same discrete Chebyshev transform: a correctly normalized DCT-I evaluated
at Chebyshev--Gauss--Lobatto nodes.  One-dimensional expansions retain degrees
0 through 15; two-dimensional expansions retain the tensor-product square
[0, 15] x [0, 15].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.encode.encoder1d import Encoder1D
from src.encode.encoder2d import Encoder2D


Array = np.ndarray
Function1D = Callable[[Array], Array]
Function2D = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class Config:
    """Plotting parameters."""

    n_plot_1d: int = 2_000
    n_plot_2d: int = 200
    output_dir: str = "./plots"
    figure_dpi: int = 200


@dataclass(frozen=True)
class Target1D:
    slug: str
    label: str
    function: Function1D


@dataclass(frozen=True)
class Target2D:
    slug: str
    label: str
    function: Function2D


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


def evaluate_chebyshev_1d(x: Array, coefficients: Array) -> Array:
    """Evaluate a Chebyshev expansion on [0, 1]."""

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1:
        raise ValueError("1D Chebyshev coefficients must be a flat array")
    values = np.polynomial.chebyshev.chebval(2.0 * x - 1.0, coefficients)
    return _real_finite(values, x.shape, "1D Chebyshev reconstruction")


def evaluate_chebyshev_2d(
    x: Array, y: Array, flat_coefficients: Array, max_degree: int
) -> Array:
    """Evaluate a row-major tensor-product Chebyshev expansion on [0, 1]^2."""

    coefficient_count = max_degree + 1
    flat_coefficients = np.asarray(flat_coefficients, dtype=float)
    expected_size = coefficient_count**2
    if flat_coefficients.shape != (expected_size,):
        raise ValueError(
            f"expected {expected_size} 2D Chebyshev coefficients, "
            f"got shape {flat_coefficients.shape}"
        )

    coefficients = flat_coefficients.reshape(
        coefficient_count, coefficient_count, order="C"
    )
    values = np.polynomial.chebyshev.chebval2d(
        2.0 * x - 1.0,
        2.0 * y - 1.0,
        coefficients,
    )
    return _real_finite(values, x.shape, "2D Chebyshev reconstruction")


def compute_error_metrics(exact: Array, approximation: Array) -> tuple[float, float]:
    """Return maximum absolute error and root-mean-square error."""

    error = np.abs(exact - approximation)
    return float(np.max(error)), float(np.sqrt(np.mean(np.square(error))))


def print_error_metrics(
    dimension: str,
    target_label: str,
    actual: Array,
    exact_transform: Array,
    autoencoder_prediction: Array,
) -> None:
    """Print reconstruction errors for the two Chebyshev approximations."""

    print(f"\n{dimension} target: {target_label}")
    for label, values in (
        ("Exact Chebyshev transform", exact_transform),
        ("Autoencoder-predicted transform", autoencoder_prediction),
    ):
        max_error, rms_error = compute_error_metrics(actual, values)
        print(f"  {label:<35} max={max_error:.6e}  rms={rms_error:.6e}")


def plot_1d_comparison(
    target_label: str,
    x: Array,
    actual: Array,
    exact_transform: Array,
    autoencoder_prediction: Array,
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Plot only the actual function and the two Chebyshev reconstructions."""

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, actual, label="Actual function", color="#ff8945", linewidth=1.8)
    axis.plot(
        x,
        exact_transform,
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
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_title(f"1D Chebyshev autoencoder comparison: {target_label}")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.6)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=figure_dpi)
    plt.close(figure)


def _shared_limits(values: tuple[Array, ...]) -> tuple[float, float]:
    """Return non-degenerate color limits shared by all function panels."""

    lower = min(float(np.min(value)) for value in values)
    upper = max(float(np.max(value)) for value in values)
    if lower == upper:
        delta = np.finfo(float).eps * max(1.0, abs(lower))
        return lower - delta, upper + delta
    return lower, upper


def plot_2d_comparison(
    target_label: str,
    x: Array,
    y: Array,
    actual: Array,
    exact_transform: Array,
    autoencoder_prediction: Array,
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Plot only the actual surface and the two Chebyshev reconstructions."""

    colormap = LinearSegmentedColormap.from_list(
        "orange_teal",
        [(0.0, "#006064"), (0.3, "#6db7bb"), (0.6, "#ff8945"), (1.0, "#882000")],
    )
    panels = (
        ("Actual function", actual),
        ("Exact Chebyshev transform", exact_transform),
        ("Autoencoder prediction", autoencoder_prediction),
    )
    value_min, value_max = _shared_limits(tuple(values for _, values in panels))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, (title, values) in zip(axes, panels):
        image = axis.pcolormesh(
            x,
            y,
            values,
            shading="auto",
            cmap=colormap,
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis)

    figure.suptitle(f"2D Chebyshev autoencoder comparison: {target_label}")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=figure_dpi)
    plt.close(figure)


def get_targets_1d() -> tuple[Target1D, ...]:
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


def get_targets_2d() -> tuple[Target2D, ...]:
    """Return the two-dimensional functions to compare."""

    return (
        Target2D("zero", "f(x, y) = 0", lambda x, y: np.zeros_like(x)),
        Target2D("x_plus_y", "f(x, y) = x + y", lambda x, y: x + y),
        Target2D("x_times_y", "f(x, y) = x y", lambda x, y: x * y),
        Target2D(
            "exp_minus_x_minus_y",
            "f(x, y) = exp(-x - y)",
            lambda x, y: np.exp(-x - y),
        ),
        Target2D(
            "sin_x_plus_y",
            "f(x, y) = sin(x + y)",
            lambda x, y: np.sin(x + y),
        ),
        Target2D(
            "exp_sin_pi_xy",
            "f(x, y) = exp(sin(pi x y))",
            lambda x, y: np.exp(np.sin(np.pi * x * y)),
        ),
        Target2D(
            "four_x2_y",
            "f(x, y) = 4 x^2 y",
            lambda x, y: 4.0 * x**2 * y,
        ),
        Target2D(
            "one_over_1_plus_x2_plus_y2",
            "f(x, y) = 1 / (1 + (x - 0.5)^2 + (y - 0.5)^2)",
            lambda x, y: 1.0 / (1.0 + (x - 0.5) ** 2 + (y - 0.5) ** 2),
        ),
    )


def main() -> None:
    """Generate Chebyshev-transform and autoencoder comparison plots."""

    config = Config()
    output_directory = Path(config.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # encoder_1d = Encoder1D()
    encoder_2d = Encoder2D()
    generated_files: list[Path] = []

    # x_plot = np.linspace(0.0, 1.0, config.n_plot_1d, endpoint=True)
    # for index, target in enumerate(get_targets_1d(), start=1):
    #     actual = _real_finite(target.function(x_plot), x_plot.shape, target.label)
    #     exact_coefficients = encoder_1d.get_chebyshev_coefficients(target.function)
    #     predicted_coefficients = encoder_1d.get_reconstructed(target.function)
    #     exact_transform = evaluate_chebyshev_1d(x_plot, exact_coefficients)
    #     autoencoder_prediction = evaluate_chebyshev_1d(
    #         x_plot, predicted_coefficients
    #     )

    #     print_error_metrics(
    #         "1D",
    #         target.label,
    #         actual,
    #         exact_transform,
    #         autoencoder_prediction,
    #     )
    #     output_path = output_directory / f"plot_1d_{index:02d}_{target.slug}.png"
    #     plot_1d_comparison(
    #         target.label,
    #         x_plot,
    #         actual,
    #         exact_transform,
    #         autoencoder_prediction,
    #         output_path,
    #         config.figure_dpi,
    #     )
    #     generated_files.append(output_path)

    grid = np.linspace(0.0, 1.0, config.n_plot_2d, endpoint=True)
    x_mesh, y_mesh = np.meshgrid(grid, grid, indexing="ij")
    for index, target in enumerate(get_targets_2d(), start=1):
        actual = _real_finite(
            target.function(x_mesh, y_mesh), x_mesh.shape, target.label
        )
        exact_coefficients = encoder_2d.get_chebyshev_coefficients(target.function)
        predicted_coefficients = encoder_2d.get_reconstructed(target.function)
        exact_transform = evaluate_chebyshev_2d(
            x_mesh,
            y_mesh,
            exact_coefficients,
            encoder_2d.max_chebyshev_degree,
        )
        autoencoder_prediction = evaluate_chebyshev_2d(
            x_mesh,
            y_mesh,
            predicted_coefficients,
            encoder_2d.max_chebyshev_degree,
        )

        print_error_metrics(
            "2D",
            target.label,
            actual,
            exact_transform,
            autoencoder_prediction,
        )
        output_path = output_directory / f"plot_2d_{index:02d}_{target.slug}.png"
        plot_2d_comparison(
            target.label,
            x_mesh,
            y_mesh,
            actual,
            exact_transform,
            autoencoder_prediction,
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
