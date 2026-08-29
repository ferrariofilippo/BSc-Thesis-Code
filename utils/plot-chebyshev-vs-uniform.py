"""Plot Chebyshev--Lobatto and equidistant nodes on the interval [0, 1]."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHEBYSHEV_COLOR = "#006064"
EQUIDISTANT_COLOR = "#ff8945"
SEGMENT_COLOR = "#324b4c"


def chebyshev_lobatto_nodes(node_count: int) -> np.ndarray:
    """Return ascending Chebyshev--Lobatto nodes mapped to [0, 1]."""

    indices = np.arange(node_count)
    return 0.5 * (1.0 - np.cos(np.pi * indices / (node_count - 1)))


def create_plot(output_directory: Path) -> tuple[Path, Path]:
    """Create the comparison plot and save PDF and PNG copies."""

    chebyshev_nodes = chebyshev_lobatto_nodes(10)
    equidistant_nodes = np.linspace(0.0, 1.0, 10)

    figure, axis = plt.subplots(figsize=(9.0, 3.2))
    chebyshev_y = 0.18
    equidistant_y = -0.18

    # The common interval is centered between the two node distributions.
    axis.plot(
        [0.0, 1.0],
        [0.0, 0.0],
        color=SEGMENT_COLOR,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=1,
    )

    axis.vlines(
        chebyshev_nodes,
        0.0,
        chebyshev_y,
        color=CHEBYSHEV_COLOR,
        linewidth=1.0,
        alpha=0.32,
        zorder=2,
    )
    axis.scatter(
        chebyshev_nodes,
        np.full_like(chebyshev_nodes, chebyshev_y),
        s=64,
        marker="o",
        color=CHEBYSHEV_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Chebyshev–Lobatto (10 nodes)",
        zorder=3,
    )

    axis.vlines(
        equidistant_nodes,
        equidistant_y,
        0.0,
        color=EQUIDISTANT_COLOR,
        linewidth=1.0,
        alpha=0.38,
        zorder=2,
    )
    axis.scatter(
        equidistant_nodes,
        np.full_like(equidistant_nodes, equidistant_y),
        s=66,
        marker="D",
        color=EQUIDISTANT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Equidistant (10 nodes)",
        zorder=3,
    )

    # Leave visual breathing room around the closed interval.
    axis.set_xlim(-0.075, 1.075)
    axis.set_ylim(-0.36, 0.36)
    axis.set_xticks(np.linspace(0.0, 1.0, 5))
    axis.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axis.set_yticks([])
    # axis.set_xlabel(r"$x \in [0,1]$", labelpad=8)
    axis.tick_params(axis="x", colors=SEGMENT_COLOR, length=0, pad=2)

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    figure.tight_layout(pad=1.2)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_path = output_directory / "chebyshev_lobatto_vs_equidistant.pdf"
    png_path = output_directory / "chebyshev_lobatto_vs_equidistant.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    """Generate the node-distribution comparison figure."""

    output_directory = Path(__file__).resolve().parent / "plots"
    generated_files = create_plot(output_directory)
    for path in generated_files:
        print(f"Created {path}")


if __name__ == "__main__":
    main()
