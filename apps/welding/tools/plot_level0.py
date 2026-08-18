#!/usr/bin/env python3
"""Visualize the level-0 WoS cooling result against analytical solutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FIGURE_WIDTH_IN = 7.2047  # 183 mm, common double-column width
FIGURE_HEIGHT_IN = 5.1969  # 132 mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a wos_welding_level0 CSV result and its final field."
    )
    parser.add_argument("summary", type=Path, help="time-summary CSV")
    parser.add_argument(
        "--field",
        type=Path,
        help="final-field CSV; defaults to <summary stem>_field.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output basename without extension",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"result file not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return np.atleast_1d(data)


def field_grid(field: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(field["x"])
    y = np.unique(field["y"])
    values = np.full((x.size, y.size), np.nan, dtype=float)
    ix = np.searchsorted(x, field["x"])
    iy = np.searchsorted(y, field["y"])
    values[ix, iy] = field[name]
    if np.isnan(values).any():
        raise ValueError(f"field CSV does not contain a complete grid for {name}")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xx, yy, values


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.10,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    field_path = args.field
    if field_path is None:
        field_path = summary_path.with_name(
            f"{summary_path.stem}_field{summary_path.suffix}"
        )
    field_path = field_path.resolve()
    output_base = args.output
    if output_base is None:
        output_base = (
            Path(__file__).resolve().parent
            / "results"
            / "figures"
            / f"{summary_path.stem}_comparison"
        )
    output_base = output_base.resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)

    summary = load_csv(summary_path)
    field = load_csv(field_path)
    xx, yy, numerical = field_grid(field, "numerical")
    _, _, exact = field_grid(field, "exact_discrete")
    _, _, error = field_grid(field, "error")

    configure_style()
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        constrained_layout=True,
    )
    layout = figure.add_gridspec(2, 3, height_ratios=(1.15, 1.0))

    curve_axis = figure.add_subplot(layout[0, :])
    numerical_axis = figure.add_subplot(layout[1, 0])
    exact_axis = figure.add_subplot(layout[1, 1])
    error_axis = figure.add_subplot(layout[1, 2])

    time = summary["time"]
    center = summary["center_mean"]
    standard_error = summary["center_standard_error"]
    discrete = summary["center_exact_discrete"]
    continuous = summary["center_exact_continuous"]
    interval = 1.96 * standard_error

    curve_axis.fill_between(
        time,
        center - interval,
        center + interval,
        color="#4C78A8",
        alpha=0.18,
        linewidth=0,
        label="95% Monte Carlo interval",
    )
    curve_axis.plot(
        time,
        center,
        "o-",
        color="#2F5F8F",
        markersize=3.2,
        label="WoS implicit Euler",
        zorder=3,
    )
    curve_axis.plot(
        time,
        discrete,
        color="#E6863B",
        linestyle="--",
        label="Discrete analytical solution",
    )
    curve_axis.plot(
        time,
        continuous,
        color="#555555",
        linestyle=":",
        label="Continuous analytical solution",
    )
    curve_axis.set_xlabel("Time")
    curve_axis.set_ylabel("Normalised centre temperature excess, $\\theta$")
    curve_axis.set_xlim(time.min(), time.max())
    curve_axis.set_ylim(bottom=0.0)
    curve_axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    curve_axis.legend(ncol=2, loc="upper right")
    final_rmse = summary["field_rmse"][-1]
    final_max = summary["field_max_absolute_error"][-1]
    curve_axis.text(
        0.015,
        0.08,
        f"Final field RMSE = {final_rmse:.3g}\nMax. absolute error = {final_max:.3g}",
        transform=curve_axis.transAxes,
        ha="left",
        va="bottom",
        color="#333333",
    )
    add_panel_label(curve_axis, "a")

    temperature_max = float(max(np.max(numerical), np.max(exact)))
    temp_style = dict(cmap="viridis", shading="auto", vmin=0.0, vmax=temperature_max)
    numerical_map = numerical_axis.pcolormesh(xx, yy, numerical, **temp_style)
    exact_axis.pcolormesh(xx, yy, exact, **temp_style)

    error_limit = float(np.max(np.abs(error)))
    if error_limit == 0.0:
        error_limit = 1.0
    error_map = error_axis.pcolormesh(
        xx,
        yy,
        error,
        cmap="RdBu_r",
        shading="auto",
        vmin=-error_limit,
        vmax=error_limit,
    )

    for axis, title, label in (
        (numerical_axis, "WoS final field", "b"),
        (exact_axis, "Discrete analytical field", "c"),
        (error_axis, "Error: WoS − analytical", "d"),
    ):
        axis.set_title(title, pad=4)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_aspect("equal")
        axis.set_xlim(xx.min(), xx.max())
        axis.set_ylim(yy.min(), yy.max())
        add_panel_label(axis, label)

    temp_colorbar = figure.colorbar(
        numerical_map,
        ax=[numerical_axis, exact_axis],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    temp_colorbar.set_label("Normalised temperature excess, $\\theta$")
    error_colorbar = figure.colorbar(
        error_map,
        ax=error_axis,
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    error_colorbar.set_label("Signed error")

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    pdf_path = output_base.with_suffix(".pdf")
    tiff_path = output_base.with_suffix(".tiff")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    for path in (png_path, svg_path, pdf_path, tiff_path):
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
