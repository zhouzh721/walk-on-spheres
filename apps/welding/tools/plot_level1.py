#!/usr/bin/env python3
"""Create the level-1 manufactured and stationary-welding validation figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "results" / "data"
FIGURE_DIR = APP_DIR / "results" / "figures"


def load_csv(name: str) -> np.ndarray:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))


def grid(field: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(field["x"])
    y = np.unique(field["y"])
    values = np.full((x.size, y.size), np.nan, dtype=float)
    values[np.searchsorted(x, field["x"]), np.searchsorted(y, field["y"])] = field[name]
    if np.isnan(values).any():
        raise ValueError(f"incomplete grid: {name}")
    return x, y, values


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def map_panel(axis: plt.Axes, x: np.ndarray, y: np.ndarray,
              values: np.ndarray, title: str, cmap: str,
              vmin: float | None = None, vmax: float | None = None):
    image = axis.pcolormesh(x, y, values.T, shading="auto", cmap=cmap,
                            vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("$x$ (m)")
    axis.set_ylabel("$y$ (m)")
    axis.set_aspect("equal")
    return image


def main() -> None:
    manufactured = load_csv("level1_manufactured_field.csv")
    gaussian = load_csv("level1_gaussian_field.csv")
    reference_data = load_csv("level1_gaussian_reference.csv")

    mx, my, manufactured_wos = grid(manufactured, "temperature_rise")
    _, _, manufactured_exact = grid(manufactured, "exact")
    _, _, manufactured_error = grid(manufactured, "error")
    _, _, manufactured_se = grid(manufactured, "standard_error")

    gx, gy, gaussian_wos = grid(gaussian, "temperature_rise")
    _, _, gaussian_se = grid(gaussian, "standard_error")
    _, _, gaussian_reference = grid(reference_data, "reference_temperature_rise")
    _, _, gaussian_error = grid(reference_data, "error")

    configure_style()
    figure = plt.figure(figsize=(11.2, 6.4), constrained_layout=True)
    layout = figure.add_gridspec(3, 4, height_ratios=(0.66, 0.66, 0.92))

    manufactured_limit = max(float(manufactured_wos.max()), float(manufactured_exact.max()))
    manufactured_error_limit = max(float(np.abs(manufactured_error).max()), 1e-12)
    top_specs = (
        (manufactured_wos, "1A WoS temperature rise", "inferno", 0.0, manufactured_limit),
        (manufactured_exact, "1A exact solution", "inferno", 0.0, manufactured_limit),
        (manufactured_error, "1A error: WoS − exact", "RdBu_r", -manufactured_error_limit, manufactured_error_limit),
        (manufactured_se, "1A standard error", "magma", 0.0, None),
    )
    for column, spec in enumerate(top_specs):
        axis = figure.add_subplot(layout[0, column])
        image = map_panel(axis, mx, my, *spec)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="K")

    gaussian_limit = max(float(gaussian_wos.max()), float(gaussian_reference.max()))
    gaussian_error_limit = max(float(np.abs(gaussian_error).max()), 1e-12)
    middle_specs = (
        (gaussian_wos, "1B WoS temperature rise", "inferno", 0.0, gaussian_limit),
        (gaussian_reference, "1B FD reference", "inferno", 0.0, gaussian_limit),
        (gaussian_error, "1B error: WoS − reference", "RdBu_r", -gaussian_error_limit, gaussian_error_limit),
        (gaussian_se, "1B standard error", "magma", 0.0, None),
    )
    for column, spec in enumerate(middle_specs):
        axis = figure.add_subplot(layout[1, column])
        image = map_panel(axis, gx, gy, *spec)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="K")

    center_i = gx.size // 2
    center_j = gy.size // 2
    x_axis = figure.add_subplot(layout[2, :2])
    x_axis.fill_between(
        gx,
        gaussian_wos[:, center_j] - 1.96 * gaussian_se[:, center_j],
        gaussian_wos[:, center_j] + 1.96 * gaussian_se[:, center_j],
        color="#4C78A8", alpha=0.18, linewidth=0,
        label="95% Monte Carlo interval",
    )
    x_axis.plot(gx, gaussian_wos[:, center_j], "o-", color="#2F5F8F",
                markersize=2.6, label="WoS")
    x_axis.plot(gx, gaussian_reference[:, center_j], "--", color="#E6863B",
                label="Finite-difference reference")
    x_axis.set_title("1B longitudinal centreline")
    x_axis.set_xlabel("$x$ (m)")
    x_axis.set_ylabel("Temperature rise (K)")
    x_axis.grid(True, color="#D9D9D9", linewidth=0.5)
    x_axis.legend()

    y_axis = figure.add_subplot(layout[2, 2:])
    y_axis.fill_between(
        gy,
        gaussian_wos[center_i, :] - 1.96 * gaussian_se[center_i, :],
        gaussian_wos[center_i, :] + 1.96 * gaussian_se[center_i, :],
        color="#4C78A8", alpha=0.18, linewidth=0,
        label="95% Monte Carlo interval",
    )
    y_axis.plot(gy, gaussian_wos[center_i, :], "o-", color="#2F5F8F",
                markersize=2.6, label="WoS")
    y_axis.plot(gy, gaussian_reference[center_i, :], "--", color="#E6863B",
                label="Finite-difference reference")
    y_axis.set_title("1B transverse centreline")
    y_axis.set_xlabel("$y$ (m)")
    y_axis.set_ylabel("Temperature rise (K)")
    y_axis.grid(True, color="#D9D9D9", linewidth=0.5)
    y_axis.legend()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_base = FIGURE_DIR / "level1_stationary_heat_validation"
    outputs = [
        (output_base.with_suffix(".png"), {"dpi": 300}),
        (output_base.with_suffix(".svg"), {}),
        (output_base.with_suffix(".pdf"), {}),
        (output_base.with_suffix(".tiff"), {"dpi": 600}),
    ]
    for path, options in outputs:
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
