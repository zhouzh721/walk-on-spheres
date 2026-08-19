#!/usr/bin/env python3
"""Animate the complete Level-2 WoS field against a refined FD reference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from validate_level2 import (
    DirichletSineSolver,
    gaussian_integral,
    heating_fraction,
    load_metadata,
    numeric,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = APP_DIR / "results" / "data" / "level2_cubic_baseline"


class TrueColorWebPWriter(animation.PillowWriter):
    """Export lossless RGB WebP while releasing RGBA frames progressively."""

    def finish(self) -> None:
        rgb_frames: list[Image.Image] = []
        for index, frame in enumerate(self._frames):
            rgb_frames.append(frame.convert("RGB"))
            self._frames[index] = None
        self._frames.clear()
        duration_ms = int(round(1000.0 / self.fps))
        rgb_frames[0].save(
            self.outfile,
            save_all=True,
            append_images=rgb_frames[1:],
            duration=duration_ms,
            loop=0,
            lossless=True,
            quality=100,
            method=6,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Level-2 output prefix without CSV suffixes",
    )
    parser.add_argument(
        "--reference-refinement",
        type=int,
        default=4,
        help="FD interval refinement (default: 4)",
    )
    parser.add_argument(
        "--reference-substeps",
        type=int,
        default=16,
        help="FD substeps per WoS step (default: 16)",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("-o", "--output", type=Path)
    arguments = parser.parse_args()
    if arguments.reference_refinement <= 0:
        parser.error("--reference-refinement must be positive")
    if arguments.reference_substeps <= 0:
        parser.error("--reference-substeps must be positive")
    if not math.isfinite(arguments.fps) or arguments.fps <= 0.0:
        parser.error("--fps must be a positive finite number")
    if arguments.dpi <= 0:
        parser.error("--dpi must be positive")
    return arguments


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
        }
    )


def load_numeric_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    )


def load_wos_fields(
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = np.unique(history["step"]).astype(int)
    x = np.unique(history["x"])
    y = np.unique(history["y"])
    expected_rows = steps.size * x.size * y.size
    if history.size != expected_rows:
        raise ValueError(
            f"history has {history.size} rows; expected {expected_rows}"
        )
    times = np.empty(steps.size, dtype=float)
    fields = np.full((steps.size, y.size, x.size), np.nan, dtype=float)
    step_index = {step: index for index, step in enumerate(steps)}
    for row in history:
        step = int(round(float(row["step"])))
        frame = step_index[step]
        times[frame] = float(row["time"])
        ix = int(np.searchsorted(x, row["x"]))
        iy = int(np.searchsorted(y, row["y"]))
        fields[frame, iy, ix] = row["temperature_rise"]
    if np.isnan(fields).any():
        raise ValueError("history CSV does not contain complete temperature fields")
    return steps, times, x, y, fields


def refined_reference_fields(
    metadata: dict[str, str],
    x: np.ndarray,
    y: np.ndarray,
    step_count: int,
    refinement: int,
    substeps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = numeric(metadata, "dt")
    heat_on_time = numeric(metadata, "heat_on_time")
    thickness = numeric(metadata, "thickness")
    density = numeric(metadata, "density")
    specific_heat = numeric(metadata, "specific_heat")
    conductivity = numeric(metadata, "conductivity")
    absorbed_power = numeric(metadata, "absorbed_power")
    sigma_x = numeric(metadata, "sigma_x")
    sigma_y = numeric(metadata, "sigma_y")
    diffusivity = conductivity / (density * specific_heat)

    reference_nx = (x.size - 1) * refinement + 1
    reference_ny = (y.size - 1) * refinement + 1
    reference_x = np.linspace(x[0], x[-1], reference_nx)
    reference_y = np.linspace(y[0], y[-1], reference_ny)
    reference_dt = dt / substeps
    solver = DirichletSineSolver(
        reference_nx,
        reference_ny,
        reference_x[1] - reference_x[0],
        reference_y[1] - reference_y[0],
        diffusivity * reference_dt,
    )

    xx, yy = np.meshgrid(
        reference_x[1:-1], reference_y[1:-1], indexing="ij"
    )
    integral_x = gaussian_integral(x[0], x[-1], 0.0, sigma_x)
    integral_y = gaussian_integral(y[0], y[-1], 0.0, sigma_y)
    source = absorbed_power / (thickness * integral_x * integral_y) * np.exp(
        -0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)
    )
    source_rate_coefficients = solver.forward(source / (density * specific_heat))
    coefficients = np.zeros(
        (reference_nx - 2, reference_ny - 2), dtype=float
    )
    query_fields = np.zeros((step_count + 1, y.size, x.size), dtype=float)
    display_fields = np.zeros(
        (step_count + 1, reference_y.size, reference_x.size), dtype=float
    )

    for step in range(1, step_count + 1):
        outer_begin = (step - 1) * dt
        for substep in range(substeps):
            time_begin = outer_begin + substep * reference_dt
            fraction = heating_fraction(
                time_begin, time_begin + reference_dt, heat_on_time
            )
            coefficients = (
                coefficients + reference_dt * fraction * source_rate_coefficients
            ) / solver.denominator

        refined = np.zeros((reference_nx, reference_ny), dtype=float)
        refined[1:-1, 1:-1] = solver.inverse(coefficients)
        # Retain the complete refined field for display.  Separately extract
        # the embedded WoS nodes for a like-for-like pointwise error field.
        display_fields[step] = refined.T
        query_fields[step] = refined[::refinement, ::refinement].T
    return query_fields, display_fields, reference_x, reference_y


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


def rounded_limit(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    return math.ceil(value / magnitude * 2.0) * magnitude / 2.0


def main() -> None:
    args = parse_args()
    prefix = args.prefix.resolve()
    summary = load_numeric_csv(Path(f"{prefix}_summary.csv"))
    history = load_numeric_csv(Path(f"{prefix}_history.csv"))
    metadata = load_metadata(Path(f"{prefix}_metadata.csv"))
    steps, times, x, y, wos_fields = load_wos_fields(history)

    dt = numeric(metadata, "dt")
    total_time = numeric(metadata, "total_time")
    heat_on_time = numeric(metadata, "heat_on_time")
    step_count = int(round(total_time / dt))
    expected_steps = np.arange(step_count + 1)
    if not np.array_equal(steps, expected_steps):
        raise ValueError(
            "animation requires every Level-2 field; rerun with --field-stride 1"
        )
    if summary.size != steps.size or not np.allclose(summary["time"], times):
        raise ValueError("summary and history time axes do not match")

    (
        reference_fields,
        reference_display_fields,
        reference_x,
        reference_y,
    ) = refined_reference_fields(
        metadata,
        x,
        y,
        step_count,
        args.reference_refinement,
        args.reference_substeps,
    )
    errors = wos_fields - reference_fields
    center_i = x.size // 2
    center_j = y.size // 2
    reference_center = reference_fields[:, center_j, center_i]
    interior_errors = errors[:, 1:-1, 1:-1]
    field_rmse = np.sqrt(np.mean(interior_errors**2, axis=(1, 2)))
    reference_peak = np.max(reference_fields, axis=(1, 2))
    normalized_rmse = np.divide(
        field_rmse,
        reference_peak,
        out=np.zeros_like(field_rmse),
        where=reference_peak > 0.0,
    )

    output_path = args.output
    if output_path is None:
        output_path = (
            APP_DIR
            / "results"
            / "animations"
            / f"{prefix.name}_comparison_truecolor.webp"
        )
    output_path = output_path.resolve().with_suffix(".webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    figure = plt.figure(
        figsize=(7.2047, 5.1969),  # 183 x 132 mm
        constrained_layout=True,
    )
    layout = figure.add_gridspec(2, 3, height_ratios=(0.95, 1.0))
    curve_axis = figure.add_subplot(layout[0, :])
    wos_axis = figure.add_subplot(layout[1, 0])
    reference_axis = figure.add_subplot(layout[1, 1])
    error_axis = figure.add_subplot(layout[1, 2])

    center_wos = summary["center_temperature_rise"]
    center_se = summary["center_standard_error"]
    curve_axis.axvspan(
        0.0, heat_on_time, color="#E6863B", alpha=0.08, linewidth=0
    )
    curve_axis.axvline(
        heat_on_time, color="#8C5A2B", linewidth=0.9, linestyle=":"
    )
    curve_axis.fill_between(
        times,
        np.maximum(0.0, center_wos - 3.0 * center_se),
        center_wos + 3.0 * center_se,
        color="#2F5F8F",
        alpha=0.14,
        linewidth=0,
        label="WoS ±3 conditional SE",
    )
    curve_axis.plot(
        times,
        center_wos,
        color="#2F5F8F",
        linewidth=1.4,
        label="WoS implicit Euler",
    )
    curve_axis.plot(
        times,
        reference_center,
        color="#E6863B",
        linewidth=1.3,
        linestyle="--",
        label=(
            f"FD reference ({args.reference_refinement}× space, "
            f"{args.reference_substeps}× time)"
        ),
    )
    cursor = curve_axis.axvline(times[0], color="#B23A48", linewidth=1.0)
    wos_marker, = curve_axis.plot(
        [times[0]], [center_wos[0]], "o", color="#B23A48", markersize=4
    )
    reference_marker, = curve_axis.plot(
        [times[0]],
        [reference_center[0]],
        "s",
        color="#E6863B",
        markersize=3.5,
    )
    curve_axis.text(
        0.5 * heat_on_time,
        0.95,
        "Arc on",
        transform=curve_axis.get_xaxis_transform(),
        color="#8C5A2B",
        ha="center",
        va="top",
        fontweight="bold",
    )
    curve_axis.text(
        0.5 * (heat_on_time + total_time),
        0.95,
        "Cooling",
        transform=curve_axis.get_xaxis_transform(),
        color="#555555",
        ha="center",
        va="top",
        fontweight="bold",
    )
    curve_axis.set_xlabel("Time (s)")
    curve_axis.set_ylabel("Centre temperature rise (K)")
    curve_axis.set_xlim(0.0, total_time)
    curve_axis.set_ylim(
        0.0, 1.08 * max(float(np.max(center_wos)), float(np.max(reference_center)))
    )
    curve_axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    curve_axis.legend(ncol=3, loc="lower center")
    add_panel_label(curve_axis, "a")

    x_mm = 1_000.0 * x
    y_mm = 1_000.0 * y
    extent = (
        float(x_mm.min()),
        float(x_mm.max()),
        float(y_mm.min()),
        float(y_mm.max()),
    )
    temperature_limit = rounded_limit(
        max(float(np.max(wos_fields)), float(np.max(reference_display_fields)))
    )
    error_limit = rounded_limit(float(np.max(np.abs(errors))))
    temperature_norm = mpl.colors.Normalize(
        vmin=0.0, vmax=temperature_limit, clip=True
    )
    error_norm = mpl.colors.TwoSlopeNorm(
        vmin=-error_limit, vcenter=0.0, vmax=error_limit
    )
    wos_image = wos_axis.imshow(
        wos_fields[0],
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
        norm=temperature_norm,
        aspect="equal",
    )
    reference_image = reference_axis.imshow(
        reference_display_fields[0],
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
        norm=temperature_norm,
        aspect="equal",
    )
    error_image = error_axis.imshow(
        errors[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        interpolation="nearest",
        norm=error_norm,
        aspect="equal",
    )
    for axis, title, label in (
        (wos_axis, f"WoS field ({x.size} × {y.size})", "b"),
        (
            reference_axis,
            f"FD reference ({reference_x.size} × {reference_y.size})",
            "c",
        ),
        (
            error_axis,
            f"Node error ({x.size} × {y.size})",
            "d",
        ),
    ):
        axis.set_title(title, pad=4)
        axis.set_xlabel("$x$ (mm)")
        axis.set_ylabel("$y$ (mm)")
        add_panel_label(axis, label)

    temperature_colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=temperature_norm, cmap="inferno"),
        ax=[wos_axis, reference_axis],
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        ticks=np.linspace(0.0, temperature_limit, 5),
    )
    temperature_colorbar.set_label(
        "Temperature rise (K; fixed range for all frames)"
    )
    error_colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap="RdBu_r"),
        ax=error_axis,
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        ticks=np.linspace(-error_limit, error_limit, 5),
    )
    error_colorbar.set_label("Signed error (K; fixed range for all frames)")
    time_label = figure.suptitle("", fontsize=9, fontweight="bold")

    def update(frame: int):
        wos_image.set_data(wos_fields[frame])
        reference_image.set_data(reference_display_fields[frame])
        error_image.set_data(errors[frame])
        cursor.set_xdata([times[frame], times[frame]])
        wos_marker.set_data([times[frame]], [center_wos[frame]])
        reference_marker.set_data([times[frame]], [reference_center[frame]])
        if math.isclose(times[frame], heat_on_time, abs_tol=0.5 * dt):
            phase = "arc shutoff"
        elif times[frame] < heat_on_time:
            phase = "heating"
        else:
            phase = "cooling"
        time_label.set_text(
            f"Level 2 {phase}: step {steps[frame]}, t = {times[frame]:.2f} s"
            f"  |  field RMSE = {field_rmse[frame]:.3f} K"
            f" ({100.0 * normalized_rmse[frame]:.3f}%)"
        )
        return (
            wos_image,
            reference_image,
            error_image,
            cursor,
            wos_marker,
            reference_marker,
            time_label,
        )

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=np.arange(steps.size),
        interval=1_000.0 / args.fps,
        blit=False,
        repeat=True,
    )

    preview_dir = APP_DIR / "results" / "figures"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_base = preview_dir / f"{prefix.name}_comparison"
    for target_time, suffix in (
        (heat_on_time, "arc_off"),
        (total_time, "final"),
    ):
        frame = int(np.argmin(np.abs(times - target_time)))
        update(frame)
        figure.savefig(
            preview_base.with_name(f"{preview_base.name}_{suffix}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    update(steps.size - 1)
    figure.savefig(
        preview_base.with_name(f"{preview_base.name}_final.svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        preview_base.with_name(f"{preview_base.name}_final.pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        preview_base.with_name(f"{preview_base.name}_final.tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    update(0)
    movie.save(
        output_path,
        writer=TrueColorWebPWriter(fps=args.fps),
        dpi=args.dpi,
        progress_callback=lambda frame, total: print(
            f"rendering frame {frame + 1}/{total}", end="\r", flush=True
        ),
    )
    print(f"\n{output_path}")
    print(
        f"fixed temperature range: 0 to {temperature_limit:g} K; "
        f"fixed error range: {-error_limit:g} to {error_limit:g} K"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
