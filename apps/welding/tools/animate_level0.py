#!/usr/bin/env python3
"""Animate every computed level-0 cooling field against the analytical field."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FIGURE_WIDTH_IN = 7.2047  # 183 mm
FIGURE_HEIGHT_IN = 5.1969  # 132 mm


class FixedPaletteGifWriter(animation.PillowWriter):
    """Write every GIF frame with one palette and also export lossless WebP."""

    def __init__(self, *, fps: float, webp_path: Path) -> None:
        super().__init__(fps=fps)
        self.webp_path = webp_path

    def finish(self) -> None:
        rgb_frames = [frame.convert("RGB") for frame in self._frames]
        duration_ms = int(round(1000.0 / self.fps))

        # Lossless WebP retains the rendered RGB colours without GIF's
        # 256-colour limit or frame-dependent quantisation.
        self.webp_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_frames[0].save(
            self.webp_path,
            save_all=True,
            append_images=rgb_frames[1:],
            duration=duration_ms,
            loop=0,
            lossless=True,
            quality=100,
            method=6,
        )

        # Build one master palette from all frames.  Quantising every frame
        # against it prevents frame-dependent GIF palette changes.
        thumb_width = min(720, rgb_frames[0].width)
        thumb_height = max(
            1, round(rgb_frames[0].height * thumb_width / rgb_frames[0].width)
        )
        atlas = Image.new("RGB", (thumb_width, thumb_height * len(rgb_frames)))
        for index, frame in enumerate(rgb_frames):
            thumbnail = frame.resize(
                (thumb_width, thumb_height), resample=Image.Resampling.LANCZOS
            )
            atlas.paste(thumbnail, (0, index * thumb_height))
        master_palette = atlas.quantize(
            colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        gif_frames = [
            frame.quantize(palette=master_palette, dither=Image.Dither.NONE)
            for frame in rgb_frames
        ]
        gif_frames[0].save(
            self.outfile,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate all saved WoS cooling steps and analytical comparisons."
    )
    parser.add_argument("summary", type=Path, help="time-summary CSV")
    parser.add_argument(
        "--history",
        type=Path,
        help="full-history CSV; defaults to <summary stem>_history.csv",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="GIF frames per second")
    parser.add_argument("-o", "--output", type=Path, help="output GIF path")
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
        }
    )


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"result file not found: {path}")
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))


def grid_series(
    history: np.ndarray, steps: np.ndarray, x: np.ndarray, y: np.ndarray, name: str
) -> np.ndarray:
    values = np.full((steps.size, y.size, x.size), np.nan, dtype=float)
    step_index = {int(step): index for index, step in enumerate(steps)}
    for row in history:
        frame = step_index[int(row["step"])]
        ix = int(np.searchsorted(x, row["x"]))
        iy = int(np.searchsorted(y, row["y"]))
        values[frame, iy, ix] = row[name]
    if np.isnan(values).any():
        raise ValueError(f"history CSV does not contain a complete grid for {name}")
    return values


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
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("--fps must be a positive finite number")

    summary_path = args.summary.resolve()
    history_path = args.history
    if history_path is None:
        history_path = summary_path.with_name(
            f"{summary_path.stem}_history{summary_path.suffix}"
        )
    history_path = history_path.resolve()
    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent
            / "results"
            / "animations"
            / f"{summary_path.stem}_animation.gif"
        )
    output_path = output_path.resolve().with_suffix(".gif")
    webp_path = output_path.with_name(f"{output_path.stem}_truecolor.webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = load_csv(summary_path)
    history = load_csv(history_path)
    steps = np.unique(history["step"]).astype(int)
    times = np.array(
        [history["time"][history["step"] == step][0] for step in steps], dtype=float
    )
    x = np.unique(history["x"])
    y = np.unique(history["y"])
    expected_rows = steps.size * x.size * y.size
    if history.size != expected_rows:
        raise ValueError(
            f"history has {history.size} rows; expected {expected_rows} "
            "for a complete step-by-grid dataset"
        )
    if summary.size != steps.size or not np.allclose(summary["time"], times):
        raise ValueError("summary and history time steps do not match")

    numerical = grid_series(history, steps, x, y, "numerical")
    exact = grid_series(history, steps, x, y, "exact_discrete")
    error = grid_series(history, steps, x, y, "error")

    configure_style()
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=True
    )
    layout = figure.add_gridspec(2, 3, height_ratios=(0.95, 1.0))
    curve_axis = figure.add_subplot(layout[0, :])
    numerical_axis = figure.add_subplot(layout[1, 0])
    exact_axis = figure.add_subplot(layout[1, 1])
    error_axis = figure.add_subplot(layout[1, 2])

    curve_axis.plot(
        summary["time"],
        summary["center_mean"],
        "o-",
        color="#2F5F8F",
        markersize=3.0,
        label="WoS implicit Euler",
    )
    curve_axis.plot(
        summary["time"],
        summary["center_exact_discrete"],
        color="#E6863B",
        linestyle="--",
        label="Discrete analytical solution",
    )
    curve_axis.plot(
        summary["time"],
        summary["center_exact_continuous"],
        color="#555555",
        linestyle=":",
        label="Continuous analytical solution",
    )
    cursor = curve_axis.axvline(times[0], color="#B23A48", linewidth=1.0)
    wos_marker, = curve_axis.plot(
        [times[0]], [summary["center_mean"][0]], "o", color="#B23A48", markersize=4
    )
    exact_marker, = curve_axis.plot(
        [times[0]],
        [summary["center_exact_discrete"][0]],
        "s",
        color="#E6863B",
        markersize=3.5,
    )
    curve_axis.set_xlabel("Time")
    curve_axis.set_ylabel("Normalised centre temperature excess, $\\theta$")
    curve_axis.set_xlim(times.min(), times.max())
    curve_axis.set_ylim(0.0, 1.04)
    curve_axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    curve_axis.legend(ncol=3, loc="upper right")
    add_panel_label(curve_axis, "a")

    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    # Level 0 uses the normalised temperature excess theta, whose prescribed
    # range is [0, 1].  Keep one shared norm object for every frame and both
    # fields so that neither imshow nor the colour bar can autoscale over time.
    temperature_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    error_limit = float(np.max(np.abs(error)))
    if error_limit == 0.0:
        error_limit = 1.0
    error_norm = mpl.colors.TwoSlopeNorm(
        vmin=-error_limit, vcenter=0.0, vmax=error_limit
    )
    numerical_image = numerical_axis.imshow(
        numerical[0], origin="lower", extent=extent, cmap="viridis",
        interpolation="nearest", norm=temperature_norm, aspect="equal"
    )
    exact_image = exact_axis.imshow(
        exact[0], origin="lower", extent=extent, cmap="viridis",
        interpolation="nearest", norm=temperature_norm, aspect="equal"
    )
    error_image = error_axis.imshow(
        error[0], origin="lower", extent=extent, cmap="RdBu_r",
        interpolation="nearest", norm=error_norm, aspect="equal"
    )

    for axis, title, label in (
        (numerical_axis, "WoS field", "b"),
        (exact_axis, "Discrete analytical field", "c"),
        (error_axis, "Error: WoS − analytical", "d"),
    ):
        axis.set_title(title, pad=4)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        add_panel_label(axis, label)

    temperature_colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=temperature_norm, cmap="viridis"),
        ax=[numerical_axis, exact_axis], orientation="horizontal",
        fraction=0.08, pad=0.16, ticks=np.linspace(0.0, 1.0, 6)
    )
    temperature_colorbar.set_label(
        "Normalised temperature excess, $\\theta$ (fixed 0–1 for all frames)"
    )
    error_colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap="RdBu_r"),
        ax=error_axis, orientation="horizontal", fraction=0.08, pad=0.16,
        ticks=np.linspace(-error_limit, error_limit, 5)
    )
    error_colorbar.set_label("Signed error (fixed range for all frames)")
    time_label = figure.suptitle("", fontsize=9, fontweight="bold")

    def update(frame: int):
        numerical_image.set_data(numerical[frame])
        exact_image.set_data(exact[frame])
        error_image.set_data(error[frame])
        cursor.set_xdata([times[frame], times[frame]])
        wos_marker.set_data([times[frame]], [summary["center_mean"][frame]])
        exact_marker.set_data(
            [times[frame]], [summary["center_exact_discrete"][frame]]
        )
        time_label.set_text(
            f"Whole-domain cooling: step {steps[frame]}, time = {times[frame]:.3f}"
        )
        return (
            numerical_image, exact_image, error_image, cursor,
            wos_marker, exact_marker, time_label,
        )

    movie = animation.FuncAnimation(
        figure, update, frames=np.arange(steps.size),
        interval=1000.0 / args.fps, blit=False, repeat=True
    )
    update(steps.size - 1)
    preview_base = (
        Path(__file__).resolve().parent
        / "results"
        / "figures"
        / f"{summary_path.stem}_animation_final"
    )
    preview_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        preview_base.with_suffix(".png"), dpi=300,
        bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        preview_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        preview_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        preview_base.with_suffix(".tiff"), dpi=600,
        bbox_inches="tight", facecolor="white"
    )
    update(0)
    movie.save(
        output_path,
        writer=FixedPaletteGifWriter(fps=args.fps, webp_path=webp_path),
        dpi=300,
        progress_callback=lambda frame, total: print(
            f"rendering frame {frame + 1}/{total}", end="\r", flush=True
        ),
    )
    print(f"\n{output_path}")
    print(webp_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
