#!/usr/bin/env python3
"""Animate Level-3 moving-source WoS, FD reference, and signed error fields."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from validate_level2 import load_metadata, numeric
from validate_level3 import (
    load_fields,
    load_numeric_csv,
    refined_reference_solution,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = APP_DIR / "results" / "data" / "level3_moving_heat"


class TrueColorWebPWriter(animation.PillowWriter):
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
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--reference-refinement", type=int, default=4)
    parser.add_argument("--reference-substeps", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("-o", "--output", type=Path)
    arguments = parser.parse_args()
    if arguments.reference_refinement <= 0 or arguments.reference_substeps <= 0:
        parser.error("reference refinement and substeps must be positive")
    if not math.isfinite(arguments.fps) or arguments.fps <= 0.0:
        parser.error("--fps must be positive and finite")
    return arguments


def rounded_limit(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    return math.ceil(2.0 * value / magnitude) * magnitude / 2.0


def add_plate_outline(
    axis: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    """Separate the rectangular plate from the light panel background."""
    xmin, xmax, ymin, ymax = extent
    width = xmax - xmin
    height = ymax - ymin
    axis.set_facecolor("#EEF1F4")
    # The dark outer stroke remains visible against the panel background;
    # the thin white inner stroke remains visible over low-temperature pixels.
    axis.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=False,
            edgecolor="#24272B",
            linewidth=3.0,
            clip_on=False,
            zorder=20,
        )
    )
    axis.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            clip_on=False,
            zorder=21,
        )
    )
    padding = 0.045 * max(width, height)
    axis.set_xlim(xmin - padding, xmax + padding)
    axis.set_ylim(ymin - padding, ymax + padding)
    axis.set_xticks(np.linspace(xmin, xmax, 5))
    axis.set_yticks(np.linspace(ymin, ymax, 5))


def main() -> None:
    args = parse_args()
    prefix = args.prefix.resolve()
    metadata = load_metadata(Path(f"{prefix}_metadata.csv"))
    summary = load_numeric_csv(Path(f"{prefix}_summary.csv"))
    history = load_numeric_csv(Path(f"{prefix}_history.csv"))
    steps, times, x, y, wos_fields, _ = load_fields(history)
    expected_steps = np.arange(
        int(round(numeric(metadata, "total_time") / numeric(metadata, "dt"))) + 1
    )
    if not np.array_equal(steps, expected_steps):
        raise ValueError("animation requires every field; use field stride 1")
    reference_cache = Path(
        f"{prefix}_reference_refine_{args.reference_refinement}_"
        f"substeps_{args.reference_substeps}.npz"
    )
    if reference_cache.is_file():
        with np.load(reference_cache) as cache:
            if not (
                np.array_equal(cache["steps"], steps)
                and np.allclose(cache["times"], times)
                and np.allclose(cache["x"], x)
                and np.allclose(cache["y"], y)
                and {"reference_x", "reference_y", "refined_fields"}
                <= set(cache.files)
            ):
                raise ValueError(
                    "cached Level-3 reference is missing native refined fields "
                    "or its axes do not match"
                )
            reference_fields = cache["fields"]
            reference_x = cache["reference_x"]
            reference_y = cache["reference_y"]
            refined_fields = cache["refined_fields"]
    else:
        reference_x, reference_y, refined_fields, reference_fields = (
            refined_reference_solution(
                metadata,
                steps,
                x,
                y,
                args.reference_refinement,
                args.reference_substeps,
            )
        )
    errors = wos_fields - reference_fields
    centre_iy = int(np.argmin(np.abs(y)))
    reference_centre_iy = int(np.argmin(np.abs(reference_y)))
    rmse = np.sqrt(np.mean(errors[:, 1:-1, 1:-1] ** 2, axis=(1, 2)))
    reference_peak = np.max(reference_fields, axis=(1, 2))
    normalized_rmse = np.divide(
        rmse,
        reference_peak,
        out=np.zeros_like(rmse),
        where=reference_peak > 0.0,
    )

    output = args.output
    if output is None:
        output = APP_DIR / "results" / "animations" / (
            f"{prefix.name}_comparison_truecolor.webp"
        )
    output = output.resolve().with_suffix(".webp")
    output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    layout = figure.add_gridspec(2, 3, height_ratios=(0.9, 1.0))
    profile_axis = figure.add_subplot(layout[0, :])
    wos_axis = figure.add_subplot(layout[1, 0])
    reference_axis = figure.add_subplot(layout[1, 1])
    error_axis = figure.add_subplot(layout[1, 2])

    heat_off_time = numeric(metadata, "heat_off_time")
    total_time = numeric(metadata, "total_time")
    source_start_x = numeric(metadata, "source_start_x")
    source_end_x = numeric(metadata, "source_end_x")
    weld_y = numeric(metadata, "weld_y")
    source_radius_mm = 1e3 * numeric(metadata, "sigma_x")
    reference_dt = numeric(metadata, "dt") / args.reference_substeps
    wos_profile, = profile_axis.plot(
        1e3 * x,
        wos_fields[0, centre_iy],
        color="#2F5F8F",
        lw=1.3,
        marker="o",
        ms=2.0,
        markevery=2,
        label=f"WoS ({x.size} x nodes)",
    )
    reference_profile, = profile_axis.plot(
        1e3 * reference_x,
        refined_fields[0, reference_centre_iy],
        color="#E6863B",
        lw=1.2,
        ls="--",
        label=f"refined FD ({reference_x.size} x nodes; dt={reference_dt:g} s)",
    )
    profile_source_line = profile_axis.axvline(
        1e3 * source_start_x,
        color="#54A9D6",
        lw=1.1,
        ls=":",
        label="source centre (active 0–5 s)",
    )
    profile_axis.set_title("Weld centreline profile: T(x, y=0)")
    profile_axis.set(xlabel="x (mm)", ylabel="Temperature rise (K)")
    profile_axis.set_xlim(1e3 * float(x.min()), 1e3 * float(x.max()))
    profile_axis.set_xticks(np.linspace(1e3 * x.min(), 1e3 * x.max(), 5))
    profile_axis.set_ylim(
        0.0,
        1.08 * max(float(np.max(wos_fields)), float(np.max(refined_fields))),
    )
    profile_axis.grid(True, color="#D9D9D9", lw=0.5, alpha=0.7)
    profile_axis.legend(loc="upper left", ncol=3)

    extent = (
        1e3 * float(x.min()),
        1e3 * float(x.max()),
        1e3 * float(y.min()),
        1e3 * float(y.max()),
    )
    temperature_limit = rounded_limit(
        max(float(np.max(wos_fields)), float(np.max(reference_fields)))
    )
    error_limit = rounded_limit(float(np.max(np.abs(errors))))
    # Avoid using pure black for zero temperature rise: the plate outline and
    # light panel background now carry the domain/outside distinction.
    temperature_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "inferno_plate",
        mpl.colormaps["inferno"](np.linspace(0.08, 1.0, 256)),
    )
    temperature_norm = mpl.colors.Normalize(0.0, temperature_limit)
    error_norm = mpl.colors.TwoSlopeNorm(
        vmin=-error_limit, vcenter=0.0, vmax=error_limit
    )
    wos_image = wos_axis.imshow(
        wos_fields[0], origin="lower", extent=extent, cmap=temperature_cmap,
        norm=temperature_norm, aspect="equal"
    )
    reference_image = reference_axis.imshow(
        refined_fields[0], origin="lower", extent=extent, cmap=temperature_cmap,
        norm=temperature_norm, aspect="equal"
    )
    error_image = error_axis.imshow(
        errors[0], origin="lower", extent=extent, cmap="RdBu_r",
        norm=error_norm, aspect="equal"
    )
    source_circles = []
    for axis in (wos_axis, reference_axis):
        axis.plot(
            [1e3 * source_start_x, 1e3 * source_end_x],
            [1e3 * weld_y, 1e3 * weld_y],
            color="white",
            lw=0.8,
            ls="--",
            alpha=0.7,
        )
        source_circle = patches.Circle(
            (1e3 * source_start_x, 1e3 * weld_y),
            radius=source_radius_mm,
            fill=False,
            edgecolor="#54D6FF",
            linewidth=1.5,
            zorder=18,
        )
        axis.add_patch(source_circle)
        source_circles.append(source_circle)
    for axis, title in (
        (wos_axis, f"WoS ({x.size} × {y.size})"),
        (
            reference_axis,
            (
                f"refined FD ({reference_x.size} × {reference_y.size}; "
                f"dt={reference_dt:g} s)"
            ),
        ),
        (error_axis, "WoS − FD at WoS nodes"),
    ):
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        add_plate_outline(axis, extent)
    figure.colorbar(
        mpl.cm.ScalarMappable(norm=temperature_norm, cmap=temperature_cmap),
        ax=[wos_axis, reference_axis],
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        label="Temperature rise (K; fixed range)",
    )
    figure.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap="RdBu_r"),
        ax=error_axis,
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        label="Signed error (K; fixed range)",
    )
    title = figure.suptitle("", fontsize=9, fontweight="bold")

    def update(frame: int):
        wos_image.set_data(wos_fields[frame])
        reference_image.set_data(refined_fields[frame])
        error_image.set_data(errors[frame])
        wos_profile.set_ydata(wos_fields[frame, centre_iy])
        reference_profile.set_ydata(
            refined_fields[frame, reference_centre_iy]
        )
        active = times[frame] <= heat_off_time + 1e-12
        source_x_mm = 1e3 * summary["source_center_x"][frame]
        profile_source_line.set_visible(active)
        profile_source_line.set_xdata([source_x_mm, source_x_mm])
        for source_circle in source_circles:
            source_circle.set_visible(active)
            source_circle.center = (source_x_mm, 1e3 * weld_y)
        phase = "moving heat source" if active else "source off, cooling"
        title.set_text(
            f"Level 3: {phase}; t={times[frame]:.1f} s | "
            f"RMSE={rmse[frame]:.3f} K ({100*normalized_rmse[frame]:.3f}%)"
        )
        return (
            wos_image,
            reference_image,
            error_image,
            wos_profile,
            reference_profile,
            profile_source_line,
            *source_circles,
            title,
        )

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=np.arange(steps.size),
        interval=1000.0 / args.fps,
        blit=False,
        repeat=True,
    )
    preview_dir = APP_DIR / "results" / "figures"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for target_time, suffix in ((heat_off_time, "source_off"), (total_time, "final")):
        frame = int(np.argmin(np.abs(times - target_time)))
        update(frame)
        preview_base = preview_dir / f"{prefix.name}_comparison_{suffix}"
        figure.savefig(
            preview_base.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            preview_base.with_suffix(".svg"),
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            preview_base.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
    update(0)
    movie.save(
        output,
        writer=TrueColorWebPWriter(fps=args.fps),
        dpi=args.dpi,
        progress_callback=lambda frame, total: print(
            f"rendering frame {frame+1}/{total}", end="\r", flush=True
        ),
    )
    print(f"\n{output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
