#!/usr/bin/env python3
"""Validate Level 2 against an independent implicit finite-difference solve."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = APP_DIR / "results" / "data" / "level2_transient_heat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Level-2 output prefix without _summary/_history/_metadata suffixes",
    )
    parser.add_argument(
        "--rmse-limit",
        type=float,
        default=0.05,
        help="maximum normalized field RMSE at a saved nonzero step",
    )
    parser.add_argument(
        "--center-limit",
        type=float,
        default=0.10,
        help="maximum center error normalized by the FD peak",
    )
    parser.add_argument(
        "--reference-substeps",
        type=int,
        default=1,
        help=(
            "implicit FD substeps per WoS step; values above one include "
            "the WoS run's time-discretization error in the comparison"
        ),
    )
    parser.add_argument(
        "--reference-refinement",
        type=int,
        default=1,
        help=(
            "integer refinement of each FD grid interval; the refined grid "
            "retains every WoS query node (default: 1)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="validation CSV path (default includes the reference substep count)",
    )
    arguments = parser.parse_args()
    if arguments.reference_substeps <= 0:
        parser.error("--reference-substeps must be positive")
    if arguments.reference_refinement <= 0:
        parser.error("--reference-refinement must be positive")
    return arguments


def load_numeric_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    )


def load_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {row["key"]: row["value"] for row in rows}


def numeric(metadata: dict[str, str], key: str) -> float:
    try:
        return float(metadata[key])
    except KeyError as error:
        raise KeyError(f"missing metadata key: {key}") from error


def heating_fraction(time_begin: float, time_end: float, heat_on_time: float) -> float:
    overlap_begin = max(0.0, time_begin)
    overlap_end = min(time_end, heat_on_time)
    if overlap_end <= overlap_begin:
        return 0.0
    return (overlap_end - overlap_begin) / (time_end - time_begin)


def gaussian_integral(lower: float, upper: float, center: float, sigma: float) -> float:
    root_two = math.sqrt(2.0)
    return sigma * math.sqrt(math.pi / 2.0) * (
        math.erf((upper - center) / (root_two * sigma))
        - math.erf((lower - center) / (root_two * sigma))
    )


class DirichletSineSolver:
    """Diagonalize the five-point Dirichlet Laplacian with sine bases."""

    def __init__(self, nx: int, ny: int, dx: float, dy: float, coefficient: float):
        self.interior_nx = nx - 2
        self.interior_ny = ny - 2
        ix = np.arange(1, self.interior_nx + 1, dtype=float)
        iy = np.arange(1, self.interior_ny + 1, dtype=float)
        px = np.arange(1, self.interior_nx + 1, dtype=float)
        py = np.arange(1, self.interior_ny + 1, dtype=float)
        self.sine_x = np.sin(
            math.pi * np.outer(ix, px) / (self.interior_nx + 1)
        )
        self.sine_y = np.sin(
            math.pi * np.outer(iy, py) / (self.interior_ny + 1)
        )
        lambda_x = 4.0 / dx**2 * np.sin(
            math.pi * px / (2.0 * (self.interior_nx + 1))
        ) ** 2
        lambda_y = 4.0 / dy**2 * np.sin(
            math.pi * py / (2.0 * (self.interior_ny + 1))
        ) ** 2
        self.denominator = 1.0 + coefficient * (
            lambda_x[:, None] + lambda_y[None, :]
        )
        self.forward_scale = (
            2.0 / (self.interior_nx + 1)
            * 2.0 / (self.interior_ny + 1)
        )

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        coefficients = self.forward(right_hand_side)
        coefficients /= self.denominator
        return self.inverse(coefficients)

    def forward(self, field: np.ndarray) -> np.ndarray:
        return self.forward_scale * (
            self.sine_x.T @ field @ self.sine_y
        )

    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        return self.sine_x @ coefficients @ self.sine_y.T


def history_fields(
    history: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    x = np.unique(history["x"])
    y = np.unique(history["y"])
    if x.size != nx or y.size != ny:
        raise ValueError(
            f"history grid is {x.size} x {y.size}, metadata says {nx} x {ny}"
        )
    fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step_value in np.unique(history["step"]):
        step = int(round(float(step_value)))
        rows = history[np.isclose(history["step"], step_value)]
        if rows.size != nx * ny:
            raise ValueError(f"step {step} has {rows.size} rows, expected {nx * ny}")
        temperature = np.full((nx, ny), np.nan)
        standard_error = np.full((nx, ny), np.nan)
        ii = np.searchsorted(x, rows["x"])
        jj = np.searchsorted(y, rows["y"])
        temperature[ii, jj] = rows["temperature_rise"]
        standard_error[ii, jj] = rows["standard_error"]
        if np.isnan(temperature).any() or np.isnan(standard_error).any():
            raise ValueError(f"incomplete history field at step {step}")
        fields[step] = temperature, standard_error
    return x, y, fields


def main() -> None:
    args = parse_args()
    prefix = args.prefix.resolve()
    metadata = load_metadata(Path(f"{prefix}_metadata.csv"))
    summary = load_numeric_csv(Path(f"{prefix}_summary.csv"))
    history = load_numeric_csv(Path(f"{prefix}_history.csv"))

    nx = int(round(numeric(metadata, "nx")))
    ny = int(round(numeric(metadata, "ny")))
    dt = numeric(metadata, "dt")
    total_time = numeric(metadata, "total_time")
    heat_on_time = numeric(metadata, "heat_on_time")
    length_x = numeric(metadata, "length_x")
    length_y = numeric(metadata, "length_y")
    thickness = numeric(metadata, "thickness")
    conductivity = numeric(metadata, "conductivity")
    density = numeric(metadata, "density")
    specific_heat = numeric(metadata, "specific_heat")
    absorbed_power = numeric(metadata, "absorbed_power")
    sigma_x = numeric(metadata, "sigma_x")
    sigma_y = numeric(metadata, "sigma_y")
    diffusivity = conductivity / (density * specific_heat)
    step_count = int(round(total_time / dt))

    x, y, saved_fields = history_fields(history, nx, ny)
    if not np.isclose(x[-1] - x[0], length_x) or not np.isclose(
        y[-1] - y[0], length_y
    ):
        raise ValueError("history coordinates disagree with plate dimensions")

    refinement = args.reference_refinement
    reference_nx = (nx - 1) * refinement + 1
    reference_ny = (ny - 1) * refinement + 1
    reference_x = np.linspace(x[0], x[-1], reference_nx)
    reference_y = np.linspace(y[0], y[-1], reference_ny)
    reference_dx = reference_x[1] - reference_x[0]
    reference_dy = reference_y[1] - reference_y[0]

    integral_x = gaussian_integral(x[0], x[-1], 0.0, sigma_x)
    integral_y = gaussian_integral(y[0], y[-1], 0.0, sigma_y)
    xx, yy = np.meshgrid(
        reference_x[1:-1], reference_y[1:-1], indexing="ij"
    )
    source = absorbed_power / (thickness * integral_x * integral_y) * np.exp(
        -0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)
    )

    reference_dt = dt / args.reference_substeps
    solver = DirichletSineSolver(
        reference_nx,
        reference_ny,
        reference_dx,
        reference_dy,
        diffusivity * reference_dt,
    )
    fd_coefficients = np.zeros(
        (reference_nx - 2, reference_ny - 2), dtype=float
    )
    source_rate_coefficients = solver.forward(source / (density * specific_heat))
    validation_rows: list[dict[str, float]] = []
    failures: list[str] = []

    if not np.isfinite(history["temperature_rise"]).all():
        failures.append("non-finite WoS temperature values")
    if not np.isfinite(history["standard_error"]).all():
        failures.append("non-finite WoS standard errors")
    if np.min(history["temperature_rise"]) < -1e-10:
        failures.append(
            "negative temperature rise violates the nonnegative maximum principle: "
            f"minimum is {np.min(history['temperature_rise']):.12g} K"
        )
    if np.any(summary["max_steps_hits"] != 0):
        failures.append("one or more WoS paths reached max_steps")

    expected_input = absorbed_power * heat_on_time
    actual_input = float(summary["cumulative_input_energy"][-1])
    input_error = actual_input - expected_input
    if abs(input_error) > 1e-10 * max(1.0, expected_input):
        failures.append(
            f"source energy mismatch: got {actual_input:.12g}, expected {expected_input:.12g}"
        )

    for step in range(step_count + 1):
        if step > 0:
            outer_time_begin = (step - 1) * dt
            for substep in range(args.reference_substeps):
                time_begin = outer_time_begin + substep * reference_dt
                time_end = time_begin + reference_dt
                fraction = heating_fraction(
                    time_begin, time_end, heat_on_time
                )
                fd_coefficients = (
                    fd_coefficients
                    + reference_dt * fraction * source_rate_coefficients
                ) / solver.denominator

        if step not in saved_fields:
            continue
        wos_field, standard_error = saved_fields[step]
        reference_field = np.zeros((reference_nx, reference_ny), dtype=float)
        reference_field[1:-1, 1:-1] = solver.inverse(fd_coefficients)
        fd_field = reference_field[::refinement, ::refinement]
        if fd_field.shape != wos_field.shape:
            raise RuntimeError(
                f"refined FD extraction produced {fd_field.shape}, "
                f"expected {wos_field.shape}"
            )
        interior_error = wos_field[1:-1, 1:-1] - fd_field[1:-1, 1:-1]
        rmse = float(np.sqrt(np.mean(interior_error**2)))
        maximum_absolute = float(np.max(np.abs(interior_error)))
        fd_peak = float(np.max(fd_field))
        normalized_rmse = rmse / fd_peak if fd_peak > 0.0 else 0.0
        center_i = nx // 2
        center_j = ny // 2
        center_error = float(
            wos_field[center_i, center_j] - fd_field[center_i, center_j]
        )
        normalized_center_error = (
            abs(center_error) / fd_peak if fd_peak > 0.0 else 0.0
        )
        interior_se = standard_error[1:-1, 1:-1]
        coverage_3se = float(
            np.mean(np.abs(interior_error) <= 3.0 * interior_se)
        )
        validation_rows.append(
            {
                "step": step,
                "time": step * dt,
                "fd_reference_dt": reference_dt,
                "fd_reference_refinement": refinement,
                "fd_center_rise": float(fd_field[center_i, center_j]),
                "wos_center_rise": float(wos_field[center_i, center_j]),
                "center_error": center_error,
                "normalized_center_error": normalized_center_error,
                "field_rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "maximum_absolute_error": maximum_absolute,
                "mean_conditional_standard_error": float(np.mean(interior_se)),
                "coverage_3_conditional_se": coverage_3se,
            }
        )
        if step > 0 and normalized_rmse > args.rmse_limit:
            failures.append(
                f"step {step}: normalized RMSE {normalized_rmse:.3%} "
                f"exceeds {args.rmse_limit:.3%}"
            )
        if step > 0 and normalized_center_error > args.center_limit:
            failures.append(
                f"step {step}: normalized center error "
                f"{normalized_center_error:.3%} exceeds {args.center_limit:.3%}"
            )

    if not validation_rows:
        raise ValueError("history contains no fields that match the simulated steps")
    validation_path = args.output
    if validation_path is None:
        validation_path = Path(
            f"{prefix}_validation_refine_{refinement}"
            f"_substeps_{args.reference_substeps}.csv"
        )
    validation_path = validation_path.resolve()
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    key_times = sorted(
        set(
            min(validation_rows, key=lambda row: abs(row["time"] - target))["time"]
            for target in (0.5, heat_on_time, heat_on_time + 1.0, 6.0, total_time)
        )
    )
    print("Level-2 independent finite-difference validation")
    print(f"  prefix:              {prefix}")
    print(f"  WoS grid / steps:    {nx} x {ny} / {step_count}")
    print(
        f"  FD reference grid:   {reference_nx} x {reference_ny} "
        f"({refinement}x intervals)"
    )
    print(
        f"  FD reference dt:     {reference_dt:g} s "
        f"({args.reference_substeps} substeps/WoS step)"
    )
    print(f"  heating interval:    0 to {heat_on_time:g} s")
    print(f"  source energy:       {actual_input:.12g} J")
    print(f"  source-energy error: {input_error:.6g} J")
    print()
    print(" time      FD center    WoS center    field RMSE    normalized    3SE coverage")
    for row in validation_rows:
        if row["time"] in key_times:
            print(
                f"{row['time']:6.2f}  {row['fd_center_rise']:12.5g}"
                f"  {row['wos_center_rise']:12.5g}"
                f"  {row['field_rmse']:12.5g}"
                f"  {row['normalized_rmse']:10.3%}"
                f"  {row['coverage_3_conditional_se']:11.3%}"
            )
    print(f"\n  validation CSV: {validation_path}")
    if failures:
        print("\nFAILED")
        for failure in failures[:20]:
            print(f"  - {failure}")
        if len(failures) > 20:
            print(f"  - ... and {len(failures) - 20} more")
        raise SystemExit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
