#!/usr/bin/env python3
"""Validate Level 3 against an independently advanced moving-source FD solve."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from validate_level2 import (
    DirichletSineSolver,
    gaussian_integral,
    load_metadata,
    numeric,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = APP_DIR / "results" / "data" / "level3_moving_heat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Level-3 output prefix without CSV suffixes",
    )
    parser.add_argument("--reference-refinement", type=int, default=4)
    parser.add_argument("--reference-substeps", type=int, default=16)
    parser.add_argument(
        "--max-normalized-rmse",
        type=float,
        default=0.01,
        help="hard full-field normalized RMSE gate (default: 0.01)",
    )
    parser.add_argument(
        "--max-peak-relative-error",
        type=float,
        default=0.02,
        help="peak temperature-rise relative error gate (default: 0.02)",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.reference_refinement <= 0:
        parser.error("--reference-refinement must be positive")
    if arguments.reference_substeps <= 0:
        parser.error("--reference-substeps must be positive")
    if not 0.0 < arguments.max_normalized_rmse < 1.0:
        parser.error("--max-normalized-rmse must be between zero and one")
    return arguments


def load_numeric_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    )


def load_fields(
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    standard_errors = np.full_like(fields, np.nan)
    step_to_frame = {step: frame for frame, step in enumerate(steps)}
    for row in history:
        frame = step_to_frame[int(round(float(row["step"])))]
        times[frame] = float(row["time"])
        ix = int(np.searchsorted(x, row["x"]))
        iy = int(np.searchsorted(y, row["y"]))
        fields[frame, iy, ix] = row["temperature_rise"]
        standard_errors[frame, iy, ix] = row["standard_error"]
    if np.isnan(fields).any() or np.isnan(standard_errors).any():
        raise ValueError("history CSV does not contain complete fields")
    return steps, times, x, y, fields, standard_errors


def moving_source_rate(
    xx: np.ndarray,
    yy: np.ndarray,
    time: float,
    metadata: dict[str, str],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    heat_off_time = numeric(metadata, "heat_off_time")
    if not 0.0 <= time < heat_off_time:
        return np.zeros_like(xx)
    start_x = numeric(metadata, "source_start_x")
    speed = numeric(metadata, "weld_speed")
    weld_y = numeric(metadata, "weld_y")
    sigma_x = numeric(metadata, "sigma_x")
    sigma_y = numeric(metadata, "sigma_y")
    absorbed_power = numeric(metadata, "absorbed_power")
    thickness = numeric(metadata, "thickness")
    density = numeric(metadata, "density")
    specific_heat = numeric(metadata, "specific_heat")
    centre_x = start_x + speed * time
    integral_x = gaussian_integral(xmin, xmax, centre_x, sigma_x)
    integral_y = gaussian_integral(ymin, ymax, weld_y, sigma_y)
    power_density = absorbed_power / (thickness * integral_x * integral_y) * np.exp(
        -0.5
        * (
            ((xx - centre_x) / sigma_x) ** 2
            + ((yy - weld_y) / sigma_y) ** 2
        )
    )
    return power_density / (density * specific_heat)


def refined_reference_solution(
    metadata: dict[str, str],
    steps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    refinement: int,
    substeps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return native refined axes/fields plus values sampled at WoS nodes."""
    dt = numeric(metadata, "dt")
    conductivity = numeric(metadata, "conductivity")
    density = numeric(metadata, "density")
    specific_heat = numeric(metadata, "specific_heat")
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
    interior_x = reference_x[1:-1]
    interior_y = reference_y[1:-1]
    heat_off_time = numeric(metadata, "heat_off_time")
    start_x = numeric(metadata, "source_start_x")
    speed = numeric(metadata, "weld_speed")
    weld_y = numeric(metadata, "weld_y")
    sigma_x = numeric(metadata, "sigma_x")
    sigma_y = numeric(metadata, "sigma_y")
    absorbed_power = numeric(metadata, "absorbed_power")
    thickness = numeric(metadata, "thickness")
    heat_capacity = density * specific_heat
    integral_y = gaussian_integral(
        float(y[0]), float(y[-1]), weld_y, sigma_y
    )
    gaussian_y = np.exp(-0.5 * ((interior_y - weld_y) / sigma_y) ** 2)
    sine_y_projection = solver.sine_y.T @ gaussian_y
    coefficients = np.zeros(
        (reference_nx - 2, reference_ny - 2), dtype=float
    )
    refined_fields = np.zeros(
        (steps.size, reference_y.size, reference_x.size), dtype=float
    )
    fields_at_wos_nodes = np.zeros((steps.size, y.size, x.size), dtype=float)
    step_to_frame = {int(step): frame for frame, step in enumerate(steps)}
    maximum_step = int(steps[-1])
    for outer_step in range(1, maximum_step + 1):
        outer_begin = (outer_step - 1) * dt
        for substep in range(substeps):
            time_midpoint = outer_begin + (substep + 0.5) * reference_dt
            if time_midpoint < heat_off_time:
                centre_x = start_x + speed * time_midpoint
                integral_x = gaussian_integral(
                    float(x[0]), float(x[-1]), centre_x, sigma_x
                )
                gaussian_x = np.exp(
                    -0.5 * ((interior_x - centre_x) / sigma_x) ** 2
                )
                sine_x_projection = solver.sine_x.T @ gaussian_x
                amplitude = absorbed_power / (
                    thickness * integral_x * integral_y * heat_capacity
                )
                source_coefficients = (
                    solver.forward_scale
                    * amplitude
                    * np.outer(sine_x_projection, sine_y_projection)
                )
                coefficients = (
                    coefficients + reference_dt * source_coefficients
                ) / solver.denominator
            else:
                coefficients = coefficients / solver.denominator
        if outer_step in step_to_frame:
            refined = np.zeros((reference_nx, reference_ny), dtype=float)
            refined[1:-1, 1:-1] = solver.inverse(coefficients)
            frame = step_to_frame[outer_step]
            refined_fields[frame] = refined.T
            fields_at_wos_nodes[frame] = refined.T[
                ::refinement, ::refinement
            ]
    return reference_x, reference_y, refined_fields, fields_at_wos_nodes


def refined_reference(
    metadata: dict[str, str],
    steps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    refinement: int,
    substeps: int,
) -> np.ndarray:
    """Compatibility wrapper returning the reference at WoS query nodes."""
    return refined_reference_solution(
        metadata, steps, x, y, refinement, substeps
    )[3]


def nearest_index(values: np.ndarray, value: float) -> int:
    index = int(np.argmin(np.abs(values - value)))
    if not math.isclose(float(values[index]), value, abs_tol=1e-12):
        raise ValueError(f"probe coordinate {value} is not on the query grid")
    return index


def validate_probes(
    prefix: Path,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wos_fields: np.ndarray,
    reference_fields: np.ndarray,
    dt: float,
) -> tuple[list[dict[str, float | str]], list[str]]:
    path = Path(f"{prefix}_probes.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    probe_names = list(dict.fromkeys(row["probe_id"] for row in source_rows))
    results: list[dict[str, float | str]] = []
    failures: list[str] = []
    for name in probe_names:
        rows = [row for row in source_rows if row["probe_id"] == name]
        px = float(rows[0]["x"])
        py = float(rows[0]["y"])
        ix = nearest_index(x, px)
        iy = nearest_index(y, py)
        wos_curve = wos_fields[:, iy, ix]
        reference_curve = reference_fields[:, iy, ix]
        wos_peak_frame = int(np.argmax(wos_curve))
        reference_peak_frame = int(np.argmax(reference_curve))
        wos_peak = float(wos_curve[wos_peak_frame])
        reference_peak = float(reference_curve[reference_peak_frame])
        peak_error = wos_peak - reference_peak
        relative_error = abs(peak_error) / max(reference_peak, 1e-12)
        peak_time_error = float(
            times[wos_peak_frame] - times[reference_peak_frame]
        )
        results.append(
            {
                "probe_id": name,
                "x": px,
                "y": py,
                "wos_peak_rise": wos_peak,
                "reference_peak_rise": reference_peak,
                "peak_error": peak_error,
                "peak_relative_error": relative_error,
                "wos_peak_time": float(times[wos_peak_frame]),
                "reference_peak_time": float(times[reference_peak_frame]),
                "peak_time_error": peak_time_error,
            }
        )
        if abs(peak_error) > max(5.0, 0.02 * reference_peak):
            failures.append(f"{name}: probe peak error is {peak_error:.3f} K")
        if abs(peak_time_error) > dt + 1e-12:
            failures.append(
                f"{name}: probe peak-time error is {peak_time_error:.3f} s"
            )
    return results, failures


def main() -> None:
    args = parse_args()
    prefix = args.prefix.resolve()
    summary = load_numeric_csv(Path(f"{prefix}_summary.csv"))
    history = load_numeric_csv(Path(f"{prefix}_history.csv"))
    metadata = load_metadata(Path(f"{prefix}_metadata.csv"))
    steps, times, x, y, wos_fields, standard_errors = load_fields(history)
    expected_steps = np.arange(
        int(round(numeric(metadata, "total_time") / numeric(metadata, "dt"))) + 1
    )
    if not np.array_equal(steps, expected_steps):
        raise ValueError("validation requires every field; rerun with field stride 1")

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
    reference_cache = Path(
        f"{prefix}_reference_refine_{args.reference_refinement}_"
        f"substeps_{args.reference_substeps}.npz"
    )
    np.savez_compressed(
        reference_cache,
        steps=steps,
        times=times,
        x=x,
        y=y,
        fields=reference_fields,
        reference_x=reference_x,
        reference_y=reference_y,
        refined_fields=refined_fields,
    )
    errors = wos_fields - reference_fields
    rows: list[dict[str, float | int]] = []
    failures: list[str] = []
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    location_gate = max(dx, dy) + 1e-12
    for frame, (step, time) in enumerate(zip(steps, times)):
        interior_error = errors[frame, 1:-1, 1:-1]
        interior_se = standard_errors[frame, 1:-1, 1:-1]
        rmse = float(np.sqrt(np.mean(interior_error**2)))
        reference_peak = float(np.max(reference_fields[frame]))
        wos_peak = float(np.max(wos_fields[frame]))
        normalized_rmse = rmse / reference_peak if reference_peak > 0.0 else 0.0
        reference_peak_iy, reference_peak_ix = np.unravel_index(
            int(np.argmax(reference_fields[frame])), reference_fields[frame].shape
        )
        wos_peak_iy, wos_peak_ix = np.unravel_index(
            int(np.argmax(wos_fields[frame])), wos_fields[frame].shape
        )
        peak_location_error = math.hypot(
            float(x[wos_peak_ix] - x[reference_peak_ix]),
            float(y[wos_peak_iy] - y[reference_peak_iy]),
        )
        coverage = float(
            np.mean(np.abs(interior_error) <= 3.0 * interior_se)
        ) if step > 0 else 1.0
        peak_relative_error = (
            abs(wos_peak - reference_peak) / reference_peak
            if reference_peak > 0.0
            else 0.0
        )
        rows.append(
            {
                "step": int(step),
                "time": float(time),
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "max_abs_error": float(np.max(np.abs(errors[frame]))),
                "wos_peak_rise": wos_peak,
                "reference_peak_rise": reference_peak,
                "peak_relative_error": peak_relative_error,
                "peak_location_error": peak_location_error,
                "coverage_3_conditional_se": coverage,
            }
        )
        if step > 0 and normalized_rmse > args.max_normalized_rmse:
            failures.append(
                f"t={time:.2f}: normalized RMSE {normalized_rmse:.3%} exceeds gate"
            )
        if step > 0 and peak_relative_error > args.max_peak_relative_error:
            failures.append(
                f"t={time:.2f}: peak relative error {peak_relative_error:.3%} exceeds gate"
            )
        if step > 0 and peak_location_error > location_gate:
            failures.append(
                f"t={time:.2f}: peak-location error {1e3*peak_location_error:.3f} mm"
            )

    absorbed_power = numeric(metadata, "absorbed_power")
    heat_off_time = numeric(metadata, "heat_off_time")
    expected_energy = absorbed_power * heat_off_time
    energy_error = abs(float(summary["cumulative_input_energy"][-1]) - expected_energy)
    if energy_error > 1e-10 * expected_energy:
        failures.append(f"source-energy error is {energy_error:.6g} J")
    if np.any(summary["max_steps_hits"] > 0.0):
        failures.append("at least one WoS path reached the maximum step count")
    if np.any(summary["negative_interior_points"] > 0.0):
        failures.append("negative interior temperature rises were reported")
    cooling = summary["time"] > heat_off_time + 1e-12
    if np.any(np.abs(summary["absorbed_power"][cooling]) > 1e-12):
        failures.append("nonzero source power was reported during cooling")
    expected_x = np.minimum(
        numeric(metadata, "source_start_x")
        + numeric(metadata, "weld_speed") * summary["time"],
        numeric(metadata, "source_end_x"),
    )
    if not np.allclose(summary["source_center_x"], expected_x, atol=1e-12):
        failures.append("reported moving-source trajectory is inconsistent")

    probe_rows, probe_failures = validate_probes(
        prefix,
        times,
        x,
        y,
        wos_fields,
        reference_fields,
        numeric(metadata, "dt"),
    )
    failures.extend(probe_failures)

    validation_path = args.output
    suffix = (
        f"_validation_refine_{args.reference_refinement}_substeps_"
        f"{args.reference_substeps}.csv"
    )
    if validation_path is None:
        validation_path = Path(f"{prefix}{suffix}")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    probe_path = validation_path.with_name(
        validation_path.stem + "_probes.csv"
    )
    with probe_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)

    print("Level-3 moving-source validation")
    print(
        f"  reference: {(x.size-1)*args.reference_refinement+1} x "
        f"{(y.size-1)*args.reference_refinement+1}, "
        f"dt={numeric(metadata, 'dt')/args.reference_substeps:g} s"
    )
    print(f"  source energy: {summary['cumulative_input_energy'][-1]:.6f} J")
    print("\n  time    RMSE (K)    NRMSE    peak error    location")
    report_times = {
        0.5, 1.5, 2.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0
    }
    for row in rows:
        if any(math.isclose(row["time"], target, abs_tol=1e-10) for target in report_times):
            print(
                f"  {row['time']:4.1f}  {row['rmse']:10.4f}  "
                f"{row['normalized_rmse']:8.3%}  "
                f"{row['peak_relative_error']:10.3%}  "
                f"{1e3*row['peak_location_error']:7.3f} mm"
            )
    print("\n  probe             peak error    time error")
    for row in probe_rows:
        print(
            f"  {row['probe_id']:<18} {row['peak_error']:9.3f} K  "
            f"{row['peak_time_error']:9.3f} s"
        )
    print(f"\n  validation CSV: {validation_path}")
    print(f"  probe CSV:      {probe_path}")
    if failures:
        print("\nFAILED")
        for failure in failures[:30]:
            print(f"  - {failure}")
        if len(failures) > 30:
            print(f"  - ... and {len(failures)-30} more")
        raise SystemExit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
