#!/usr/bin/env python3
"""Build an independent finite-difference reference for welding level 1."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "results" / "data"


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))


def scalar(row: np.ndarray, name: str) -> float:
    return float(np.atleast_1d(row[name])[0])


def structured_field(field: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(field["x"])
    y = np.unique(field["y"])
    values = np.full((x.size, y.size), np.nan, dtype=float)
    values[np.searchsorted(x, field["x"]), np.searchsorted(y, field["y"])] = field[name]
    if np.isnan(values).any():
        raise ValueError(f"incomplete field for {name}")
    return x, y, values


def apply_negative_laplacian(interior: np.ndarray, hx: float, hy: float) -> np.ndarray:
    result = (2.0 / hx**2 + 2.0 / hy**2) * interior
    result[1:, :] -= interior[:-1, :] / hx**2
    result[:-1, :] -= interior[1:, :] / hx**2
    result[:, 1:] -= interior[:, :-1] / hy**2
    result[:, :-1] -= interior[:, 1:] / hy**2
    return result


def conjugate_gradient(
    right_hand_side: np.ndarray,
    hx: float,
    hy: float,
    relative_tolerance: float = 1e-11,
    max_iterations: int = 20_000,
) -> tuple[np.ndarray, int, float]:
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    residual_squared = float(np.sum(residual * residual))
    initial_norm = np.sqrt(residual_squared)
    if initial_norm == 0.0:
        return solution, 0, 0.0

    for iteration in range(1, max_iterations + 1):
        operator_direction = apply_negative_laplacian(direction, hx, hy)
        alpha = residual_squared / float(np.sum(direction * operator_direction))
        solution += alpha * direction
        residual -= alpha * operator_direction
        next_residual_squared = float(np.sum(residual * residual))
        relative_residual = np.sqrt(next_residual_squared) / initial_norm
        if relative_residual <= relative_tolerance:
            return solution, iteration, relative_residual
        beta = next_residual_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = next_residual_squared

    raise RuntimeError(
        f"finite-difference CG did not converge after {max_iterations} iterations"
    )


def main() -> None:
    summary = load_csv(DATA_DIR / "level1_gaussian_summary.csv")
    field = load_csv(DATA_DIR / "level1_gaussian_field.csv")
    coarse_x, coarse_y, wos_rise = structured_field(field, "temperature_rise")
    _, _, standard_error = structured_field(field, "standard_error")

    refinement = 4
    fine_nx = (coarse_x.size - 1) * refinement + 1
    fine_ny = (coarse_y.size - 1) * refinement + 1
    length_x = scalar(summary, "length_x")
    length_y = scalar(summary, "length_y")
    fine_x = np.linspace(0.0, length_x, fine_nx)
    fine_y = np.linspace(0.0, length_y, fine_ny)
    hx = fine_x[1] - fine_x[0]
    hy = fine_y[1] - fine_y[0]

    xx, yy = np.meshgrid(fine_x[1:-1], fine_y[1:-1], indexing="ij")
    source_x = scalar(summary, "source_x")
    source_y = scalar(summary, "source_y")
    sigma_x = scalar(summary, "sigma_x")
    sigma_y = scalar(summary, "sigma_y")
    absorbed_power = scalar(summary, "absorbed_power")
    thickness = scalar(summary, "thickness")
    conductivity = scalar(summary, "conductivity")
    integral_x = scalar(summary, "gaussian_integral_x")
    integral_y = scalar(summary, "gaussian_integral_y")
    shape = np.exp(
        -0.5
        * (
            ((xx - source_x) / sigma_x) ** 2
            + ((yy - source_y) / sigma_y) ** 2
        )
    )
    volumetric_power = absorbed_power * shape / (thickness * integral_x * integral_y)
    poisson_source = volumetric_power / conductivity

    reference_interior, iterations, relative_residual = conjugate_gradient(
        poisson_source, hx, hy
    )
    reference_fine = np.zeros((fine_nx, fine_ny), dtype=float)
    reference_fine[1:-1, 1:-1] = reference_interior
    reference = reference_fine[::refinement, ::refinement]
    if reference.shape != wos_rise.shape:
        raise RuntimeError("reference/coarse grid shape mismatch")

    error = wos_rise - reference
    interior = np.s_[1:-1, 1:-1]
    interior_error = error[interior]
    rmse = float(np.sqrt(np.mean(interior_error**2)))
    reference_peak = float(np.max(reference))
    wos_peak = float(np.max(wos_rise))
    normalized_rmse = rmse / reference_peak
    max_absolute_error = float(np.max(np.abs(interior_error)))
    centre = (coarse_x.size // 2, coarse_y.size // 2)
    centre_error = float(error[centre])
    centre_standard_error = float(standard_error[centre])
    peak_location_error = float(
        np.hypot(
            scalar(summary, "peak_x") - source_x,
            scalar(summary, "peak_y") - source_y,
        )
    )
    peak_location_limit = float(np.hypot(coarse_x[1] - coarse_x[0], coarse_y[1] - coarse_y[0]))
    within_three_se = float(
        np.mean(
            np.abs(interior_error)
            <= 3.0 * np.maximum(standard_error[interior], np.finfo(float).eps)
        )
    )

    reference_path = DATA_DIR / "level1_gaussian_reference.csv"
    with reference_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "x",
                "y",
                "reference_temperature_rise",
                "wos_temperature_rise",
                "standard_error",
                "error",
                "absolute_error",
            ]
        )
        for i, x in enumerate(coarse_x):
            for j, y in enumerate(coarse_y):
                writer.writerow(
                    [
                        f"{x:.17g}",
                        f"{y:.17g}",
                        f"{reference[i, j]:.17g}",
                        f"{wos_rise[i, j]:.17g}",
                        f"{standard_error[i, j]:.17g}",
                        f"{error[i, j]:.17g}",
                        f"{abs(error[i, j]):.17g}",
                    ]
                )

    centre_limit = max(3.0 * centre_standard_error, 0.03 * reference_peak)
    gates = {
        "power": scalar(summary, "power_relative_error") < 1e-6,
        "field_nrmse": normalized_rmse < 0.05,
        "centre": abs(centre_error) <= centre_limit,
        "peak_location": peak_location_error <= peak_location_limit,
        "symmetry": (
            scalar(summary, "symmetry_rms_relative_residual") < 0.02
            and scalar(
                summary, "symmetry_within_three_standard_errors_fraction"
            )
            > 0.99
        ),
        "max_steps": scalar(summary, "max_steps_hits") == 0.0,
    }
    validation_path = DATA_DIR / "level1_gaussian_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "fd_refinement",
                "fd_iterations",
                "fd_relative_residual",
                "wos_peak_rise",
                "reference_peak_rise",
                "field_rmse",
                "field_normalized_rmse",
                "field_max_absolute_error",
                "centre_error",
                "centre_standard_error",
                "peak_location_error",
                "peak_location_limit",
                "within_three_standard_errors_fraction",
                "power_gate",
                "field_nrmse_gate",
                "centre_gate",
                "peak_location_gate",
                "symmetry_gate",
                "max_steps_gate",
                "all_gates",
            ]
        )
        writer.writerow(
            [
                refinement,
                iterations,
                f"{relative_residual:.17g}",
                f"{wos_peak:.17g}",
                f"{reference_peak:.17g}",
                f"{rmse:.17g}",
                f"{normalized_rmse:.17g}",
                f"{max_absolute_error:.17g}",
                f"{centre_error:.17g}",
                f"{centre_standard_error:.17g}",
                f"{peak_location_error:.17g}",
                f"{peak_location_limit:.17g}",
                f"{within_three_se:.17g}",
                *[int(gates[name]) for name in gates],
                int(all(gates.values())),
            ]
        )

    print(f"finite-difference iterations: {iterations}")
    print(f"finite-difference relative residual: {relative_residual:.3e}")
    print(f"WoS/reference peak rise: {wos_peak:.6g} / {reference_peak:.6g} K")
    print(f"field RMSE / normalized RMSE: {rmse:.6g} K / {normalized_rmse:.3%}")
    print(f"centre error and standard error: {centre_error:.6g} / {centre_standard_error:.6g} K")
    print(f"peak location error / limit: {peak_location_error:.6g} / {peak_location_limit:.6g} m")
    print(f"points within 3 standard errors: {within_three_se:.2%}")
    for name, passed in gates.items():
        print(f"{name} gate: {'PASS' if passed else 'FAIL'}")
    print(reference_path)
    print(validation_path)

    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
