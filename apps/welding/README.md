# Welding heat experiments

The welding application is separate from the original steady-state `wos`
launcher. It currently contains four development levels through the validated
Level-3 moving-source welding baseline.

## Level 0: transient cooling

`wos_welding_level0` advances an implicit-Euler heat problem on the unit
square. It compares every time step with the discrete and continuous
analytical cooling solutions.

## Level 1: stationary heat input

`wos_welding_level1` runs two steady 2D Poisson cases on a rectangular thin
plate:

1. a manufactured sine solution with a known volumetric source;
2. a finite-domain-normalized stationary Gaussian welding heat source.

The second case uses SI units. Its built-in baseline is a
`0.12 m x 0.08 m x 0.006 m` plate, constant conductivity, a centred Gaussian
source, and zero temperature rise on all four edges. The Gaussian is
normalized so that

```text
thickness * integral(volumetric power density dA) = efficiency * power.
```

Level 1 deliberately has no runtime parameter interface yet. Change the
documented `Level1Config` values in `src/level1_stationary_heat.cpp` while the
case definition and validation gates are being stabilized.

The source integral reuses the core 2D Poisson Green-function importance
sampler from `include/wos/poisson.hpp`. This substantially reduces estimator
variance relative to uniform disk sampling for the localized Gaussian heat
input and keeps the welding estimator consistent with the main `poisson`
equation.

## Level 2: heating, arc shutoff, and cooling

`wos_welding_level2` combines the Level-0 implicit-Euler time loop with the
Level-1 finite-plate Gaussian source. The stationary source is active for
`3 s` by default and the plate then cools until `10 s`. At every time step it
solves

```text
(-Delta + alpha^2) theta^(n+1)
    = q'''^(n+1) / k + alpha^2 theta^n,
alpha^2 = rho c / (k dt),
```

using screened-Green source sampling. The complete previous field is stored
on a Cartesian query grid and interpolated at random source points. Level 2
uses local tensor-product cubic interpolation by default; retain
`--history-interpolation bilinear` for regression comparisons. Cubic
interpolation reduces the cumulative artificial diffusion caused by repeatedly
projecting the previous solution onto a piecewise bilinear field. Because this
baseline has a nonnegative source, initial condition, and boundary rise, the
history contribution is clamped at zero to enforce the corresponding maximum
principle against tiny cubic overshoots near the cold boundary.

This first baseline deliberately retains constant properties, zero
temperature rise on all four edges, one MPI rank, and a stationary source.
The reported per-point standard error is conditional on the stored previous
field; uncertainty propagated from earlier time steps must later be measured
with repeated complete runs using independent seeds.

The current fixed-seed numerical baseline uses a `49 x 33` history grid,
`dt = 0.1 s`, and 500 paths per interior point. On eight OpenMP threads the
complete 10 s case took about 101 s in the development environment. Against a
finite-difference reference with four times finer spatial intervals and 16
times finer time steps, the normalized full-field RMSE was approximately
`0.18%` at 0.5 s, `0.22%` at arc shutoff (3 s), `0.20%` at 4 s, `0.20%` at
6 s, and `0.17%` at 10 s. The 10 s centre rises were `62.80 K` (WoS) and
`62.98 K` (reference). The corresponding repeated-whole-run, multi-seed
uncertainty gate is still pending; the per-step standard errors alone do not
measure propagated uncertainty.

### Time-step and path-count sensitivity

A controlled `0--4 s` comparison used the same `49 x 33` grid and random seed
for all four combinations below. Errors are measured at 4 s against the same
`193 x 129`, `dt = 0.00625 s` finite-difference reference, whose centre rise is
`143.68 K`.

| WoS dt | Paths/point | Centre rise | Centre conditional SE | Normalized field RMSE | Runtime to 4 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 s | 500 | 141.87 K | 0.19 K | 0.199% | 39 s |
| 0.10 s | 1000 | 141.69 K | 0.14 K | 0.200% | 78 s |
| 0.05 s | 500 | 140.11 K | 0.09 K | 0.276% | 79 s |
| 0.05 s | 1000 | 139.95 K | 0.07 K | 0.244% | 157 s |

Increasing the path count reduces conditional Monte Carlo noise without
removing interpolation bias. Reducing the time step on a fixed history grid
increases the number of field reconstructions and can therefore increase the
accumulated interpolation bias, even though the implicit-Euler truncation
error is smaller. Consequently, `dt = 0.1 s` remains the Level-2 default. Use
1000 paths if lower pointwise noise justifies approximately twice the runtime;
a time step below 0.1 s should be paired with history-grid refinement or a
different history representation.

## Level 3: moving heat source followed by cooling

`wos_welding_level3` keeps the validated Level-2 material model and time
integration, but replaces the stationary source with a fixed single-pass weld:

- a `0.12 m x 0.08 m x 0.006 m` plate and `97 x 65` query/history grid
  (`1.25 mm` spacing in both in-plane directions);
- constant `k = 35 W/(m K)`, `rho = 7800 kg/m3`, and `cp = 600 J/(kg K)`;
- a finite-plate-normalized Gaussian with `sigma_x = sigma_y = 6 mm`;
- `800 W` electrical power, `0.75` efficiency, and `600 W` absorbed power;
- motion from `x = -50 mm` to `x = +50 mm` along `y = 0` at `20 mm/s`
  during `0--5 s`;
- complete source shutoff at `5 s`, followed by source-free cooling to `15 s`;
- `dt = 0.1 s`, cubic history interpolation, and screened-Green sampling.

The plate edges are at `x = +/-60 mm`; the source-centre path ends are
therefore 10 mm inboard. With the 6 mm characteristic source radius, the
displayed source circle remains fully inside the plate throughout the pass.

The physical, geometry, trajectory, grid, and time parameters are deliberately
fixed. Runtime options only select path count, random seed, OpenMP threads,
field-output stride, and output prefix. Within each outer time step, two-point
Gauss-Legendre quadrature averages the continuously moving source rather than
moving it in a single 2 mm jump. Finite-plate normalization is evaluated at
both temporal quadrature points, giving exactly `3000 J` total input energy.

The run writes complete temperature fields, per-step summaries, metadata, and
seven grid-aligned thermal-cycle probes. At `t > 5 s`, the absorbed power is
exactly zero. The `15 s` endpoint is an observation time, not a claim that the
plate has returned to ambient temperature.

The retained strict baseline uses 1000 paths per point. It took about `1189 s`
on eight OpenMP threads in the development environment. Against an independent
`385 x 257`, `dt = 0.1 s` finite-difference reference, it produced:

| Time | Full-field RMSE | Normalized RMSE | Peak-rise relative error |
| ---: | ---: | ---: | ---: |
| 0.5 s | 0.0328 K | 0.084% | 0.430% |
| 2.5 s | 0.0627 K | 0.102% | 0.083% |
| 5.0 s | 0.0820 K | 0.133% | 0.172% |
| 8.0 s | 0.0597 K | 0.139% | 0.048% |
| 10.0 s | 0.0487 K | 0.131% | 0.007% |
| 12.0 s | 0.0455 K | 0.137% | 0.001% |
| 15.0 s | 0.0385 K | 0.132% | 0.042% |

All five centreline-probe peak times agreed with the reference at the saved
`0.1 s` resolution. The most weakly heated transverse probe differed by
`0.065 K` in peak rise with no peak-time difference. The final centre rise was
`28.52 K`, confirming that ten seconds of cooling does not return the plate
to ambient. The strict validator passed without changing simulation outputs or
relaxing its original `1%` field, `2%` peak, one-cell peak-location, and one-step
probe-timing gates. The 500-path result remains a useful sensitivity case, but
fails two strict gates because its propagated random noise cannot resolve a
very flat off-axis thermal-cycle maximum.

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target wos_welding_level0 wos_welding_level1 wos_welding_level2 wos_welding_level3
./build/apps/welding/wos_welding_level1
./build/apps/welding/wos_welding_level2
./build/apps/welding/wos_welding_level3 --walks 1000 --threads 8 \
    --output-prefix apps/welding/results/data/level3_moving_heat_120mm_highres_n1000
```

Use `wos_welding_level2 --help` for its grid, time, walk-count, seed, field
output stride, and output-path options. The default run writes metadata, a
per-step summary, and full field history below `apps/welding/results/data`.
When OpenMP is available, Level 2 evaluates independent query points in
parallel; use `--threads N` to select a thread count explicitly.

Validate a completed Level-2 run against an independent implicit
finite-difference solution with:

```bash
python3 apps/welding/tools/validate_level2.py
```

The reference diagonalizes the five-point Dirichlet Laplacian independently
of the WoS implementation. The validation reports field RMSE, centre error,
maximum error, and coverage by three reported conditional standard errors.
Use `--reference-substeps N` to advance the finite-difference reference with
a smaller time step and include the WoS run's implicit-Euler truncation error
in the comparison. Use `--reference-refinement N` to refine every spatial grid
interval while retaining the WoS query nodes. The validator advances modal
coefficients directly, so refined references remain inexpensive. Validation
CSV names include both settings so different references do not overwrite one
another.

Generate the complete Level-2 heating/cooling animation with the default WoS
field, a four-times spatially refined and 16-times temporally refined FD
reference, and their signed error:

```bash
python3 apps/welding/tools/animate_level2.py \
    --prefix apps/welding/results/data/level2_cubic_baseline \
    --reference-refinement 4 --reference-substeps 16 \
    --output apps/welding/results/animations/level2_validation_animation_truecolor.webp
```

The 101 saved time steps are retained. Temperature and signed-error colour
limits are fixed over the entire animation so that apparent changes cannot be
caused by frame-wise rescaling.

Validate the retained Level-3 baseline with the independent moving-source FD
reference:

```bash
python3 apps/welding/tools/validate_level3.py \
    --prefix apps/welding/results/data/level3_moving_heat_120mm_highres_n1000 \
    --reference-refinement 4 --reference-substeps 1
```

Generate its complete moving-heating and source-off cooling comparison:

```bash
python3 apps/welding/tools/animate_level3.py \
    --prefix apps/welding/results/data/level3_moving_heat_120mm_highres_n1000 \
    --reference-refinement 4 --reference-substeps 1 \
    --output apps/welding/results/animations/level3_moving_heat_120mm_highres_n1000_comparison.webp
```

All four programs currently accept one MPI rank. Level 1 writes CSV data to
`apps/welding/results/data`.

Create the independent finite-difference reference and validate the result:

```bash
python3 apps/welding/tools/validate_level1.py
```

Create the comparison figure:

```bash
python3 apps/welding/tools/plot_level1.py
```

The validation checks absorbed-power normalization, agreement with a refined
finite-difference reference, centre temperature, statistical symmetry, and
maximum-walk termination. It exits unsuccessfully if a gate fails.

The fixed-seed baseline used during implementation produced:

- manufactured-solution field RMSE: about `0.63 K` for a `100 K` peak;
- Gaussian field RMSE against the finite-difference reference: about `8.03 K`;
- Gaussian normalized field RMSE: about `0.86%`;
- internal points within three reported standard errors: about `99.7%`;
- absorbed-power integration error: approximately `2.4e-13`.

Generated CSV files and figures are intentionally ignored by Git.
