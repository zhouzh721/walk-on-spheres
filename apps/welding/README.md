# Welding heat experiments

The welding application is separate from the original steady-state `wos`
launcher. It currently contains two validated development levels and a
Level-2 transient welding baseline.

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

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target wos_welding_level0 wos_welding_level1 wos_welding_level2
./build/apps/welding/wos_welding_level1
./build/apps/welding/wos_welding_level2
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

All three programs currently accept one MPI rank. Level 1 writes CSV data to
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
