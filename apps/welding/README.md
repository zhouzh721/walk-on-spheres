# Welding heat experiments

The welding application is separate from the original steady-state `wos`
launcher. It currently contains two validated development levels.

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

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target wos_welding_level0 wos_welding_level1
./build/apps/welding/wos_welding_level1
```

Both programs currently accept one MPI rank. Level 1 writes CSV data to
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
