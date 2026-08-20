# Walk-on-Spheres PDE Solver
This repo implements a minimal walk-on-spheres-based PDE solver for various elliptic PDEs on arbitrary input 2D or 3D meshes.

## Dependencies
- C++17 compiler
- CMake >= 3.20
- MPI (e.g. Open MPI, MPICH, Cray MPI)
- HDF5 with parallel support
- FCPW and Eigen (included as recursively initialized git submodules)
- Python >= 3.10 with `numpy`, `h5py`, `matplotlib` (result inspection and visualisation)

### MacOS
```bash
brew install cmake open-mpi hdf5-mpi
```

### HPC
Load the appropriate modules, for example:
```bash
module load cmake cray-hdf5-parallel
```

Submit using a slurm script.

## Build
```bash
git submodule update --init --recursive
cmake -B build
cmake --build build -j
```

FCPW is configured as a scalar CPU-only dependency in the initial integration;
Enoki, GPU support, demos, bindings, and FCPW's own tests are disabled.

## Geometry backend

FCPW is the geometry backend for two- and three-dimensional WoS, the welding
applications, 3D point classification, and two-dimensional WoST. Solvers and
applications depend on the project-owned `GeometryScene<N>` interface;
`BoundaryScene2D` extends it with Dirichlet/Neumann queries needed by WoST.
FCPW and Eigen types therefore do not leak into equations or solver APIs.

Geometry is normalized before conversion to FCPW's single-precision
representation, and query results are converted back to the solver's `double`
coordinates. In MPI runs rank 0 broadcasts only `Mesh<N>`; every rank then
builds its own FCPW scene. Three-dimensional domains are checked for closed
two-manifold topology, and classification retains multi-ray voting,
edge/vertex ambiguity retries, and `Indeterminate` diagnostics.

The previous project-owned BVH implementation has been removed. See
[`docs/geometry_backend.md`](docs/geometry_backend.md) for the current geometry
architecture and numerical assumptions.

## Run
```bash
mpirun -np <ranks> ./build/wos [options] [equation] [Nx Ny] [Nz] [mesh.obj]
```

All positional arguments are optional. The dimension of the mesh is auto-detected from the file. Pass an unknown equation name to get the list of registered equations.

Use `-o/--output` to choose the HDF5 output and `--force` only when an existing
file should intentionally be replaced. Without `--output`, the default remains
`wos.h5`; an existing output is never overwritten implicitly.

Random paths use a seed generated from the current time by default. Pass
`--seed <uint64>` to reproduce the same per-grid-point random paths. Point seeds
are derived from global grid indices, so a fixed seed is independent of MPI rank
decomposition for an otherwise identical run.

Examples:

```bash
# 2D Poisson with Green-function source importance sampling
mpirun -np 8 ./build/wos poisson 64 64 meshes/unit_circle.obj \
  --source-mode green --output results/circle_poisson_green_64.h5

# 2D screened Poisson on the annulus, 64x64 grid, 8 ranks
mpirun -np 8 ./build/wos screened_poisson 64 64 meshes/annulus.obj \
  --output results/annulus_screened_poisson_64.h5

# Reproducible 2D Laplace run
mpirun -np 8 ./build/wos laplace 64 64 meshes/annulus.obj \
  --seed 12345 --output results/annulus_laplace_seed_12345.h5

# 3D screened Poisson on the sphere, 32x32x32 grid
mpirun -np 8 ./build/wos screened_poisson 32 32 32 meshes/sphere.obj \
  --output results/sphere_screened_poisson_32.h5
```

Create the output directory first when needed, for example `mkdir -p results`.

## Visualise
```bash
pip install -r requirements.txt
python plot.py --mesh meshes/annulus.obj wos.h5
```

Each new HDF5 result contains the datasets `mean`, `variance`,
`standard_error`, `mean_steps`, and `location`. Location codes are `0` for
outside, `1` for inside, and `2` for boundary. Select a numerical result with
`--field`; `mean` is the default:

```bash
python plot.py results/run.h5 --mesh meshes/annulus.obj --field variance
python plot.py results/run.h5 --mesh meshes/annulus.obj --field standard_error
python plot.py results/run.h5 --mesh meshes/annulus.obj --field mean_steps
```

The same file stores `N_walks`, `epsilon`, and `seed` under `/metadata`.
Poisson and screened Poisson results additionally store `source_mode`; they
also store `alpha`, which is `0` for Poisson and the requested screening
coefficient for screened Poisson. Source mode `0` is uniform ball sampling and
mode `1` is Green-function importance sampling.
`plot.py` can still visualise `mean` from older files whose solution dataset is
named `u`.

The `--mesh` argument must match the mesh used to produce the `.h5` file. By
default, the image uses the input stem (`results/run.h5` becomes
`results/run.png`). Non-mean fields append their name, for example
`results/run_variance.png`. Use `-o/--output` for another name. Existing images
are protected; pass `--force` only to replace one intentionally.

## Inspect one grid point

Read one already-computed global grid index directly from the HDF5 result:

```bash
# 2D
python tools/inspect_result.py results/run.h5 --index 32,40

# 3D
python tools/inspect_result.py results/run.h5 --index 16,20,24
```

The command prints the physical coordinate, location, saved WoS statistics, and
actual number of walks. It does not rerun WoS or require MPI.

## Adding a new equation
Each equation lives in its own `src/equations/<name>.cpp` and is registered in `src/equations/equations.hpp`. When creating a new equation, the relevant files need to be added to the CMakeLists.txt.

## Welding heat experiments

The optional `apps/welding` application contains the staged transient-cooling
and stationary Gaussian heat-source verification cases used to develop welding
temperature simulation. See
[`apps/welding/README.md`](apps/welding/README.md) for the model assumptions,
build commands, outputs, and validation workflow.

## Custom continuous geometry

Separate tools are provided for discretising continuous geometry into the OBJ
meshes consumed by the solver:

- `tools/discretize_geometry_2d.py` samples piecewise parametric 2D curves.
- `tools/discretize_geometry_3d.py` triangulates double-parametric 3D surfaces.

Both sample counts are configurable. See [docs/custom_geometry.md](docs/custom_geometry.md)
for config formats, examples, topology checks, and command-line overrides.

Preview any generated 2D or 3D OBJ independently of solver output:

```bash
python tools/visualize_obj.py meshes/unit_circle.obj
python tools/visualize_obj.py meshes/sphere.obj --show-vertices
```

# global
SRC=/mnt/c/Users/Admin/Desktop/wos/walk-on-spheres
BUILD=$HOME/wos-build
RUN=$HOME/wos-run
CASE="laplace_$(date +%Y%m%d_%H%M%S)"
cd $SRC

# build
cmake --build "$BUILD" -j

# geometry visual
python3 tools/visualize_obj.py \
  meshes/annulus_128.obj \
  --show-vertices

# run
mpirun -np 4 "$HOME/wos-build/wos" \
  laplace 128 128 \
  "meshes/annulus_128.obj" \
  --output "results/${CASE}.h5"

# plot
python3 "plot.py" \
  "results/${CASE}.h5" \
  --mesh "meshes/annulus_128.obj"

# rsync
# rsync -av "$RUN/results/" "$SRC/results/"
