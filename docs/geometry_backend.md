# FCPW geometry architecture

## Public interfaces

`GeometryScene<N>` is the geometry contract shared by WoS, welding applications,
and point classification:

```text
GeometryScene<N>
├── mesh()
├── closest_boundary()
└── classify_point()
```

`FcpwGeometryScene<2>` stores line segments and preserves per-segment boundary
ids. `FcpwGeometryScene<3>` stores triangles and requires a closed two-manifold
mesh because the solver classifies grid points by intersection parity.

Two-dimensional WoST uses the extended interface:

```text
GeometryScene<2>
    └── BoundaryScene2D
        └── FcpwBoundaryScene2D
            ├── full FCPW scene
            ├── Dirichlet FCPW scene
            └── Neumann FCPW scene with silhouettes
```

Primitive ids returned through project interfaces always refer to the original
full mesh, even when a query runs against a Dirichlet or Neumann subset.

## Precision policy

The solver stores coordinates and PDE values as `double`; FCPW uses `float`.
Each scene translates its bounding-box center to the origin and divides all
coordinates by the largest bounding-box extent before building FCPW. Query
points use the same transform. Closest points and distances are converted back
to world-space `double` values at the interface boundary.

Segments and triangles that collapse after normalization and float conversion
are rejected. WoST visibility endpoint tolerances also include a float-scale
floor so a requested boundary endpoint is not treated as an occluder.

## Three-dimensional classification

The classifier does not replace diagnostics with a single FCPW `contains()`
call. For every query away from the boundary it:

1. generates random unit directions from the existing deterministic PRNG;
2. repeatedly requests robust closest ray hits from FCPW;
3. rejects rays whose triangle barycentric coordinates indicate an edge or
   vertex hit;
4. counts intersection parity for three valid rays;
5. returns the majority vote, or `Indeterminate` when the configured attempt
   budget cannot produce three valid rays.

This preserves `ambiguous_ray_retries` and `indeterminate_points` metadata.

## MPI ownership

Rank 0 reads the input mesh and broadcasts only vertices, primitive indices,
and 2D boundary ids. Each rank builds an equivalent local FCPW scene. FCPW
internal nodes are never serialized, so MPI data formats do not depend on FCPW
implementation details or versions.

## Dependency configuration

FCPW is included as a recursive git submodule. The initial configuration is
CPU-only with Enoki, GPU support, bindings, demos, and FCPW tests disabled:

```bash
git submodule update --init --recursive
cmake -B build
cmake --build build -j
```
