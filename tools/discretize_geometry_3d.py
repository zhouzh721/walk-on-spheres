#!/usr/bin/env python3
"""Discretize double-parametric 3D surfaces into a triangular OBJ mesh."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geometry_expression import MathExpression

Point3D = tuple[float, float, float]
Triangle = tuple[int, int, int]


def parameter_values(bounds: list[float], count: int, periodic: bool) -> list[float]:
    if count < 2:
        raise ValueError("u_points and v_points must be >= 2")
    start, stop = map(float, bounds)
    denominator = count if periodic else count - 1
    return [start + (stop - start) * i / denominator for i in range(count)]


class VertexWelder:
    def __init__(self, tolerance: float):
        if tolerance <= 0:
            raise ValueError("weld_tolerance must be positive")
        self.tolerance = tolerance
        self.vertices: list[Point3D] = []
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _key(self, point: Point3D) -> tuple[int, int, int]:
        return tuple(math.floor(value / self.tolerance) for value in point)  # type: ignore[return-value]

    def add(self, point: Point3D) -> int:
        key = self._key(point)
        for delta in itertools.product((-1, 0, 1), repeat=3):
            nearby = tuple(key[i] + delta[i] for i in range(3))
            for index in self.buckets.get(nearby, []):
                old = self.vertices[index]
                if math.dist(point, old) <= self.tolerance:
                    return index
        index = len(self.vertices)
        self.vertices.append(point)
        self.buckets[key].append(index)
        return index


def triangle_area_twice(vertices: list[Point3D], face: Triangle) -> float:
    a, b, c = (vertices[index] for index in face)
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return math.sqrt(sum(value * value for value in cross))


def signed_volume(vertices: list[Point3D], faces: list[Triangle]) -> float:
    volume6 = 0.0
    for ia, ib, ic in faces:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        volume6 += a[0] * (b[1] * c[2] - b[2] * c[1])
        volume6 += a[1] * (b[2] * c[0] - b[0] * c[2])
        volume6 += a[2] * (b[0] * c[1] - b[1] * c[0])
    return volume6 / 6.0


def topology_counts(faces: list[Triangle]) -> tuple[int, int]:
    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        edges.update((tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a))))
    boundary_edges = sum(count == 1 for count in edges.values())
    nonmanifold_edges = sum(count > 2 for count in edges.values())
    return boundary_edges, nonmanifold_edges


def build_surface(
    config: dict[str, Any],
    override_u: int | None,
    override_v: int | None,
) -> tuple[list[Point3D], list[Triangle], int, int]:
    tolerance = float(config.get("weld_tolerance", 1e-9))
    welder = VertexWelder(tolerance)
    faces: list[Triangle] = []
    face_keys: set[tuple[int, int, int]] = set()
    skipped_degenerate = 0

    patches = config.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("the 3D config needs a non-empty patches array")

    for patch_index, patch in enumerate(patches):
        name = str(patch.get("name", f"patch_{patch_index}"))
        u_bounds = patch.get("u_range")
        v_bounds = patch.get("v_range")
        if not isinstance(u_bounds, list) or len(u_bounds) != 2:
            raise ValueError(f"patch {name!r} needs u_range: [start, stop]")
        if not isinstance(v_bounds, list) or len(v_bounds) != 2:
            raise ValueError(f"patch {name!r} needs v_range: [start, stop]")

        nu = override_u if override_u is not None else int(patch.get("u_points", 32))
        nv = override_v if override_v is not None else int(patch.get("v_points", 32))
        periodic_u = bool(patch.get("periodic_u", False))
        periodic_v = bool(patch.get("periodic_v", False))
        u_values = parameter_values(u_bounds, nu, periodic_u)
        v_values = parameter_values(v_bounds, nv, periodic_v)

        expressions = [MathExpression(str(patch[axis]), {"u", "v"}) for axis in ("x", "y", "z")]
        grid: list[list[int]] = []
        for u in u_values:
            row = []
            for v in v_values:
                point = tuple(expression(u=u, v=v) for expression in expressions)
                row.append(welder.add(point))  # type: ignore[arg-type]
            grid.append(row)

        u_cells = nu if periodic_u else nu - 1
        v_cells = nv if periodic_v else nv - 1
        flip = bool(patch.get("flip_normals", False))
        for i in range(u_cells):
            ni = (i + 1) % nu
            for j in range(v_cells):
                nj = (j + 1) % nv
                candidates = [
                    (grid[i][j], grid[ni][j], grid[ni][nj]),
                    (grid[i][j], grid[ni][nj], grid[i][nj]),
                ]
                for face in candidates:
                    if flip:
                        face = (face[0], face[2], face[1])
                    key = tuple(sorted(face))
                    if len(set(face)) < 3 or triangle_area_twice(welder.vertices, face) <= tolerance * tolerance:
                        skipped_degenerate += 1
                    elif key in face_keys:
                        raise ValueError(f"patch {name!r} generated a duplicate triangle")
                    else:
                        face_keys.add(key)
                        faces.append(face)

    if not faces:
        raise ValueError("surface generation produced no non-degenerate triangles")

    boundary_edges, nonmanifold_edges = topology_counts(faces)
    closed = bool(config.get("closed", True))
    if closed and (boundary_edges or nonmanifold_edges):
        raise ValueError(
            "surface is not a closed 2-manifold: "
            f"{boundary_edges} boundary edge(s), {nonmanifold_edges} non-manifold edge(s)"
        )

    orientation = str(config.get("orientation", "outward")).lower()
    if orientation not in {"outward", "inward", "unchanged"}:
        raise ValueError("orientation must be 'outward', 'inward', or 'unchanged'")
    if closed and orientation != "unchanged":
        volume = signed_volume(welder.vertices, faces)
        if abs(volume) <= tolerance**3:
            raise ValueError("closed surface has near-zero signed volume")
        if (orientation == "outward" and volume < 0) or (orientation == "inward" and volume > 0):
            faces = [(a, c, b) for a, b, c in faces]

    return welder.vertices, faces, boundary_edges, skipped_degenerate


def write_obj(path: Path, vertices: list[Point3D], faces: list[Triangle], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# 3D surface generated by discretize_geometry_3d.py\n")
        output.write(f"# config: {config_path}\n")
        output.write(f"# vertices: {len(vertices)}, triangles: {len(faces)}\n")
        for x, y, z in vertices:
            output.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in faces:
            output.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="3D JSON surface config")
    parser.add_argument("output", type=Path, help="output triangular OBJ path")
    parser.add_argument("--u-points", type=int, help="override u samples for every patch")
    parser.add_argument("--v-points", type=int, help="override v samples for every patch")
    args = parser.parse_args()

    if args.u_points is not None and args.u_points < 2:
        parser.error("--u-points must be >= 2")
    if args.v_points is not None and args.v_points < 2:
        parser.error("--v-points must be >= 2")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        vertices, faces, boundary_edges, skipped = build_surface(
            config, args.u_points, args.v_points
        )
        write_obj(args.output, vertices, faces, args.config)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(
        f"Wrote {len(vertices)} vertices and {len(faces)} triangles to {args.output}; "
        f"boundary edges={boundary_edges}, skipped degenerate triangles={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
