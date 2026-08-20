#!/usr/bin/env python3
"""Discretize closed, piecewise-parametric 2D curves into an OBJ line mesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from geometry_expression import MathExpression

Point2D = tuple[float, float]


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("each 2D segment needs at least 2 sample points")
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def sample_segment(segment: dict[str, Any], count: int) -> list[Point2D]:
    name = segment.get("name", "unnamed")
    t_range = segment.get("t_range")
    if not isinstance(t_range, list) or len(t_range) != 2:
        raise ValueError(f"segment {name!r} needs t_range: [start, stop]")

    x_expr = MathExpression(str(segment["x"]), {"t"})
    y_expr = MathExpression(str(segment["y"]), {"t"})
    result = []
    for t in linspace(float(t_range[0]), float(t_range[1]), count):
        result.append((x_expr(t=t), y_expr(t=t)))
    return result


def signed_area(points: list[Point2D]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def parse_segment_overrides(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            name, raw_count = value.rsplit("=", 1)
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid --set-points value {value!r}; use NAME=COUNT") from exc
        if not name or count < 2:
            raise ValueError(f"invalid --set-points value {value!r}; COUNT must be >= 2")
        result[name] = count
    return result


def build_loops(
    config: dict[str, Any],
    points_per_segment: int | None,
    overrides: dict[str, int],
) -> list[tuple[str, list[Point2D]]]:
    tolerance = float(config.get("join_tolerance", 1e-9))
    if tolerance <= 0:
        raise ValueError("join_tolerance must be positive")

    loops = config.get("loops")
    if not isinstance(loops, list) or not loops:
        raise ValueError("the 2D config needs a non-empty loops array")

    used_overrides: set[str] = set()
    result: list[tuple[str, list[Point2D]]] = []
    for loop_index, loop in enumerate(loops):
        name = str(loop.get("name", f"loop_{loop_index}"))
        segments = loop.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"loop {name!r} needs a non-empty segments array")

        points: list[Point2D] = []
        for segment_index, segment in enumerate(segments):
            segment_name = str(segment.get("name", f"{name}_{segment_index}"))
            if segment_name in overrides:
                count = overrides[segment_name]
                used_overrides.add(segment_name)
            elif points_per_segment is not None:
                count = points_per_segment
            else:
                count = int(segment.get("points", 32))

            sampled = sample_segment(segment, count)
            if points:
                gap = distance(points[-1], sampled[0])
                if gap > tolerance:
                    raise ValueError(
                        f"loop {name!r} has a gap of {gap:g} before segment {segment_name!r}"
                    )
                sampled = sampled[1:]
            points.extend(sampled)

        closing_gap = distance(points[-1], points[0])
        if closing_gap > tolerance:
            raise ValueError(f"loop {name!r} is not closed; final gap is {closing_gap:g}")
        points.pop()  # OBJ line indices close the loop without a duplicate vertex.

        if len(points) < 3:
            raise ValueError(f"loop {name!r} contains fewer than 3 unique points")
        for i, point in enumerate(points):
            if distance(point, points[(i + 1) % len(points)]) <= tolerance:
                raise ValueError(f"loop {name!r} contains consecutive duplicate points")

        area = signed_area(points)
        if abs(area) <= tolerance * tolerance:
            raise ValueError(f"loop {name!r} has zero signed area")
        orientation = str(loop.get("orientation", "ccw")).lower()
        if orientation not in {"ccw", "cw"}:
            raise ValueError(f"loop {name!r} orientation must be 'ccw' or 'cw'")
        if (orientation == "ccw" and area < 0) or (orientation == "cw" and area > 0):
            points.reverse()
        result.append((name, points))

    unknown = overrides.keys() - used_overrides
    if unknown:
        raise ValueError(f"--set-points refers to unknown segments: {sorted(unknown)}")
    return result


def write_obj(path: Path, loops: list[tuple[str, list[Point2D]]], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# 2D boundary generated by discretize_geometry_2d.py\n")
        output.write(f"# config: {config_path}\n")
        for name, points in loops:
            output.write(f"# loop {name}: {len(points)} vertices\n")
            for x, y in points:
                output.write(f"v {x:.17g} {y:.17g} 0\n")

        offset = 1
        for name, points in loops:
            indices = list(range(offset, offset + len(points)))
            output.write(f"# closed polyline {name}\n")
            output.write("l " + " ".join(map(str, indices + [indices[0]])) + "\n")
            offset += len(points)


def preview(path: Path, loops: list[tuple[str, list[Point2D]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--preview requires matplotlib") from exc
    for name, points in loops:
        closed = points + [points[0]]
        plt.plot([p[0] for p in closed], [p[1] for p in closed], label=name)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="2D JSON geometry config")
    parser.add_argument("output", type=Path, help="output OBJ path")
    parser.add_argument(
        "--points-per-segment",
        type=int,
        help="override the sample count of every parametric segment",
    )
    parser.add_argument(
        "--set-points",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="override one named segment; may be repeated",
    )
    parser.add_argument("--preview", type=Path, help="optional preview PNG path")
    args = parser.parse_args()

    if args.points_per_segment is not None and args.points_per_segment < 2:
        parser.error("--points-per-segment must be >= 2")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        overrides = parse_segment_overrides(args.set_points)
        loops = build_loops(config, args.points_per_segment, overrides)
        write_obj(args.output, loops, args.config)
        if args.preview:
            preview(args.preview, loops)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    total = sum(len(points) for _, points in loops)
    print(f"Wrote {len(loops)} closed loop(s), {total} vertices to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
