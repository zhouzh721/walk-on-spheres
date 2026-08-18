#!/usr/bin/env python3
"""Visualize a 2D polyline or 3D surface OBJ using Matplotlib."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObjMesh:
    vertices: np.ndarray
    polylines: list[list[int]]
    faces: list[list[int]]


def parse_index(token: str, vertex_count: int) -> int:
    """Convert a positive 1-based OBJ vertex reference to a zero-based index."""
    raw = token.split("/", 1)[0]
    if not raw:
        raise ValueError(f"missing vertex index in OBJ token {token!r}")
    index = int(raw)
    if index <= 0:
        raise ValueError(
            f"OBJ vertex index must be a positive 1-based integer, found {index}"
        )
    resolved = index - 1
    if resolved >= vertex_count:
        raise ValueError(
            f"OBJ vertex index {index} is outside the available range of {vertex_count} vertices"
        )
    return resolved


def load_obj(path: Path) -> ObjMesh:
    vertices: list[list[float]] = []
    polylines: list[list[int]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            kind = parts[0]
            try:
                if kind == "v":
                    if len(parts) < 3:
                        raise ValueError("vertex needs at least x and y")
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3]) if len(parts) >= 4 else 0.0
                    vertices.append([x, y, z])
                elif kind == "l":
                    if len(parts) < 3:
                        raise ValueError("polyline needs at least two vertex indices")
                    polylines.append([parse_index(token, len(vertices)) for token in parts[1:]])
                elif kind == "f":
                    if len(parts) < 4:
                        raise ValueError("face needs at least three vertex indices")
                    faces.append([parse_index(token, len(vertices)) for token in parts[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

    if not vertices:
        raise ValueError(f"{path} contains no OBJ vertices")

    return ObjMesh(np.asarray(vertices, dtype=float), polylines, faces)


def choose_dimension(mesh: ObjMesh, requested: str) -> int:
    if requested != "auto":
        return int(requested[0])
    if mesh.faces:
        return 3
    if mesh.polylines:
        return 2
    return 3 if np.ptp(mesh.vertices[:, 2]) > 1e-12 else 2


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    pad = 0.05 * span if span > 0 else 0.5
    return lower - pad, upper + pad


def draw_2d(ax, mesh: ObjMesh, show_vertices: bool, vertex_size: float) -> None:
    if mesh.polylines:
        for polyline in mesh.polylines:
            points = mesh.vertices[polyline]
            ax.plot(points[:, 0], points[:, 1], color="tab:blue", linewidth=1.2)
    elif mesh.faces:
        for face in mesh.faces:
            points = mesh.vertices[face + [face[0]]]
            ax.plot(points[:, 0], points[:, 1], color="tab:blue", linewidth=1.0)

    if show_vertices or (not mesh.polylines and not mesh.faces):
        ax.scatter(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            s=vertex_size,
            color="tab:red",
            zorder=3,
        )

    ax.set_xlim(*padded_limits(mesh.vertices[:, 0]))
    ax.set_ylim(*padded_limits(mesh.vertices[:, 1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def draw_3d(
    ax,
    mesh: ObjMesh,
    show_vertices: bool,
    vertex_size: float,
    face_alpha: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if mesh.faces:
        polygons = [mesh.vertices[face] for face in mesh.faces]
        collection = Poly3DCollection(
            polygons,
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=0.35,
            alpha=face_alpha,
        )
        ax.add_collection3d(collection)

    if mesh.polylines:
        segments = []
        for polyline in mesh.polylines:
            points = mesh.vertices[polyline]
            segments.extend([points[i : i + 2] for i in range(len(points) - 1)])
        ax.add_collection3d(Line3DCollection(segments, colors="tab:blue", linewidths=1.0))

    if show_vertices or (not mesh.polylines and not mesh.faces):
        ax.scatter(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            s=vertex_size,
            color="tab:red",
            depthshade=False,
        )

    ax.set_xlim(*padded_limits(mesh.vertices[:, 0]))
    ax.set_ylim(*padded_limits(mesh.vertices[:, 1]))
    ax.set_zlim(*padded_limits(mesh.vertices[:, 2]))
    spans = np.ptp(mesh.vertices, axis=0)
    positive = spans[spans > 0]
    fallback = float(positive.min()) if len(positive) else 1.0
    ax.set_box_aspect(np.where(spans > 0, spans, fallback))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def show_interactive(plt, fig) -> None:
    """Map and draw the GUI canvas before entering the blocking event loop."""
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)
    if window is not None:
        try:
            if hasattr(window, "geometry"):  # TkAgg
                window.geometry("900x760")
            elif hasattr(window, "resize"):  # QtAgg and similar backends
                window.resize(900, 760)
            if hasattr(window, "update_idletasks"):
                window.update_idletasks()
        except Exception:
            # Window sizing is optional and backend-specific; drawing still works.
            pass

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
    plt.show(block=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="2D or 3D OBJ file")
    parser.add_argument("-o", "--output", type=Path, help="optional output PNG/PDF/SVG")
    parser.add_argument(
        "--dimension",
        choices=("auto", "2d", "3d"),
        default="auto",
        help="rendering mode; auto uses faces for 3D and polylines for 2D",
    )
    parser.add_argument("--show-vertices", action="store_true", help="draw mesh vertices")
    parser.add_argument("--vertex-size", type=float, default=10.0, help="vertex marker size")
    parser.add_argument("--face-alpha", type=float, default=0.45, help="3D face opacity [0, 1]")
    parser.add_argument("--elev", type=float, default=30.0, help="3D elevation angle")
    parser.add_argument("--azim", type=float, default=45.0, help="3D azimuth angle")
    parser.add_argument("--dpi", type=int, default=180, help="saved image resolution")
    parser.add_argument("--title", help="custom plot title")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive window (useful when only saving)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.face_alpha <= 1.0:
        parser.error("--face-alpha must be between 0 and 1")
    if args.vertex_size <= 0:
        parser.error("--vertex-size must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.no_show and args.output is None:
        parser.error("--no-show requires --output; otherwise nothing would be displayed")

    try:
        global np
        import numpy as np
        import matplotlib
    except ModuleNotFoundError:
        parser.error("NumPy and Matplotlib are required; run: pip install -r requirements.txt")

    if args.no_show:
        matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        parser.error("Matplotlib is required; run: pip install -r requirements.txt")

    try:
        mesh = load_obj(args.input)
        dimension = choose_dimension(mesh, args.dimension)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if dimension == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        draw_2d(ax, mesh, args.show_vertices, args.vertex_size)
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.91)
    else:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        draw_3d(ax, mesh, args.show_vertices, args.vertex_size, args.face_alpha)
        ax.view_init(elev=args.elev, azim=args.azim)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.91)

    title = args.title or f"{args.input.name} ({dimension}D OBJ)"
    ax.set_title(title)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved visualization to {args.output}")
    print(
        f"Loaded {len(mesh.vertices)} vertices, {len(mesh.polylines)} polyline(s), "
        f"and {len(mesh.faces)} face(s) from {args.input}"
    )

    if not args.no_show:
        show_interactive(plt, fig)
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
