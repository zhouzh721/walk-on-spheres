# Plot a selected field obtained from WoS.
# 2D output -> single pcolormesh with the polyline boundary overlay.
# 3D output -> XZ slice + YZ slice + 3D rendering with both slice planes overlaid.
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", default="wos.h5")
p.add_argument("-o", "--output", type=Path, help="output image (default: INPUT with .png suffix)")
p.add_argument("--force", action="store_true", help="allow an existing output image to be overwritten")
p.add_argument("--mesh", default="meshes/annulus.obj", help=".obj file for the mesh overlay")
p.add_argument("--cmap", default="RdBu_r")
p.add_argument(
    "--field",
    choices=("mean", "variance", "standard_error", "mean_steps"),
    default="mean",
    help="HDF5 field to visualise (default: mean)",
)
args = p.parse_args()

if args.output is None:
    input_path = Path(args.input)
    if args.field == "mean":
        args.output = input_path.with_suffix(".png")
    else:
        args.output = input_path.with_name(f"{input_path.stem}_{args.field}.png")
if args.output.exists() and not args.force:
    p.error(f"refusing to overwrite existing image: {args.output}; choose -o or pass --force")

# Selected fields have shape (Nx, Ny, Nz), with Nz=1 for 2D meshes.
with h5py.File(args.input, "r") as f:
    x: np.ndarray = f["x"][:]  # type: ignore[index]
    y: np.ndarray = f["y"][:]  # type: ignore[index]
    z: np.ndarray = f["z"][:]  # type: ignore[index]
    if args.field in f:
        values: np.ndarray = f[args.field][:]  # type: ignore[index]
    elif args.field == "mean" and "u" in f:
        # Backward compatibility with files written before the multi-field format.
        values = f["u"][:]  # type: ignore[index]
    else:
        p.error(f"field '{args.field}' is not present in {args.input}")
    location: np.ndarray | None = (
        f["location"][:] if "location" in f else None  # type: ignore[index]
    )

if location is not None:
    indeterminate = location == 3
    indeterminate_count = int(np.count_nonzero(indeterminate))
    if indeterminate_count:
        values = np.where(indeterminate, np.nan, values)
        print(
            f"Warning: {indeterminate_count} indeterminate 3D grid points "
            "are shown as missing values."
        )
is_3D = values.shape[2] > 1

field_labels = {
    "mean": "Mean",
    "variance": "Variance",
    "standard_error": "Standard error",
    "mean_steps": "Mean steps",
}
field_label = field_labels[args.field]
colorbar_label = r"$u$" if args.field == "mean" else field_label


def parse_obj_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    if not raw:
        raise ValueError(f"missing vertex index in OBJ token {token!r}")
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid OBJ vertex index {raw!r}") from exc
    if index <= 0:
        raise ValueError(
            f"OBJ vertex index must be a positive 1-based integer, found {index}"
        )
    if index > vertex_count:
        raise ValueError(
            f"OBJ vertex index {index} is outside the available range "
            f"of {vertex_count} vertices"
        )
    return index - 1


# read .obj boundary
verts_raw: list[list[float]] = []
segs: list[list[int]] = []
tris_raw: list[list[int]] = []
with open(args.mesh, encoding="utf-8") as mf:
    for line_number, line in enumerate(mf, 1):
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        try:
            if parts[0] == "v":
                if len(parts) < 3:
                    raise ValueError("vertex needs at least x and y")
                verts_raw.append(
                    [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]) if len(parts) > 3 else 0.0,
                    ]
                )
            elif parts[0] == "l":
                if len(parts) < 3:
                    raise ValueError("polyline needs at least two vertex indices")
                segs.append(
                    [parse_obj_index(token, len(verts_raw)) for token in parts[1:]]
                )
            elif parts[0] == "f":
                if len(parts) != 4:
                    raise ValueError("face must contain exactly three vertex indices")
                tris_raw.append(
                    [parse_obj_index(token, len(verts_raw)) for token in parts[1:]]
                )
        except (IndexError, ValueError) as exc:
            p.error(f"{args.mesh}:{line_number}: {exc}")

verts = np.array(verts_raw)
tris = np.array(tris_raw)

# diverging colour scale around 0 if data spans both signs
finite = values[np.isfinite(values)]  # type: ignore[index]
if finite.size == 0:
    p.error(f"field '{args.field}' contains no finite values")
if finite.min() < 0 < finite.max():
    vmax = float(np.max(np.abs(finite)))
    vmin = -vmax
else:
    vmin = float(finite.min())
    vmax = float(finite.max())
if vmin == vmax:
    padding = max(abs(vmin) * 1e-6, 1e-12)
    vmin -= padding
    vmax += padding

if not is_3D:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, values[:, :, 0].T, shading="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
    # mesh = ax.contourf(x, y, values[:, :, 0].T, levels=100, cmap=args.cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    for seg in segs:
        ax.plot(verts[seg, 0], verts[seg, 1], "k-", lw=1)
    pad = 0.10 * max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1]))  # 10% padding around mesh bbox
    ax.set_xlim(verts[:, 0].min() - pad, verts[:, 0].max() + pad)
    ax.set_ylim(verts[:, 1].min() - pad, verts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$u(x,y)$" if args.field == "mean" else f"{field_label}(x, y)")
else:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    y_mid = len(y) // 2
    x_mid = len(x) // 2
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(args.cmap)

    fig = plt.figure(figsize=(17, 6.5), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.20)

    # XZ slice at middle y
    ax_xz = fig.add_subplot(1, 3, 1)
    mesh_xz = ax_xz.pcolormesh(x, z, values[:, y_mid, :].T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title(f"{field_label}: XZ slice at y = {y[y_mid]:.2f}")

    # YZ slice at middle x
    ax_yz = fig.add_subplot(1, 3, 2)
    mesh_yz = ax_yz.pcolormesh(y, z, values[x_mid, :, :].T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title(f"{field_label}: YZ slice at x = {x[x_mid]:.2f}")

    fig.colorbar(mesh_xz, ax=[ax_xz, ax_yz], label=colorbar_label, shrink=0.9)

    # 3D mesh + slice planes overlaid
    ax_3D = fig.add_subplot(1, 3, 3, projection="3d")
    ax_3D.add_collection3d(Poly3DCollection([verts[t] for t in tris], alpha=0.33, edgecolor="k", facecolor="lightgray", linewidth=0.3))

    # XZ slice plane at y_mid, coloured by the selected field.
    XZ_x, XZ_z = np.meshgrid(x, z)
    ax_3D.plot_surface(XZ_x, np.full_like(XZ_x, y[y_mid]), XZ_z, facecolors=cmap(norm(values[:, y_mid, :].T)), shade=False, antialiased=False, rcount=len(z), ccount=len(x))

    # YZ slice plane at x_mid, coloured by the selected field.
    YZ_y, YZ_z = np.meshgrid(y, z)
    ax_3D.plot_surface(np.full_like(YZ_y, x[x_mid]), YZ_y, YZ_z, facecolors=cmap(norm(values[x_mid, :, :].T)), shade=False, antialiased=False, rcount=len(z), ccount=len(y))

    pad3D = 0.10 * np.ptp(verts, axis=0).max()  # 10% padding around bbox
    ax_3D.set_xlim(verts[:, 0].min() - pad3D, verts[:, 0].max() + pad3D)
    ax_3D.set_ylim(verts[:, 1].min() - pad3D, verts[:, 1].max() + pad3D)
    ax_3D.set_zlim(verts[:, 2].min() - pad3D, verts[:, 2].max() + pad3D)
    ax_3D.set_box_aspect(np.ptp(verts, axis=0))  # match data aspect ratio
    ax_3D.set_xlabel("x")
    ax_3D.set_ylabel("y")
    ax_3D.set_zlabel("z")
    ax_3D.set_title(f"Boundary mesh + {field_label.lower()} slices")

fig.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
