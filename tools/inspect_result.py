#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


LOCATION_NAMES = {
    0: "Outside",
    1: "Inside",
    2: "Boundary",
    3: "Indeterminate",
}


def parse_index(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("index must contain integers separated by commas") from error

    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError("index must be I,J for 2D or I,J,K for 3D")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("index components must be non-negative")
    return values


parser = argparse.ArgumentParser(
    description="Print saved WoS statistics for one global grid index."
)
parser.add_argument("input", type=Path, help="WoS HDF5 result file")
parser.add_argument(
    "--index",
    required=True,
    type=parse_index,
    metavar="I,J[,K]",
    help="global grid index to inspect",
)
args = parser.parse_args()

required_datasets = {
    "mean",
    "variance",
    "standard_error",
    "mean_steps",
    "location",
    "x",
    "y",
    "z",
    "metadata/N_walks",
}

with h5py.File(args.input, "r") as result:
    missing = sorted(name for name in required_datasets if name not in result)
    if missing:
        parser.error(
            f"{args.input} does not contain the required datasets: {', '.join(missing)}"
        )

    shape = result["mean"].shape
    if len(shape) != 3:
        parser.error(f"mean dataset must have three dimensions, found shape {shape}")

    is_3d = shape[2] > 1
    expected_components = 3 if is_3d else 2
    if len(args.index) != expected_components:
        parser.error(
            f"this result requires {expected_components} index components, "
            f"but {len(args.index)} were provided"
        )

    index = args.index if is_3d else (args.index[0], args.index[1], 0)
    if any(component >= extent for component, extent in zip(index, shape)):
        parser.error(f"index {args.index} is outside dataset bounds {shape}")

    i, j, k = index
    coordinate = (
        float(result["x"][i]),
        float(result["y"][j]),
        float(result["z"][k]),
    )
    location_code = int(round(float(result["location"][index])))
    if location_code not in LOCATION_NAMES:
        parser.error(f"unknown location code {location_code} at index {args.index}")

    mean = float(result["mean"][index])
    variance = float(result["variance"][index])
    standard_error = float(result["standard_error"][index])
    mean_steps = float(result["mean_steps"][index])
    n_walks = int(result["metadata/N_walks"][()])

location_name = LOCATION_NAMES[location_code]

print("Grid point:")
if is_3d:
    print(f"  index:                    ({i}, {j}, {k})")
    print(
        "  coordinate:               "
        f"({coordinate[0]:.12g}, {coordinate[1]:.12g}, {coordinate[2]:.12g})"
    )
else:
    print(f"  index:                    ({i}, {j})")
    print(
        "  coordinate:               "
        f"({coordinate[0]:.12g}, {coordinate[1]:.12g})"
    )
print(f"  location:                 {location_name}")

if location_name == "Inside":
    print(f"  mean:                     {mean:.12g}")
    print(f"  variance:                 {variance:.12g}")
    print(f"  standard error:           {standard_error:.12g}")
    print(f"  average steps per walk:   {mean_steps:.12g}")
    print(f"  number of walks:          {n_walks}")
elif location_name == "Boundary":
    print(f"  boundary value:           {mean:.12g}")
    print("  number of walks:          0")
elif location_name == "Indeterminate":
    print("  WoS was not run because the 3D inside test remained ambiguous.")
    print("  number of walks:          0")
else:
    print("  WoS was not run for this outside point.")
