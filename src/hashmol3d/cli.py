import argparse
import os
import numpy as np
from typing import Tuple

from .core import generate_hashmol3d
from .periodic_table import get_atomic_num
from .version import __version__


def parse_xyz(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a standard .xyz file.
    Returns (atomic_nums, coords).
    """
    z_list = []
    coords_list = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Skip header (atom count) and comment line
    try:
        atom_lines = lines[2:]
    except IndexError:
        raise ValueError("File is too short to be XYZ")

    for line in atom_lines:
        parts = line.strip().split()
        if not parts:
            continue

        # Parse Symbol or Z
        sym = parts[0]
        if sym.isdigit():
            z = int(sym)
        else:
            z = get_atomic_num(sym)
            if z == 0:
                raise ValueError(f"Unknown element symbol: {sym}")

        # Parse Coords
        try:
            x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
        except (IndexError, ValueError):
            continue  # Skip malformed lines

        z_list.append(z)
        coords_list.append([x, y, z_coord])

    return np.array(z_list, dtype=int), np.array(coords_list, dtype=float)


def compute(args):
    """Compute HashMol3D identifier for a file."""
    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"File not found: {args.filepath}")

    # Parse XYZ file
    atomic_nums, coords = parse_xyz(args.filepath)

    # Set default charge if not provided
    charge = args.charge if args.charge is not None else 0

    # Generate hash
    res = generate_hashmol3d(
        atomic_nums=atomic_nums,
        coords=coords,
        precision=args.precision,
        charge=charge,
        multiplicity=args.multiplicity,
        hash_length=args.hash_length,
    )
    print(res.hash_str)


def version(args):
    """Show HashMol3D version."""
    print(f"HashMol3D version {__version__}")


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deterministic 3D molecular geometry hashing standard"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compute command
    compute_parser = subparsers.add_parser(
        "compute", help="Compute HashMol3D identifier for a file."
    )
    compute_parser.add_argument("filepath", help="Path to the molecular file")
    compute_parser.add_argument(
        "--precision", type=float, default=1e-4, help="Distance precision in Å."
    )
    compute_parser.add_argument(
        "--charge", type=int, default=None, help="Molecular charge"
    )
    compute_parser.add_argument(
        "--multiplicity", type=int, default=None, help="Spin multiplicity"
    )
    compute_parser.add_argument(
        "--hash-length", type=int, default=32, help="Hash string length"
    )
    compute_parser.set_defaults(func=compute)

    # version command
    version_parser = subparsers.add_parser("version", help="Show HashMol3D version.")
    version_parser.set_defaults(func=version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)
