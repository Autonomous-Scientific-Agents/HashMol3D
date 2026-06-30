"""Command-line interface for HashMol3D."""

import argparse
import os
from typing import Tuple

import numpy as np

from .core import generate_hashmol3d
from .periodic_table import get_atomic_num
from .version import __version__


def parse_xyz(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a standard .xyz file.

    The first line must contain the atom count; the second line is a free-form
    comment; subsequent lines must contain ``symbol x y z`` (or ``Z x y z``).

    Returns ``(atomic_nums, coords)`` as NumPy arrays.
    """
    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file is too short (need at least 2 header lines)")

    try:
        n_declared = int(lines[0].strip())
    except ValueError as err:
        raise ValueError(
            f"First line of an XYZ file must be the atom count; got {lines[0].strip()!r}"
        ) from err
    if n_declared <= 0:
        raise ValueError(f"XYZ atom count must be positive, got {n_declared}")

    atom_lines = lines[2:]

    z_list = []
    coords_list = []
    for raw in atom_lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line: {line!r}")

        sym = parts[0]
        if sym.lstrip("-").isdigit():
            z = int(sym)
            if z <= 0 or z > 118:
                raise ValueError(f"Atomic number out of range: {z}")
        else:
            z = get_atomic_num(sym)
            if z == 0:
                raise ValueError(f"Unknown element symbol: {sym!r}")

        try:
            x = float(parts[1])
            y = float(parts[2])
            zc = float(parts[3])
        except ValueError as err:
            raise ValueError(f"Malformed coordinates in XYZ line: {line!r}") from err

        z_list.append(z)
        coords_list.append([x, y, zc])
        if len(z_list) == n_declared:
            break

    if len(z_list) != n_declared:
        raise ValueError(
            f"XYZ header declares {n_declared} atoms but only "
            f"{len(z_list)} valid atom lines were found"
        )

    return (
        np.array(z_list, dtype=int),
        np.array(coords_list, dtype=float),
    )


def compute(args):
    """Compute HashMol3D identifier for a file."""
    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"File not found: {args.filepath}")

    atomic_nums, coords = parse_xyz(args.filepath)

    charge = args.charge if args.charge is not None else 0

    res = generate_hashmol3d(
        atomic_nums=atomic_nums,
        coords=coords,
        precision=args.precision,
        charge=charge,
        multiplicity=args.multiplicity,
        hash_length=args.hash_length,
    )
    if args.verbose:
        print("descriptor:", res.descriptor)
        print("charge:", res.charge)
        print("multiplicity:", res.multiplicity)
        print("version:", res.version)
        print("hash:", res.hash_str)
    else:
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

    compute_parser = subparsers.add_parser(
        "compute", help="Compute HashMol3D identifier for an XYZ file."
    )
    compute_parser.add_argument("filepath", help="Path to the molecular file (.xyz)")
    compute_parser.add_argument(
        "--precision", type=float, default=1e-4, help="Distance precision in Å"
    )
    compute_parser.add_argument(
        "--charge", type=int, default=None, help="Total formal charge (default 0)"
    )
    compute_parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity (default: inferred from electron count)",
    )
    compute_parser.add_argument(
        "--hash-length",
        type=int,
        default=32,
        help="Number of hex characters retained (default 32; max 64)",
    )
    compute_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print the canonical descriptor and metadata",
    )
    compute_parser.set_defaults(func=compute)

    version_parser = subparsers.add_parser("version", help="Show HashMol3D version.")
    version_parser.set_defaults(func=version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)
