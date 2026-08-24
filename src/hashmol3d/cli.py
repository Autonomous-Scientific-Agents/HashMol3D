"""Command-line interface for HashMol3D."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .core import hash_molecule
from .io import read_xyz
from .version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hashmol3d",
        description=(
            "Deterministic 3D molecular geometry hash. "
            "Reads an XYZ file and prints the HashMol3D identifier."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hashmol3d {__version__}",
    )
    parser.add_argument("file", help="Path to a molecular geometry file (.xyz)")
    parser.add_argument(
        "-p",
        "--precision",
        type=float,
        default=1e-4,
        metavar="Å",
        help="Geometry-grid precision in angstroms; power of ten <= 1 (default: 1e-4)",
    )
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        default=0,
        help="Total formal charge (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity (default: inferred from electron count)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=None,
        help="Number of hex characters in the geometry hash, 1-64 "
        "(default: 32 = 128-bit; size by corpus, not molecule)",
    )
    parser.add_argument(
        "--method",
        choices=("frame", "canonical"),
        default="frame",
        help="Geometry descriptor method (default: frame)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print the canonical descriptor and metadata",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the HashMol3D CLI.

    Returns the process exit code. ``argv`` may be passed for testing;
    if omitted, ``sys.argv[1:]`` is used.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        atomic_nums, coords = read_xyz(args.file)
        result = hash_molecule(
            atomic_nums,
            coords,
            precision=args.precision,
            charge=args.charge,
            multiplicity=args.multiplicity,
            length=args.length,
            method=args.method,
        )
    except FileNotFoundError as err:
        print(f"hashmol3d: {err}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as err:
        print(f"hashmol3d: {err}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"identifier:    {result.hash_str}")
        print(f"formula:       {result.formula}")
        print(f"geometry_hash: {result.geometry_hash}")
        print(f"descriptor:    {result.descriptor}")
        print(f"version:       {result.version}")
        print(f"precision:     {result.precision}")
        print(f"charge:        {result.charge}")
        print(f"multiplicity:  {result.multiplicity}")
    else:
        print(result.hash_str)
    return 0


def main() -> None:
    """Entry point used by the ``hashmol3d`` console script."""
    sys.exit(cli())
