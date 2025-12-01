import argparse
import os
from .core import generate_hashmol3d_from_file
from .version import __version__


def compute(args):
    """Compute HashMol3D identifier for a file."""
    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"File not found: {args.filepath}")

    res = generate_hashmol3d_from_file(
        args.filepath,
        precision=args.precision,
        charge=args.charge,
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
