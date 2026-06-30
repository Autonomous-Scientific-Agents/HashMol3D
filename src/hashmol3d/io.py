"""File I/O helpers for HashMol3D."""

from __future__ import annotations

import numpy as np

from .periodic_table import get_atomic_num

__all__ = ["read_xyz"]


def read_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a standard XYZ file.

    The first line must contain the atom count; the second line is a
    free-form comment; subsequent lines contain ``symbol x y z`` (or
    ``Z x y z``).

    Args:
        path: Path to the XYZ file.

    Returns:
        ``(atomic_nums, coords)`` as NumPy arrays.

    Raises:
        ValueError: if the file is malformed (bad atom count, missing or
            extra atom lines, unknown element symbol, non-numeric coords).
    """
    with open(path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file is too short (need at least 2 header lines)")

    try:
        n_declared = int(lines[0].strip())
    except ValueError as err:
        raise ValueError(
            f"first line of an XYZ file must be the atom count; got {lines[0].strip()!r}"
        ) from err
    if n_declared <= 0:
        raise ValueError(f"XYZ atom count must be positive, got {n_declared}")

    z_list = []
    coords_list = []
    for raw in lines[2:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"malformed XYZ atom line: {line!r}")

        sym = parts[0]
        if sym.lstrip("-").isdigit():
            z = int(sym)
            if z <= 0 or z > 118:
                raise ValueError(f"atomic number out of range: {z}")
        else:
            z = get_atomic_num(sym)
            if z == 0:
                raise ValueError(f"unknown element symbol: {sym!r}")

        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as err:
            raise ValueError(f"malformed coordinates in XYZ line: {line!r}") from err

        z_list.append(z)
        coords_list.append(xyz)
        if len(z_list) == n_declared:
            break

    if len(z_list) != n_declared:
        raise ValueError(
            f"XYZ header declares {n_declared} atoms but only "
            f"{len(z_list)} valid atom lines were found"
        )

    return np.array(z_list, dtype=int), np.array(coords_list, dtype=float)
