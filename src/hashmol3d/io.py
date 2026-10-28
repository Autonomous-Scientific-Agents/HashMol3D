"""File I/O helpers for HashMol3D."""

from __future__ import annotations

import re
import warnings

import numpy as np

from .periodic_table import get_atomic_num

__all__ = ["read_xyz"]

# ASCII digits only: ``int()`` would also accept ``"1_0"`` and digits such as
# ``"²"`` that ``str.isdigit`` reports.
_COUNT = re.compile(r"\+?[0-9]+")
# An atomic number, optionally written as an integer-valued float (``6.0``),
# as ``hash_molecule`` accepts it.
_ATOMIC_NUMBER = re.compile(r"[+-]?[0-9]+(?:\.0*)?")


def read_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a standard XYZ file.

    The first line must contain the atom count; the second line is a
    free-form comment; subsequent lines contain ``symbol x y z`` (or
    ``Z x y z``).

    Args:
        path: Path to the XYZ file.

    Returns:
        ``(atomic_nums, coords)`` as NumPy arrays.

    Reading stops after the declared number of atoms, so the first frame of a
    multi-frame trajectory file is returned. Any non-blank content after that
    point triggers a ``UserWarning`` naming the file, because an under-declared
    atom count would otherwise silently hash a truncated molecule. The path is
    part of the message so that a loop over many files warns once per file
    rather than once per call site under Python's default warning filter.

    Raises:
        ValueError: if the file is malformed (bad atom count, too few atom
            lines, unknown element symbol, non-numeric coords).
    """
    # The comment line is free-form, so the file is decoded as UTF-8 whatever
    # the locale, a byte-order mark is skipped, and undecodable bytes are
    # replaced rather than failing; atom lines with such bytes still fail as
    # unknown symbols.
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file is too short (need at least 2 header lines)")

    count = lines[0].strip()
    if not _COUNT.fullmatch(count):
        raise ValueError(f"first line of an XYZ file must be the atom count; got {count!r}")
    n_declared = int(count)
    if n_declared <= 0:
        raise ValueError(f"XYZ atom count must be positive, got {n_declared}")

    z_list = []
    coords_list = []
    remaining = iter(lines[2:])
    for raw in remaining:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"malformed XYZ atom line: {line!r}")

        sym = parts[0]
        if _ATOMIC_NUMBER.fullmatch(sym):
            z = int(float(sym))
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

    if len(z_list) == n_declared:
        trailing = sum(1 for raw in remaining if raw.strip())
        if trailing:
            warnings.warn(
                f"{path}: XYZ header declares {n_declared} atoms but {trailing} "
                "non-blank line(s) follow them; only the first frame was read. "
                "Check the atom count if the file is not a multi-frame trajectory.",
                UserWarning,
                stacklevel=2,
            )

    if len(z_list) != n_declared:
        raise ValueError(
            f"XYZ header declares {n_declared} atoms but only "
            f"{len(z_list)} valid atom lines were found"
        )

    return np.array(z_list, dtype=int), np.array(coords_list, dtype=float)
