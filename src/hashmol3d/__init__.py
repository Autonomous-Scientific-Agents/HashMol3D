"""HashMol3D: deterministic 3D molecular geometry hashing."""

from .core import (
    DEFAULT_LENGTH,
    DESCRIPTOR_VERSION,
    HashMol3DResult,
    SearchBudgetExceeded,
    generate_hashmol3d,
    hash_length_for,
    hash_molecule,
)
from .io import read_xyz
from .version import __version__


def hash_xyz(path, **kwargs) -> HashMol3DResult:
    """Read an XYZ file and hash its geometry.

    Convenience wrapper combining :func:`read_xyz` and
    :func:`hash_molecule`. Any keyword arguments are forwarded to
    :func:`hash_molecule`.
    """
    atomic_nums, coords = read_xyz(path)
    return hash_molecule(atomic_nums, coords, **kwargs)


__all__ = [
    "DEFAULT_LENGTH",
    "DESCRIPTOR_VERSION",
    "HashMol3DResult",
    "SearchBudgetExceeded",
    "__version__",
    "generate_hashmol3d",
    "hash_length_for",
    "hash_molecule",
    "hash_xyz",
    "read_xyz",
]
