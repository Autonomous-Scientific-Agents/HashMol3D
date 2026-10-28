"""HashMol3D: deterministic 3D molecular geometry hashing."""

from .core import (
    DEFAULT_LENGTH,
    DESCRIPTOR_VERSION,
    HashMol3DResult,
    SearchBudgetExceeded,
    descriptor_geometry,
    generate_hashmol3d,
    hash_descriptor,
    hash_length_for,
    hash_molecule,
)
from .io import read_xyz
from .rdkit_support import canonical_smiles, hash_file, hash_rdkit, read_rdkit
from .version import __version__


def hash_xyz(path, **kwargs) -> HashMol3DResult:
    """Read an XYZ file and hash its geometry.

    Convenience wrapper for ``hash_file(path, input_format="xyz")``; the
    keyword arguments are forwarded to :func:`hash_molecule`.
    """
    return hash_file(path, input_format="xyz", **kwargs)


__all__ = [
    "DEFAULT_LENGTH",
    "DESCRIPTOR_VERSION",
    "HashMol3DResult",
    "SearchBudgetExceeded",
    "__version__",
    "canonical_smiles",
    "descriptor_geometry",
    "generate_hashmol3d",
    "hash_descriptor",
    "hash_length_for",
    "hash_file",
    "hash_molecule",
    "hash_rdkit",
    "hash_xyz",
    "read_xyz",
    "read_rdkit",
]
