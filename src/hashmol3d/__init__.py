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
from .rdkit_support import canonical_smiles, hash_file, hash_rdkit, read_rdkit
from .version import __version__


def hash_xyz(path, *, include_smiles=False, **kwargs) -> HashMol3DResult:
    """Read an XYZ file and hash its geometry.

    Convenience wrapper combining :func:`read_xyz` and
    :func:`hash_molecule`. Any keyword arguments are forwarded to
    :func:`hash_molecule`. ``include_smiles=True`` opts into RDKit bond
    perception and the separately versioned S-tag descriptor.
    """
    return hash_file(path, input_format="xyz", include_smiles=include_smiles, **kwargs)


__all__ = [
    "DEFAULT_LENGTH",
    "DESCRIPTOR_VERSION",
    "HashMol3DResult",
    "SearchBudgetExceeded",
    "__version__",
    "canonical_smiles",
    "generate_hashmol3d",
    "hash_length_for",
    "hash_file",
    "hash_molecule",
    "hash_rdkit",
    "hash_xyz",
    "read_xyz",
    "read_rdkit",
]
