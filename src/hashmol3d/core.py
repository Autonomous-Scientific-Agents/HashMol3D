"""
HashMol3D core: a deterministic identifier for 3D molecular conformers.

The identifier has the form::

    <Hill formula><state tag>-<geometry hash>

For example, ``H2Oq0m1-a1b28135...`` for neutral singlet water.

The trailing hexadecimal hash is invariant under exactly the operations
that leave the non-relativistic molecular Hamiltonian's eigenvalues
unchanged:

  * rigid translation of the coordinates
  * rigid rotation of the coordinates
  * permutation (relabeling) of atom indices
  * spatial inversion / reflection (parity)

It depends on atomic numbers, pairwise distances (rounded to a user
specified precision), and the descriptor version. Total charge and
spin multiplicity are encoded in the readable prefix, not inside the
hash, so changing charge or multiplicity only changes the prefix.

The implementation has no RDKit dependency; it uses only NumPy and the
Python standard library.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .periodic_table import get_symbol

__all__ = [
    "DESCRIPTOR_VERSION",
    "HashMol3DResult",
    "generate_hashmol3d",
    "hash_molecule",
]


# The descriptor version is part of the hashed payload. Bump it whenever
# the descriptor format changes in a way that would alter hashes.
DESCRIPTOR_VERSION = "4-GEOM-SHA256"

# Auto-scaled hash length. The number of distinguishable conformers grows
# (roughly) exponentially with N, so log2 of it grows linearly with N;
# growing the hash length linearly with N keeps birthday-collision risk
# constant. 16 hex chars (64 bits) is the floor for very small molecules;
# SHA-256 caps us at 64 hex chars (256 bits).
_MIN_LENGTH = 16
_MAX_LENGTH = 64


@dataclass(frozen=True)
class HashMol3DResult:
    """The result of hashing a molecular geometry."""

    hash_str: str
    formula: str
    geometry_hash: str
    version: str
    precision: float
    charge: int
    multiplicity: int
    descriptor: str

    def __str__(self) -> str:
        return self.hash_str


def _precision_to_decimals(precision: float) -> int:
    """Number of decimal places implied by a distance precision in Å."""
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError(f"precision must be a positive finite number, got {precision!r}")
    return int(max(0, round(-np.log10(precision))))


def _infer_multiplicity(atomic_nums: np.ndarray, charge: int, multiplicity: int | None) -> int:
    """Use the caller-supplied multiplicity, or infer one from electron count."""
    if multiplicity is not None:
        m = int(multiplicity)
        if m < 1:
            raise ValueError(f"multiplicity must be >= 1, got {m}")
        return m
    electrons = int(np.sum(atomic_nums)) - int(charge)
    return 1 if electrons % 2 == 0 else 2


def _hill_formula(atomic_nums: np.ndarray) -> str:
    """Render the molecular formula in Hill order.

    Carbon first (if present), then hydrogen (if present), then the
    remaining elements alphabetically by symbol. A count of 1 is
    omitted (e.g. ``H2O``, ``CHBrClF``).
    """
    counts: Counter = Counter(int(z) for z in atomic_nums)

    ordered: list[tuple[str, int]] = []
    if 6 in counts:
        ordered.append(("C", counts.pop(6)))
        if 1 in counts:
            ordered.append(("H", counts.pop(1)))
    rest = sorted(((get_symbol(z), n) for z, n in counts.items()), key=lambda x: x[0])
    ordered.extend(rest)

    return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in ordered)


def _state_tag(charge: int, multiplicity: int) -> str:
    """Render the readable charge/multiplicity suffix, e.g. ``q+1m2``."""
    if charge == 0:
        q_part = "q0"
    else:
        q_part = f"q{'+' if charge > 0 else '-'}{abs(charge)}"
    return f"{q_part}m{multiplicity}"


def _auto_length(n_atoms: int) -> int:
    """Default hash length in hex chars, scaling linearly with N."""
    return max(_MIN_LENGTH, min(_MAX_LENGTH, n_atoms))


def _pair_signature(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> tuple[tuple[int, ...], list[tuple[int, int, float]]]:
    """Build the permutation-invariant fingerprint of the geometry.

    Returns ``(z_sorted, pairs)`` where ``z_sorted`` is a sorted tuple of
    atomic numbers and ``pairs`` is a sorted list of
    ``(Z_min, Z_max, rounded_distance)`` triples over every unordered pair
    of atoms. Both objects are invariant under any relabeling of atoms
    (multisets) and under any rigid motion or reflection (functions only
    of Z and pairwise distances).
    """
    n = atomic_nums.shape[0]
    z_sorted = tuple(sorted(int(z) for z in atomic_nums))

    if n < 2:
        return z_sorted, []

    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    dvals = np.round(dmat[iu, ju], decimals=decimals)

    z_i = atomic_nums[iu].astype(int)
    z_j = atomic_nums[ju].astype(int)
    za = np.minimum(z_i, z_j)
    zb = np.maximum(z_i, z_j)

    pairs = [(int(a), int(b), float(d)) for a, b, d in zip(za, zb, dvals)]
    pairs.sort()
    return z_sorted, pairs


def _format_descriptor(
    version: str,
    precision: float,
    decimals: int,
    z_sorted: tuple[int, ...],
    pairs: list[tuple[int, int, float]],
) -> str:
    """Render the canonical descriptor string that is fed to SHA-256.

    Charge and multiplicity are *not* included: they are part of the
    readable prefix of the final identifier, not of the hashed payload.
    """
    prec_str = f"{precision:.1e}"
    z_part = ",".join(str(z) for z in z_sorted)
    fmt = f"{{:.{decimals}f}}"
    d_part = ",".join(f"{a}-{b}:{fmt.format(d)}" for a, b, d in pairs)
    return "|".join(
        [
            "V:" + version,
            "P:" + prec_str,
            "Z:" + z_part,
            "D:" + d_part,
        ]
    )


def hash_molecule(
    atomic_nums,
    coords,
    *,
    precision: float = 1e-4,
    charge: int = 0,
    multiplicity: int | None = None,
    length: int | None = None,
) -> HashMol3DResult:
    """Compute the HashMol3D identifier for a 3D molecular geometry.

    The identifier has the form ``<Hill formula><state tag>-<geom hash>``,
    e.g. ``H2Oq0m1-a1b28135...``. Charge and multiplicity appear in the
    readable prefix; only the geometry contributes to the hash.

    Args:
        atomic_nums: integer array-like of atomic numbers, shape ``(N,)``.
        coords: float array-like of Cartesian coordinates in Å, shape
            ``(N, 3)``.
        precision: distance precision in Å (default ``1e-4``).
        charge: total formal charge (default ``0``).
        multiplicity: spin multiplicity (``1`` = singlet, ``2`` = doublet,
            ...). If ``None``, inferred as singlet/doublet from the
            electron count.
        length: number of hex characters retained from the SHA-256 digest.
            Must be in ``[1, 64]``. If ``None`` (default), scales with the
            number of atoms as ``clip(N, 16, 64)`` so collision risk stays
            roughly constant as molecules grow.

    Returns:
        :class:`HashMol3DResult`.
    """
    atomic_nums = np.asarray(atomic_nums, dtype=int).reshape(-1)
    coords = np.asarray(coords, dtype=float)

    if atomic_nums.size == 0:
        raise ValueError("molecule must contain at least one atom")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")
    if coords.shape[0] != atomic_nums.size:
        raise ValueError(
            f"atomic_nums has {atomic_nums.size} entries but coords has {coords.shape[0]} rows"
        )
    if not np.all(atomic_nums > 0):
        raise ValueError("atomic numbers must be positive integers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contain non-finite values")

    if length is None:
        length = _auto_length(int(atomic_nums.size))
    elif not isinstance(length, int) or not (1 <= length <= 64):
        raise ValueError("length must be an int in [1, 64]")

    charge = int(charge)
    decimals = _precision_to_decimals(precision)
    used_mult = _infer_multiplicity(atomic_nums, charge, multiplicity)

    z_sorted, pairs = _pair_signature(atomic_nums, coords, decimals)
    descriptor = _format_descriptor(DESCRIPTOR_VERSION, precision, decimals, z_sorted, pairs)
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:length]

    formula = _hill_formula(atomic_nums)
    identifier = f"{formula}{_state_tag(charge, used_mult)}-{digest}"

    return HashMol3DResult(
        hash_str=identifier,
        formula=formula,
        geometry_hash=digest,
        version=DESCRIPTOR_VERSION,
        precision=precision,
        charge=charge,
        multiplicity=used_mult,
        descriptor=descriptor,
    )


def generate_hashmol3d(
    atomic_nums,
    coords,
    precision: float = 1e-4,
    charge: int = 0,
    multiplicity: int | None = None,
    hash_length: int = 32,
) -> HashMol3DResult:
    """Deprecated alias for :func:`hash_molecule`.

    .. deprecated:: 0.4.0
        Use :func:`hash_molecule` instead. The ``hash_length`` keyword is
        renamed to ``length`` in the new function. Note that as of 0.5.0
        the returned ``hash_str`` is the full readable identifier
        (``<formula><state>-<hash>``), not just the hex digest.
    """
    warnings.warn(
        "generate_hashmol3d() is deprecated; use hash_molecule() instead "
        "(the hash_length kwarg is now called length).",
        DeprecationWarning,
        stacklevel=2,
    )
    return hash_molecule(
        atomic_nums,
        coords,
        precision=precision,
        charge=charge,
        multiplicity=multiplicity,
        length=hash_length,
    )
