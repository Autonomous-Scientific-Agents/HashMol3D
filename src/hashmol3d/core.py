"""
HashMol3D core: a deterministic identifier for 3D molecular conformers.

The identifier has the form::

    <Hill formula><state tag>-<geometry hash>

For example, ``H2Oq0m1-a1b28135...`` for neutral singlet water.

The trailing hexadecimal hash is invariant under the operations that
leave the non-relativistic molecular Hamiltonian's eigenvalues unchanged:

  * rigid translation of the coordinates
  * rigid rotation of the coordinates
  * permutation (relabeling) of atom indices
  * spatial inversion / reflection (parity)

The hash is built from the full element-labeled distance matrix written
in a canonical atom order (found by Weisfeiler-Leman color refinement
plus an individualization-refinement search). Two geometries receive the
same descriptor if and only if their rounded distance matrices are equal
up to an atom relabeling, i.e. if and only if the geometries are
congruent at the chosen precision. Unlike a plain multiset of pairwise
distances, this is a *complete* invariant: homometric pairs (distinct
geometries sharing a distance multiset) do not collide; see
``docs/design_notes.md``.

It depends on atomic numbers, pairwise distances (rounded to a user
specified precision), and the descriptor version. Total charge and
spin multiplicity are encoded in the readable prefix, not inside the
hash, so changing charge or multiplicity only changes the prefix.

The implementation has no RDKit dependency; it uses only NumPy and the
Python standard library.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .periodic_table import get_symbol

__all__ = [
    "DEFAULT_LENGTH",
    "DESCRIPTOR_VERSION",
    "HashMol3DResult",
    "generate_hashmol3d",
    "hash_length_for",
    "hash_molecule",
]


# The descriptor version is part of the hashed payload. Bump it whenever
# the descriptor format changes in a way that would alter hashes.
DESCRIPTOR_VERSION = "5-CANON-SHA256"

# Default geometry-hash length in hex characters. The hash is a truncated
# SHA-256 digest, and its collision resistance is governed by how many
# distinct geometries share a single namespace -- not by molecule size. For a
# namespace of n distinct geometries hashed into b = 4*length bits, the
# expected number of birthday collisions is ~ n^2 / 2^(b+1). 32 hex chars
# (128 bits) keeps that below ~1 for corpora up to ~10^16 geometries and below
# 1e-9 up to ~10^14; use hash_length_for() to size the hash to a specific
# corpus and target probability. SHA-256 caps us at 64 hex chars (256 bits).
DEFAULT_LENGTH = 32
_MAX_LENGTH = 64

# Cap on the number of partition-refinement states visited by the canonical
# atom-ordering search. The search tree's size is a function of the geometry
# alone (never of the input atom order), so hitting the cap is itself a
# permutation-invariant event; such inputs deterministically fall back to the
# stable-WL multiset descriptor (section tag "W" instead of "C"). Real
# molecules stay far below the cap: the tree has a single node for generic
# geometries and ~|symmetry group| leaves for perfectly symmetric ones
# (e.g. 361 nodes for a 120-atom monoelemental ring at default precision).
_NODE_BUDGET = 10_000

# Scaled distances must stay below 2^62 so they fit int64 with headroom.
_MAX_SCALED = float(2**62)

# Finer precisions overflow ``10.0 ** decimals`` (a double caps out near
# 10^308) before the geometry-dependent ``_MAX_SCALED`` guard can fire, so
# reject them up front with a clear message instead of a raw OverflowError.
_MAX_DECIMALS = 300

# Minimum relative gap between principal-moment eigenvalues for the O(N)
# frame method (normative for the "F" descriptor section). Below this the
# principal axes are degenerate or nearly so (symmetric tops, linear and
# near-linear molecules), the frame reorients under tiny perturbations, and
# hash_molecule(method="frame") warns and falls back to the canonical
# method. Frame noise amplification is bounded by ~1/gap, so 0.05 keeps it
# within the same order as the distance-based paths.
_FRAME_GAP_MIN = 0.05


def hash_length_for(n_items: int, target_prob: float = 1e-9) -> int:
    """Hash length (hex chars) keeping collision risk below ``target_prob``.

    Collision resistance depends on how many distinct geometries share a
    namespace, not on molecule size. Using the birthday approximation
    ``P ~ n^2 / 2^(b+1)`` for ``b`` hash bits, the required length is
    ``ceil((2*log2(n) - log2(target_prob) - 1) / 4)`` hex characters, clamped
    to ``[1, 64]`` (64 hex = full SHA-256, enough for ~1e38 items at 1e-9).

    Args:
        n_items: expected number of distinct geometries in one namespace.
        target_prob: acceptable probability of *any* collision (default 1e-9).

    Returns:
        Recommended hash length in hex characters, in ``[1, 64]``.
    """
    n = int(n_items)
    if not (0.0 < target_prob < 1.0):
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob!r}")
    if n <= 1:
        return 1
    bits = 2.0 * math.log2(n) - math.log2(target_prob) - 1.0
    hexlen = int(math.ceil(bits / 4.0))
    return max(1, min(_MAX_LENGTH, hexlen))


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


def _precision_to_decimals(precision: float) -> tuple[int, float]:
    """Grid resolution implied by a distance precision in Å.

    ``precision`` must be a power of ten no greater than 1 Å, so the
    descriptor's precision field uniquely identifies the integer grid used
    for quantization. Returns ``(decimals, effective_precision)`` where
    ``effective_precision == 10.0 ** (-decimals)`` is the canonical grid
    spacing serialized into the descriptor -- the single source of truth for
    both the hashed ``P:`` field and :attr:`HashMol3DResult.precision`. For
    example, ``1e-4`` -> ``(4, 1e-4)``.

    ``bool`` and ``numpy.bool_`` are rejected: ``bool`` is an ``int``
    subclass and ``numpy.bool_`` coerces to ``1.0`` the same way, so either
    would otherwise be read as ``precision=1.0``. The scalar math uses the
    ``math`` module so validation is dtype-independent -- a ``numpy`` float
    is coerced to a plain ``float`` first rather than taking a value-based
    casting shortcut. The ``1e-6`` relative tolerance absorbs floating-point
    round-off (a float32 spelling of ``1e-4`` is ~3e-8 off), so any float
    representation of a power of ten is accepted and canonicalized to the
    exact grid; inputs farther than that from every power of ten -- the
    ambiguous mid-decade values this contract forbids (``0.05``, ``3.16e-4``,
    ...) included -- are rejected.
    """
    if isinstance(precision, (bool, np.bool_)):
        raise ValueError(f"precision must be a real number, not bool; got {precision!r}")
    precision = float(precision)
    if not math.isfinite(precision) or precision <= 0:
        raise ValueError(f"precision must be a positive finite number, got {precision!r}")
    # ``decimals < 0`` catches powers of ten greater than 1 Å (e.g. 10.0,
    # which is close to 10**-(-1)); ``isclose`` catches everything that is
    # not a power of ten at all. Both share one message.
    decimals = round(-math.log10(precision))
    if decimals < 0 or not math.isclose(precision, 10.0 ** (-decimals), rel_tol=1e-6, abs_tol=0.0):
        raise ValueError(
            "precision must be a power of ten no greater than 1.0 Å "
            f"(1.0, 1e-1, 1e-2, ...); got {precision!r}"
        )
    if decimals > _MAX_DECIMALS:
        raise ValueError(
            f"precision too fine: at most {_MAX_DECIMALS} decimal places are "
            f"supported, got {precision!r}"
        )
    return decimals, 10.0 ** (-decimals)


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
    """Render the readable charge/multiplicity suffix, e.g. ``q1m2``.

    Positive and zero charges are written without a sign (``q0``, ``q1``);
    only negative charges carry a leading ``-`` (``q-1``).
    """
    return f"q{charge}m{multiplicity}"


def _scaled_distances(coords: np.ndarray, decimals: int) -> np.ndarray:
    """Pairwise distances as integer multiples of the precision grid.

    Returns an ``(N, N)`` int64 matrix ``q`` with
    ``q[i, j] = rint(||r_i - r_j|| * 10**decimals)`` and a zero diagonal.
    This is the same binning as rounding each distance to ``decimals``
    places; integers make every later comparison and serialization exact.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    scaled = np.linalg.norm(diff, axis=-1) * (10.0**decimals)
    if scaled.size and float(scaled.max()) >= _MAX_SCALED:
        raise ValueError(
            "precision too fine for this geometry's extent: scaled distances "
            "exceed the exact-integer range; use a coarser precision"
        )
    q = np.rint(scaled).astype(np.int64)
    np.fill_diagonal(q, 0)
    return q


class _SearchBudgetExceeded(Exception):
    """Internal: the canonical-ordering search exceeded its node budget."""


def _refine_partition(colors: np.ndarray, rank_q: np.ndarray, n_ranks: int) -> np.ndarray:
    """Weisfeiler-Leman color refinement to a stable ordered partition.

    ``colors`` are dense ranks (0..k-1). Each round recolors atom ``i`` by
    the pair (own color, sorted multiset of (color_j, distance-rank) over
    all other atoms) and re-ranks lexicographically, so equal geometries
    yield identical colors regardless of the input atom order. Old colors
    are the primary sort key, so cells only ever split.
    """
    n = colors.shape[0]
    n_colors = int(colors.max()) + 1
    while n_colors < n:
        # Composite per-pair key (color_j, rank_q[i, j]); n * n_ranks stays
        # far below 2^63 for any geometry that fits in memory.
        key = colors[None, :] * n_ranks + rank_q
        np.fill_diagonal(key, -1)  # self entry: sorts first, dropped below
        rows = np.sort(key, axis=1)[:, 1:]
        sig = np.concatenate([colors[:, None], rows], axis=1)
        _, inv = np.unique(sig, axis=0, return_inverse=True)
        new_colors = inv.reshape(-1).astype(np.int64)
        new_n = int(new_colors.max()) + 1
        if new_n == n_colors:
            break
        colors, n_colors = new_colors, new_n
    return colors


def _canonical_signature(
    atomic_nums: np.ndarray, qmat: np.ndarray
) -> tuple[str, tuple[int, ...], str]:
    """Canonical geometry signature: ``(section_tag, z_ordered, body)``.

    Canonical path (tag ``"C"``): finds an atom ordering that is a pure
    function of the geometry -- iterated WL refinement, then an
    individualization-refinement search whose leaves are discrete
    orderings, keeping the lexicographically smallest distance matrix --
    and returns the upper triangle of the scaled distance matrix in that
    order. Equal bodies then mean equal labeled distance matrices, so the
    signature is a complete congruence invariant at the chosen precision.

    Fallback path (tag ``"W"``): if the search tree exceeds
    ``_NODE_BUDGET`` states (possible only when rounding makes many atoms
    mutually indistinguishable), returns the stable-WL per-atom signature
    multiset instead. The tree size is permutation-invariant, so the same
    geometry always takes the same path; distinct tags keep the two paths
    from ever colliding with each other.
    """
    n = atomic_nums.shape[0]
    z = atomic_nums.astype(np.int64)
    if n == 1:
        return "C", (int(z[0]),), ""

    # Dense ranks of Z (initial colors) and of the scaled distances.
    _, inv = np.unique(z, return_inverse=True)
    colors0 = inv.reshape(-1).astype(np.int64)
    _, invq = np.unique(qmat, return_inverse=True)
    rank_q = invq.reshape(qmat.shape).astype(np.int64)
    n_ranks = int(rank_q.max()) + 1

    root = _refine_partition(colors0, rank_q, n_ranks)
    iu, ju = np.triu_indices(n, k=1)

    best: bytes | None = None
    best_order: np.ndarray | None = None
    nodes = 0

    def visit(colors: np.ndarray) -> list | None:
        """Count a search node; emit a leaf candidate or a branch frame."""
        nonlocal best, best_order, nodes
        nodes += 1
        if nodes > _NODE_BUDGET:
            raise _SearchBudgetExceeded
        k = int(colors.max()) + 1
        if k == n:  # discrete partition: colors are a full ordering
            order = np.argsort(colors)
            qc = qmat[np.ix_(order, order)]
            # Big-endian bytes of non-negative int64 compare like numbers.
            cand = np.ascontiguousarray(qc[iu, ju], dtype=">i8").tobytes()
            if best is None or cand < best:
                best, best_order = cand, order
            return None
        # Branch on the smallest non-singleton cell (ties: smallest color).
        counts = np.bincount(colors, minlength=k)
        nonsingle = np.flatnonzero(counts > 1)
        target = int(nonsingle[np.argmin(counts[nonsingle])])
        members = np.flatnonzero(colors == target)
        return [colors, members, 0]

    try:
        # Iterative DFS; each frame is [colors, cell members, next index].
        stack: list[list] = []
        frame = visit(root)
        if frame is not None:
            stack.append(frame)
        while stack:
            colors, members, idx = stack[-1]
            if idx >= len(members):
                stack.pop()
                continue
            stack[-1][2] = idx + 1
            # Individualize one cell member: it gets a color sorting just
            # before its former cellmates, then refine.
            child = colors * 2 + 1
            child[members[idx]] -= 1
            _, inv = np.unique(child, return_inverse=True)
            child = _refine_partition(inv.reshape(-1).astype(np.int64), rank_q, n_ranks)
            frame = visit(child)
            if frame is not None:
                stack.append(frame)
    except _SearchBudgetExceeded:
        colors = root
        atoms = []
        for i in range(n):
            row = sorted((int(colors[j]), int(qmat[i, j])) for j in range(n) if j != i)
            atoms.append((int(z[i]), int(colors[i]), tuple(row)))
        atoms.sort()
        body = ";".join(
            f"{zi},{ci}:" + ",".join(f"{cj}-{d}" for cj, d in row) for zi, ci, row in atoms
        )
        return "W", tuple(sorted(int(v) for v in z)), body

    assert best_order is not None
    qc = qmat[np.ix_(best_order, best_order)][iu, ju]
    body = ",".join(map(str, qc.tolist()))
    return "C", tuple(int(v) for v in z[best_order]), body


def _lex_less(a: np.ndarray, b: np.ndarray) -> bool:
    """Row-major lexicographic '<' for equal-shape integer arrays."""
    af, bf = a.ravel(), b.ravel()
    neq = np.flatnonzero(af != bf)
    if neq.size == 0:
        return False
    i = int(neq[0])
    return bool(af[i] < bf[i])


def _frame_signature(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> tuple[str, tuple[int, ...], str] | None:
    """O(N) principal-axes signature, or ``None`` when the frame is unreliable.

    Coordinates are expressed in the eigenbasis of the Z-weighted gyration
    tensor (axes ordered by ascending eigenvalue), quantized to the
    precision grid, and serialized as sorted ``(Z, x, y, z)`` rows. Axis
    *sign* conventions are not needed: all eight sign combinations are
    evaluated and the lexicographically smallest row list wins, which makes
    the signature reflection-invariant by construction. Tensor sums are
    computed over sorted addends so the result is exactly independent of
    the input atom order.

    The frame is reliable only when the eigenvalues are well separated:
    if any relative gap falls below :data:`_FRAME_GAP_MIN` (degenerate or
    nearly degenerate principal moments -- symmetric tops, linear
    molecules), the axes are ill-defined and ``None`` is returned so the
    caller can fall back to the canonical method.
    """
    n = atomic_nums.shape[0]
    z = atomic_nums.astype(np.int64)
    w = z.astype(float)

    def ksum(a: np.ndarray) -> float:
        # Permutation-invariant summation: sort addends first.
        return float(np.sum(np.sort(a)))

    centroid = np.array([ksum(w * coords[:, k]) for k in range(3)]) / ksum(w)
    c = coords - centroid
    t = np.empty((3, 3))
    for a in range(3):
        for b in range(a, 3):
            t[a, b] = t[b, a] = ksum(w * c[:, a] * c[:, b])
    lam, vec = np.linalg.eigh(t)
    if not lam[2] > 0.0:
        return None  # single atom or all atoms coincident
    if float(np.min(np.diff(lam)) / lam[2]) < _FRAME_GAP_MIN:
        return None

    scaled = (c @ vec) * (10.0**decimals)
    if float(np.abs(scaled).max()) >= _MAX_SCALED:
        raise ValueError(
            "precision too fine for this geometry's extent: scaled coordinates "
            "exceed the exact-integer range; use a coarser precision"
        )
    q = np.rint(scaled).astype(np.int64)

    best: np.ndarray | None = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                rows = np.column_stack([z, q * np.array([sx, sy, sz])])
                order = np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
                cand = rows[order]
                if best is None or _lex_less(cand, best):
                    best = cand
    assert best is not None and best.shape == (n, 4)
    body = ";".join(f"{r[0]}:{r[1]},{r[2]},{r[3]}" for r in best.tolist())
    return "F", tuple(int(v) for v in best[:, 0].tolist()), body


def _format_descriptor(
    version: str,
    precision: float,
    z_ordered: tuple[int, ...],
    tag: str,
    body: str,
) -> str:
    """Render the canonical descriptor string that is fed to SHA-256.

    Charge and multiplicity are *not* included: they are part of the
    readable prefix of the final identifier, not of the hashed payload.
    Distances appear as exact integers on the precision grid (their scale
    is fixed by the ``P:`` field).
    """
    prec_str = f"{precision:.1e}"
    z_part = ",".join(str(v) for v in z_ordered)
    return "|".join(
        [
            "V:" + version,
            "P:" + prec_str,
            "Z:" + z_part,
            tag + ":" + body,
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
    method: str = "canonical",
) -> HashMol3DResult:
    """Compute the HashMol3D identifier for a 3D molecular geometry.

    The identifier has the form ``<Hill formula><state tag>-<geom hash>``,
    e.g. ``H2Oq0m1-a1b28135...``. Charge and multiplicity appear in the
    readable prefix; only the geometry contributes to the hash.

    Args:
        atomic_nums: integer array-like of atomic numbers, shape ``(N,)``.
        coords: float array-like of Cartesian coordinates in Å, shape
            ``(N, 3)``.
        precision: distance precision in Å (default ``1e-4``). Must be a
            power of ten no greater than 1 Å (``1.0``, ``1e-1``,
            ``1e-2``, ...); see :func:`_precision_to_decimals`.
        charge: total formal charge (default ``0``).
        multiplicity: spin multiplicity (``1`` = singlet, ``2`` = doublet,
            ...). If ``None``, inferred as singlet/doublet from the
            electron count.
        length: number of hex characters retained from the SHA-256 digest.
            Must be in ``[1, 64]``. If ``None`` (default), uses
            :data:`DEFAULT_LENGTH` (32 hex = 128 bits). Collision resistance
            depends on how many distinct geometries share a namespace, not on
            molecule size; use :func:`hash_length_for` to size the hash to a
            target corpus and probability.
        method: ``"canonical"`` (default) or ``"frame"``. The canonical
            method hashes the labeled distance matrix in a canonical atom
            order (complete, but O(N^2) time and memory). The frame method
            hashes coordinates in the principal-axes frame of the Z-weighted
            gyration tensor -- O(N log N) time and O(N) memory, suited to
            proteins and other large systems -- and is equally complete
            *when the frame is well-defined*. If the principal moments are
            degenerate or nearly so (relative eigenvalue gap below
            :data:`_FRAME_GAP_MIN`; symmetric tops, linear molecules), the
            frame is unreliable: a :class:`UserWarning` is emitted and the
            canonical method is used instead (detectable via the ``C:`` or
            ``W:`` section in ``result.descriptor`` instead of ``F:``).
            Identifiers from different methods are **not comparable**; pick
            one method per corpus.

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
        length = DEFAULT_LENGTH
    else:
        # Accept any integer type (incl. NumPy integers) but not bool, which
        # is an int subclass and would silently truncate the hash.
        if isinstance(length, bool) or not isinstance(length, numbers.Integral):
            raise ValueError("length must be an int in [1, 64]")
        length = int(length)
        if not (1 <= length <= 64):
            raise ValueError("length must be an int in [1, 64]")

    if method not in ("canonical", "frame"):
        raise ValueError(f"method must be 'canonical' or 'frame', got {method!r}")

    charge = int(charge)
    # ``precision`` is validated (power of ten <= 1 Å, no bool) and
    # canonicalized in one place; ``effective_precision`` is what gets hashed
    # and reported, so the descriptor grid can never drift from the value.
    decimals, effective_precision = _precision_to_decimals(precision)
    used_mult = _infer_multiplicity(atomic_nums, charge, multiplicity)

    signature = None
    if method == "frame":
        signature = _frame_signature(atomic_nums, coords, decimals)
        if signature is None:
            warnings.warn(
                "principal moments are degenerate or nearly degenerate "
                f"(relative eigenvalue gap < {_FRAME_GAP_MIN}); the inertia "
                "frame is unreliable for this geometry, falling back to "
                "method='canonical'. The returned hash is a canonical-method "
                "hash and will not match frame-method hashes.",
                UserWarning,
                stacklevel=2,
            )
    if signature is None:
        qmat = _scaled_distances(coords, decimals)
        signature = _canonical_signature(atomic_nums, qmat)
    tag, z_ordered, body = signature
    descriptor = _format_descriptor(DESCRIPTOR_VERSION, effective_precision, z_ordered, tag, body)
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:length]

    formula = _hill_formula(atomic_nums)
    identifier = f"{formula}{_state_tag(charge, used_mult)}-{digest}"

    return HashMol3DResult(
        hash_str=identifier,
        formula=formula,
        geometry_hash=digest,
        version=DESCRIPTOR_VERSION,
        precision=effective_precision,
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
