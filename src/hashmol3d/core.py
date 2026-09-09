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

By default the hash is built from element-labelled coordinates in a
deterministic canonical frame. Principal axes handle ordinary geometries;
intrinsic point/line representations and canonical atom anchors resolve
degenerate eigenspaces without random perturbations. The full canonical
labelled distance matrix remains available with ``method="canonical"``.
The normal ``F`` and ``C`` paths are complete representations at their
quantization grids; unlike a plain multiset of pairwise distances, homometric
pairs do not collide. Exhausting the canonical search budget raises
``SearchBudgetExceeded`` without creating a descriptor or hash.

The descriptor depends on atomic numbers, geometry quantized at a user
specified precision, and the descriptor version. Total charge and spin
multiplicity are encoded in the readable prefix, not inside the hash, so
changing charge or multiplicity only changes the prefix.

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
    "SearchBudgetExceeded",
    "generate_hashmol3d",
    "hash_length_for",
    "hash_molecule",
]


# The descriptor version is part of the hashed payload. Bump it whenever
# the descriptor format changes in a way that would alter hashes.
DESCRIPTOR_VERSION = "7-FRAME-SHA256"

# Default geometry-hash length in hex characters. The hash is a truncated
# SHA-256 digest, and its collision resistance is governed by how many
# distinct geometries share a single namespace -- not by molecule size. For a
# namespace of n distinct geometries hashed into b = 4*length bits, the
# expected number of birthday collisions is ~ n^2 / 2^(b+1). 32 hex chars
# (128 bits) keeps that below ~1 for corpora up to ~2.6e19 geometries and below
# 1e-9 up to ~8.2e14; use hash_length_for() to size the hash to a specific
# corpus and target probability. SHA-256 caps us at 64 hex chars (256 bits).
DEFAULT_LENGTH = 32
_MAX_LENGTH = 64

# Highest atomic number the periodic table recognizes (H..Og); inputs outside
# ``[1, _MAX_Z]`` are rejected up front rather than surfacing later as a
# symbol-lookup KeyError.
_MAX_Z = 118

# Default cap on partition-refinement states in canonical atom ordering.
# Exhaustion produces an error, never a partial or weaker descriptor. Callers
# can increase node_budget without changing any successfully completed hash.
_NODE_BUDGET = 10_000

# Scaled distances must stay below 2^62 so they fit int64 with headroom.
_MAX_SCALED = float(2**62)

# Finer precisions overflow ``10.0 ** decimals`` (a double caps out near
# 10^308) before the geometry-dependent ``_MAX_SCALED`` guard can fire, so
# reject them up front with a clear message instead of a raw OverflowError.
_MAX_DECIMALS = 300

# Minimum relative gap between principal-moment eigenvalues for the fast
# principal-axes branch of the frame method. Below this, only the isolated
# eigenspaces are retained and atoms canonically anchor the ambiguous axes.
# Frame noise amplification is bounded by ~1/gap, so 0.05 keeps it within
# the same order as the distance-based paths.
_FRAME_GAP_MIN = 0.05

# An atom-derived axis shorter than ten coordinate-grid units is too easily
# reoriented by sub-precision noise. Such marginal geometries use the exact
# distance fallback instead. Exact point-like and linear geometries are
# handled separately and therefore do not need artificial transverse axes.
_FRAME_ANCHOR_MIN_GRID = 10.0

# Relative residual accepted as collinear up to float64 roundoff. This is
# independent of the requested grid: it must not flatten a resolved bend.
_FRAME_LINEAR_REL_TOL = 64.0 * np.finfo(np.float64).eps

# Canonically tied anchors must all be evaluated. Bound that work so an
# adversarial highly symmetric geometry cannot turn the frame method into an
# unbounded candidate search; exceeding the budget deterministically selects
# the canonical distance fallback.
_FRAME_CANDIDATE_BUDGET = 10_000


def hash_length_for(n_items: int, target_prob: float = 1e-9) -> int:
    """Hash length (hex chars) keeping collision risk below ``target_prob``.

    Collision resistance depends on how many distinct geometries share a
    namespace, not on molecule size. Using the birthday approximation
    ``P ~ n^2 / 2^(b+1)`` for ``b`` hash bits, the required length is
    ``ceil((2*log2(n) - log2(target_prob) - 1) / 4)`` hex characters, clamped
    to ``[1, 64]`` (64 hex = full SHA-256, enough for ~1.5e34 items at 1e-9).

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
    if hexlen > _MAX_LENGTH:
        # The birthday approximation ``P ~ n^2 / 2^(b+1)`` overestimates the
        # collision probability once it leaves its small-probability regime, so
        # a length estimate above the digest size does not prove the true
        # probability at 256 bits exceeds ``target_prob``. Report the clamp
        # without asserting a shortfall.
        warnings.warn(
            f"the birthday approximation requests {hexlen} hex characters for "
            f"a collision probability of {target_prob!r} at {n} items, more "
            f"than the {_MAX_LENGTH}-character SHA-256 digest; returning the "
            f"full SHA-256 digest ({_MAX_LENGTH} hex characters).",
            UserWarning,
            stacklevel=2,
        )
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
    """Grid resolution implied by a geometry precision in Å.

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


def _as_exact_int(value, name: str) -> int:
    """Coerce an integer-valued scalar to ``int`` without silent truncation.

    Accepts Python and NumPy integers and integer-valued floats (``3.0``) but
    rejects booleans and genuinely fractional values (``0.5``), which
    ``int(...)`` would otherwise truncate toward zero and hash as a different
    state. This keeps the state prefix a faithful record of the input.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool; got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f) or f != round(f):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(f)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def _validate_atomic_nums(atomic_nums) -> np.ndarray:
    """Validate and normalize atomic numbers to an ``int64`` array.

    Rejects booleans, complex values, non-finite and fractional values
    (``6.9``), and values outside ``[1, _MAX_Z]`` *before* coercing to ``int``,
    so a fractional or out-of-range input fails with a clear domain error rather
    than being silently floored (``6.9`` -> carbon) or surfacing later as a
    symbol-lookup ``KeyError`` (``119``).

    Booleans and complex numbers are rejected against the *original* elements,
    not the coerced array: ``np.asarray([6, True])`` collapses to a plain
    integer array that hides the boolean, and ``np.asarray([6 + 1j])`` would
    later discard the imaginary part. An object-dtype view preserves the
    caller's Python scalars so these masked cases are caught. An already-coerced
    numeric array cannot reveal a boolean the caller may have supplied, so that
    scan is limited to Python sequences and object arrays, which retain the
    information.
    """
    arr = np.asarray(atomic_nums)
    if arr.dtype == bool:
        raise ValueError("atomic numbers must be integers, not booleans")
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError("atomic numbers must be real integers, not complex numbers")
    if not isinstance(atomic_nums, np.ndarray) or arr.dtype == object:
        for x in np.asarray(atomic_nums, dtype=object).reshape(-1):
            if isinstance(x, (bool, np.bool_)):
                raise ValueError("atomic numbers must be integers, not booleans")
            if isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real):
                raise ValueError("atomic numbers must be real integers, not complex numbers")
    try:
        as_float = arr.astype(float).reshape(-1)
    except (TypeError, ValueError) as err:
        raise ValueError(f"atomic numbers must be integers in [1, {_MAX_Z}]") from err
    if as_float.size == 0:
        return as_float.astype(np.int64)
    if not np.all(np.isfinite(as_float)):
        raise ValueError("atomic numbers must be finite integers")
    if not np.all(as_float == np.round(as_float)):
        raise ValueError("atomic numbers must be integers, not fractional values")
    if not np.all((as_float >= 1) & (as_float <= _MAX_Z)):
        raise ValueError(f"atomic numbers must be in [1, {_MAX_Z}]")
    return as_float.astype(np.int64)


def _infer_multiplicity(atomic_nums: np.ndarray, charge: int, multiplicity: int | None) -> int:
    """Use the caller-supplied multiplicity, or infer one from electron count."""
    if multiplicity is not None:
        m = _as_exact_int(multiplicity, "multiplicity")
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


class SearchBudgetExceeded(RuntimeError):
    """Canonical search exhausted its node budget; no identifier was created."""


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
    atomic_nums: np.ndarray, qmat: np.ndarray, node_budget: int = _NODE_BUDGET
) -> tuple[str, tuple[int, ...], str]:
    """Canonical geometry signature: ``(section_tag, z_ordered, body)``.

    Canonical path (tag ``"C"``): finds an atom ordering that is a pure
    function of the geometry -- iterated WL refinement, then an
    individualization-refinement search whose leaves are discrete
    orderings, keeping the lexicographically smallest distance matrix --
    and returns the upper triangle of the scaled distance matrix in that
    order. Equal bodies then mean equal labeled distance matrices, so the
    signature is a complete congruence invariant at the chosen precision.

    The complete search must finish within ``node_budget`` visited states.
    Otherwise raise ``SearchBudgetExceeded``; partial candidates are discarded.
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
        if nodes > node_budget:
            raise SearchBudgetExceeded(
                f"canonical search exceeded the node budget ({node_budget}); "
                "no identifier or hash was created. Increase node_budget "
                "(CLI: --node-budget) and retry."
            )
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


def _sorted_signed_rows(
    z: np.ndarray, q: np.ndarray, sign_choices: tuple[tuple[int, int, int], ...]
) -> np.ndarray:
    """Smallest sorted ``(Z, x, y, z)`` rows over ``sign_choices``."""
    best: np.ndarray | None = None
    for signs in sign_choices:
        rows = np.column_stack([z, q * np.asarray(signs, dtype=np.int64)])
        order = np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
        cand = rows[order]
        if best is None or _lex_less(cand, best):
            best = cand
    assert best is not None
    return best


_ALL_AXIS_SIGNS = tuple((sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1))


def _rows_in_basis(
    z: np.ndarray, centered: np.ndarray, basis: np.ndarray, scale: float
) -> np.ndarray:
    """Quantize coordinates in an orthonormal basis and resolve axis signs."""
    scaled = (centered @ basis) * scale
    if scaled.size and float(np.abs(scaled).max()) >= _MAX_SCALED:
        raise ValueError(
            "precision too fine for this geometry's extent: scaled coordinates "
            "exceed the exact-integer range; use a coarser precision"
        )
    q = np.rint(scaled).astype(np.int64)
    return _sorted_signed_rows(z, q, _ALL_AXIS_SIGNS)


def _frame_body(best: np.ndarray) -> tuple[str, tuple[int, ...], str]:
    """Serialize a winning frame row array as an ``F`` signature."""
    body = ";".join(f"{r[0]}:{r[1]},{r[2]},{r[3]}" for r in best.tolist())
    return "F", tuple(int(v) for v in best[:, 0].tolist()), body


def _frame_signature(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> tuple[str, tuple[int, ...], str] | None:
    """Canonical frame signature, or ``None`` for the distance fallback.

    Coordinates are expressed in the eigenbasis of the Z-weighted gyration
    tensor (axes ordered by ascending eigenvalue), quantized to the
    precision grid, and serialized as sorted ``(Z, x, y, z)`` rows. Axis
    *sign* conventions are not needed: all eight sign combinations are
    evaluated and the lexicographically smallest row list wins, which makes
    the signature reflection-invariant by construction. Tensor sums are
    computed over sorted addends so the result is exactly independent of
    the input atom order.

    Well-separated eigenvalues take the O(N log N) principal-axes path. If
    exactly one eigenspace is isolated, that axis is retained and a centered
    atom vector canonically anchors the degenerate plane. If all moments are
    degenerate, a canonical ordered pair of non-collinear atom vectors
    supplies the frame. Exact point-like and linear inputs are represented in
    their intrinsic zero- or one-dimensional coordinates instead of inventing
    meaningless axes.

    Anchor keys use only rotation-, reflection-, and permutation-invariant
    integers on the precision grid. Every tied candidate is evaluated and the
    lexicographically smallest row list wins. ``None`` is reserved for
    ill-conditioned anchors or a candidate-budget overflow; the caller then
    uses the complete canonical distance descriptor.
    """
    n = atomic_nums.shape[0]
    z = atomic_nums.astype(np.int64)
    w = z.astype(float)
    scale = 10.0**decimals

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
    radii = np.linalg.norm(c, axis=1)
    max_radius_grid = float(radii.max()) * scale
    if max_radius_grid >= _MAX_SCALED:
        raise ValueError(
            "precision too fine for this geometry's extent: scaled coordinates "
            "exceed the exact-integer range; use a coarser precision"
        )

    # At the requested grid all atoms occupy the centroid. No orientation is
    # meaningful, and every coordinate necessarily rounds to zero.
    if max_radius_grid < 0.5:
        q = np.zeros((n, 3), dtype=np.int64)
        return _frame_body(_sorted_signed_rows(z, q, ((1, 1, 1),)))

    def line_rows(axis: np.ndarray) -> np.ndarray:
        q = np.zeros((n, 3), dtype=np.int64)
        scaled_axial = (c @ axis) * scale
        if scaled_axial.size and float(np.abs(scaled_axial).max()) >= _MAX_SCALED:
            raise ValueError(
                "precision too fine for this geometry's extent: scaled coordinates "
                "exceed the exact-integer range; use a coarser precision"
            )
        # The nonzero intrinsic coordinate occupies the largest-moment slot,
        # matching ascending eigenvalue order for an exact line.
        q[:, 2] = np.rint(scaled_axial).astype(np.int64)
        return _sorted_signed_rows(z, q, ((1, 1, 1), (1, 1, -1)))

    if not lam[2] > 0.0:
        return None

    # A true line needs only its isolated longitudinal axis, even when its
    # extent is too small for the general atom-anchor rule. Recognize it up
    # to relative float64 roundoff, not by a grid-dependent bend tolerance.
    # Larger systems retain the existing intrinsic-line/anchor decisions.
    if max_radius_grid < _FRAME_ANCHOR_MIN_GRID:
        axis = vec[:, 2]
        projected = c - np.outer(c @ axis, axis)
        max_projected = float(np.linalg.norm(projected, axis=1).max())
        if max_projected <= _FRAME_LINEAR_REL_TOL * float(radii.max()):
            return _frame_body(line_rows(axis))
        return None

    gap0 = float((lam[1] - lam[0]) / lam[2])
    gap1 = float((lam[2] - lam[1]) / lam[2])
    if min(gap0, gap1) >= _FRAME_GAP_MIN:
        return _frame_body(_rows_in_basis(z, c, vec, scale))

    def quantized(values: np.ndarray) -> np.ndarray:
        scaled = values * scale
        if scaled.size and float(np.abs(scaled).max()) >= _MAX_SCALED:
            raise ValueError(
                "precision too fine for this geometry's extent: scaled coordinates "
                "exceed the exact-integer range; use a coarser precision"
            )
        return np.rint(scaled).astype(np.int64)

    best: np.ndarray | None = None
    candidates = 0

    def consider(basis: np.ndarray) -> bool:
        """Evaluate one basis; return false when the budget is exhausted."""
        nonlocal best, candidates
        candidates += 1
        if candidates > _FRAME_CANDIDATE_BUDGET:
            return False
        cand = _rows_in_basis(z, c, basis, scale)
        if best is None or _lex_less(cand, best):
            best = cand
        return True

    # One isolated eigenvalue leaves only a two-dimensional plane to anchor.
    # Preserve eigenvalue-axis ordering so this branch approaches the regular
    # principal frame continuously away from the degeneracy.
    if (gap0 < _FRAME_GAP_MIN) != (gap1 < _FRAME_GAP_MIN):
        unique_slot = 2 if gap0 < _FRAME_GAP_MIN else 0
        axis = vec[:, unique_slot]
        axial = c @ axis
        projected = c - np.outer(axial, axis)
        projected_norm = np.linalg.norm(projected, axis=1)
        max_projected_grid = float(projected_norm.max()) * scale

        # In the lower-pair-degenerate case an exact line has no transverse
        # information to resolve. Small numerical residuals must not invent an
        # orientation in its null plane.
        if unique_slot == 2 and max_projected_grid < 0.5:
            return _frame_body(line_rows(axis))
        if max_projected_grid < _FRAME_ANCHOR_MIN_GRID:
            return None

        q_projected = quantized(projected_norm)
        q_axial = quantized(np.abs(axial))
        keys = [(int(q_projected[i]), int(z[i]), int(q_axial[i])) for i in range(n)]
        winning_key = max(keys)
        anchors = [i for i, key in enumerate(keys) if key == winning_key]
        if len(anchors) > _FRAME_CANDIDATE_BUDGET:
            return None

        for i in anchors:
            plane_axis = projected[i] / projected_norm[i]
            cross_axis = np.cross(axis, plane_axis)
            if unique_slot == 0:
                basis = np.column_stack([axis, plane_axis, cross_axis])
            else:
                basis = np.column_stack([plane_axis, cross_axis, axis])
            if not consider(basis):
                return None
        assert best is not None
        return _frame_body(best)

    # All three moments are near-degenerate. A far, heavy atom supplies the
    # first axis; a maximally non-collinear second atom supplies the plane.
    q_radii = quantized(radii)
    first_keys = [(int(q_radii[i]), int(z[i])) for i in range(n)]
    winning_first_key = max(first_keys)
    first_anchors = [i for i, key in enumerate(first_keys) if key == winning_first_key]
    if len(first_anchors) > _FRAME_CANDIDATE_BUDGET:
        return None

    for i in first_anchors:
        first_axis = c[i] / radii[i]
        along = c @ first_axis
        projected = c - np.outer(along, first_axis)
        projected_norm = np.linalg.norm(projected, axis=1)
        max_projected_grid = float(projected_norm.max()) * scale
        if max_projected_grid < _FRAME_ANCHOR_MIN_GRID:
            # This can occur only for a marginally non-point cloud; a true
            # line would have an isolated largest eigenvalue above.
            return None
        q_projected = quantized(projected_norm)
        q_along = quantized(np.abs(along))
        second_keys = [
            (int(q_projected[j]), int(z[j]), int(q_along[j]))
            for j in range(n)
            if j != i and q_projected[j] > 0
        ]
        if not second_keys:
            return None
        winning_second_key = max(second_keys)
        second_anchors = [
            j
            for j in range(n)
            if j != i
            and q_projected[j] > 0
            and (int(q_projected[j]), int(z[j]), int(q_along[j])) == winning_second_key
        ]
        if candidates + len(second_anchors) > _FRAME_CANDIDATE_BUDGET:
            return None
        for j in second_anchors:
            second_axis = projected[j] / projected_norm[j]
            third_axis = np.cross(first_axis, second_axis)
            basis = np.column_stack([first_axis, second_axis, third_axis])
            if not consider(basis):
                return None

    assert best is not None
    return _frame_body(best)


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
    method: str = "frame",
    node_budget: int = _NODE_BUDGET,
) -> HashMol3DResult:
    """Compute the HashMol3D identifier for a 3D molecular geometry.

    The identifier has the form ``<Hill formula><state tag>-<geom hash>``,
    e.g. ``H2Oq0m1-a1b28135...``. Charge and multiplicity appear in the
    readable prefix; only the geometry contributes to the hash.

    Args:
        atomic_nums: integer array-like of atomic numbers, shape ``(N,)``.
        coords: float array-like of Cartesian coordinates in Å, shape
            ``(N, 3)``.
        precision: geometry-grid precision in Å (default ``1e-4``). Must be a
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
        method: ``"frame"`` (default) or ``"canonical"``. The canonical
            method hashes the labeled distance matrix in a canonical atom
            order (complete, but O(N^2) memory). The frame method
            hashes coordinates in the principal-axes frame of the Z-weighted
            gyration tensor -- normally O(N log N) time and O(N) memory,
            suited to proteins and other large systems. Degenerate principal
            moments are resolved by intrinsic point/line descriptors or by
            canonical atom anchors. Ill-conditioned anchors and candidate-
            budget overflows emit :class:`UserWarning` and use the canonical
            method instead (detectable via the ``C:`` section in
            ``result.descriptor`` instead of ``F:``).
            Identifiers from different methods are **not comparable**; pick
            one method per corpus.
        node_budget: positive integer cap on canonical search states (default
            10,000), also used after frame fallback. Increasing it changes
            whether a search can finish, not its completed descriptor.

    Raises:
        SearchBudgetExceeded: if canonical search cannot finish within the
            node budget. No descriptor, hash, or result is created; increase
            ``node_budget`` and retry.
        ValueError: if input or an option is invalid.

    Returns:
        :class:`HashMol3DResult`.
    """
    atomic_nums = _validate_atomic_nums(atomic_nums)
    coords = np.asarray(coords, dtype=float)

    if atomic_nums.size == 0:
        raise ValueError("molecule must contain at least one atom")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")
    if coords.shape[0] != atomic_nums.size:
        raise ValueError(
            f"atomic_nums has {atomic_nums.size} entries but coords has {coords.shape[0]} rows"
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contain non-finite values")

    if length is None:
        length = DEFAULT_LENGTH
    else:
        # Accept any integer type (incl. NumPy integers) but not bool, which
        # is an int subclass and would silently truncate the hash.
        if isinstance(length, bool) or not isinstance(length, numbers.Integral):
            raise ValueError(f"length must be an int in [1, {_MAX_LENGTH}]")
        length = int(length)
        if not (1 <= length <= _MAX_LENGTH):
            raise ValueError(f"length must be an int in [1, {_MAX_LENGTH}]")

    if method not in ("canonical", "frame"):
        raise ValueError(f"method must be 'canonical' or 'frame', got {method!r}")

    if (
        isinstance(node_budget, (bool, np.bool_))
        or not isinstance(node_budget, numbers.Integral)
        or node_budget < 1
    ):
        raise ValueError("node_budget must be a positive integer")
    node_budget = int(node_budget)

    charge = _as_exact_int(charge, "charge")
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
                "no numerically stable canonical frame was found within the "
                f"{_FRAME_CANDIDATE_BUDGET}-candidate budget; falling back "
                "to method='canonical'. A C descriptor is returned only if "
                "the canonical search completes within node_budget.",
                UserWarning,
                stacklevel=2,
            )
    if signature is None:
        qmat = _scaled_distances(coords, decimals)
        signature = _canonical_signature(atomic_nums, qmat, node_budget)
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
