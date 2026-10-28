# HashMol3D API Reference

## Identifier format

A HashMol3D identifier has the form:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-8ec3152acf35859ba58ced3e101c87e1`.

- **Hill formula**: when carbon is present, carbon first, then hydrogen,
  then the remaining elements alphabetically; without carbon, all elements
  alphabetically. A count of 1 is omitted (`H2O`, `C6H6`, `CHBrClF`,
  `BrH`).
- **State tag**: `q<charge>m<mult>`. Zero and positive charges have no
  sign (`q0`, `q1`); only negative charges carry a leading `-`
  (`q-1`, `q-2`).
- **Geometry hash**: hex truncation of SHA-256 over the geometry-only
  descriptor (atomic numbers + canonical-frame coordinates or canonical
  pairwise distances + precision +
  descriptor version). Charge and multiplicity are *not* hashed — they
  live in the readable prefix — so molecules that differ only in charge
  or multiplicity share the same `geometry_hash` and you can find them
  by suffix match.

## `hash_molecule(atomic_nums, coords, *, precision=1e-4, charge=0, multiplicity=None, length=None, method="frame", node_budget=10000)`

Compute the HashMol3D identifier for a 3D molecular geometry.

All optional arguments are **keyword-only**.

**Parameters:**

- `atomic_nums`: integer array-like of atomic numbers, shape `(N,)`
- `coords`: float array-like of Cartesian coordinates in Å, shape `(N, 3)`
- `precision`: geometry-grid precision in Å (default `1e-4`); must be a power
  of ten no greater than 1 Å (`1.0`, `1e-1`, `1e-2`, ...), and no finer than
  `2**-35` (≈ 2.9e-11) times the largest distance of an atom from the
  Z-weighted centroid. Finer grids are below float64 resolution (identifiers
  would change under rotation) and raise `ValueError`.
- `charge`: total formal charge (default `0`)
- `multiplicity`: spin multiplicity. If `None`, inferred as singlet/doublet
  from electron parity.
- `length`: number of hex characters in the geometry hash. Must be in
  `[1, 64]`. If `None` (default), uses `DEFAULT_LENGTH` (32 hex = 128 bits).
  Collision resistance depends on how many distinct geometries share a
  namespace, not on molecule size; use `hash_length_for()` to size the hash
  to a target corpus.
- `method`: `"frame"` (default) normally uses O(N log N) time and O(N)
  memory. Degenerate eigenspaces use intrinsic point/line coordinates or
  canonical atom anchors. `"canonical"` selects the O(N²) labelled
  distance-matrix descriptor. A frame that is ill-conditioned or exceeds
  its 10,000-candidate budget warns and uses the canonical path.

- `node_budget`: positive integer maximum number of canonical search states
  (default 10,000), including after frame fallback. Exhaustion raises the public
  `hashmol3d.SearchBudgetExceeded` exception without creating a descriptor,
  hash, or result. Increase this budget and retry; completed hashes are unchanged.

**Raises:**

- `ValueError` for invalid input: an empty molecule, mismatched or
  non-`(N, 3)` shapes, non-finite coordinates, atomic numbers outside
  `[1, 118]`, a precision that is not a power of ten ≤ 1 Å or is below the
  resolution floor, an invalid `length` or `method`, `node_budget < 1`, or
  `multiplicity < 1`.
- `hashmol3d.SearchBudgetExceeded` (a `RuntimeError` subclass) when the
  canonical search exhausts `node_budget`.

**Warns:** `UserWarning` when `method="frame"` finds no stable frame and falls
back to the canonical method.

**Returns:** a `HashMol3DResult`, a frozen dataclass (importable from
`hashmol3d`) with fields:

- `hash_str`: full HashMol3D identifier (formula + state + `-` + hash)
- `formula`: Hill-order molecular formula
- `geometry_hash`: hex-only geometry portion (useful for grouping
  molecules that differ only in charge/multiplicity)
- `version`: descriptor version string: `DESCRIPTOR_VERSION` for
  geometry-only results; RDKit S results append `-S1-RDKIT-<rdkitVersion>`
- `precision`: geometry-grid precision used (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `descriptor`: the stored descriptor. Its `F` rows carry canonical-frame
  coordinates on a grid 10⁴ times finer than `precision`; the geometry hash
  is the SHA-256 digest of these values rounded back to `precision`
  ([specification §6](specification.md#6-canonical-descriptor-string)).
  Re-hash it with `hash_descriptor`.

`str(result)` returns `result.hash_str`.

**Invariance:** in exact arithmetic the `geometry_hash` (and therefore the
full identifier, at fixed charge and multiplicity) is invariant under
permutation of atoms, rigid translation, rigid rotation, and spatial
inversion (parity).

Invariance under numerical noise is **not** guaranteed, including noise
smaller than `precision`: see [specification §2](specification.md), which is
normative here. A perturbation changes the identifier when it moves the
quantized representation to a different grid cell, and its magnitude alone
does not determine whether it does.

## `hash_descriptor(descriptor, *, length=None)`

Return the geometry hash of a stored descriptor without recomputing any
geometry. `hash_descriptor(r.descriptor, length=len(r.geometry_hash)) ==
r.geometry_hash` holds exactly for every result `r`, for `F` and `C`
descriptors, and for stored descriptors of earlier versions. `length`
defaults to 32 hex characters. The readable `<formula><state>-` prefix is not
part of the descriptor; keep it alongside.

The descriptor is validated first, against the rules of its own version,
and rejected with `ValueError` if it breaks any of them: known version
(5–10) with a section that version produced (`F`, `C`, or the `W` fallback
of versions 5–6), the precision field as that version wrote it (any
positive value for version 5, a power of ten ≤ 1 Å from version 6), atomic
numbers in range and ascending, `F`/`W` row elements matching `Z` in
canonical order, the right number of `C` distances, canonical integers
with absolute value below 2⁶², and for version 10 the rounding margin and the exact
resolution floor ([specification §6.1](specification.md#61-idempotent-re-hashing)).
Corrupted or hand-edited descriptors therefore fail instead of hashing. The
checks are necessary conditions: they do not test whether `C` distances are
realizable in three dimensions. A descriptor from the RDKit S extension
(`V:<base>-S1-RDKIT-<rdkit>|...|S:<SMILES>`, base version 8 or later) is
validated as its base descriptor plus a non-empty `S` field and re-hashes to
its S digest; the SMILES is not interpreted.

## `descriptor_geometry(descriptor)`

Return `(atomic_nums, coords)` for an `F` descriptor of the current version:
the molecule centered on its Z-weighted centroid in its canonical frame, in
Å. Hashing the result with `hash_molecule` at the descriptor's precision
(default `method="frame"`) reproduces the descriptor's identifier, in any
orientation and atom order. The coordinates are re-hashed before they are
returned; if recomputing the frame would move a value across a rounding
boundary, they are deterministically settled by a small fraction of a grid
cell ([specification §6.1](specification.md#61-idempotent-re-hashing)). If no
settled geometry re-hashes correctly, which has been observed only for
exactly symmetric inputs with tied anchor frames, a `UserWarning` is emitted
and the stored rows are returned unchanged.

Raises `ValueError` for a `C` descriptor (a distance matrix has no
coordinates), for a descriptor of another version, or for a descriptor that
fails the validation of `hash_descriptor`. A geometry accepted at the
resolution floor reconstructs to one that `hash_molecule` also accepts: the
floor is decided on the hashed rows, which the reconstruction shares.

```python
from hashmol3d import descriptor_geometry, hash_descriptor, hash_molecule

res = hash_molecule(z, coords)
assert hash_descriptor(res.descriptor) == res.geometry_hash
z2, coords2 = descriptor_geometry(res.descriptor)
assert hash_molecule(z2, coords2).hash_str == res.hash_str
```

## `hash_xyz(path, *, include_smiles=False, **kwargs)`

Convenience wrapper that reads an XYZ file and forwards the keyword
arguments to `hash_molecule`.
With `include_smiles=True`, opt into RDKit bond perception and the separately
versioned S descriptor. The default remains unchanged.

```python
from hashmol3d import hash_xyz

result = hash_xyz("water.xyz", precision=1e-3, charge=0)
print(result.hash_str)  # H2Oq0m1-...
print(result.geometry_hash)  # ...
```

## `read_xyz(path)`

Parse a standard XYZ file and return `(atomic_nums, coords)` as NumPy
arrays. Validates the atom count declared in the header and raises
`ValueError` on malformed input. Reading stops after the declared number of
atoms, so a multi-frame trajectory yields its first frame; any non-blank
content after that point emits a `UserWarning`, since an under-declared count
would otherwise silently hash a truncated molecule.

```python
from hashmol3d import read_xyz

atomic_nums, coords = read_xyz("molecule.xyz")
```

## `hash_length_for(n_items, target_prob=1e-9) -> int`

Return the minimum geometry-hash length (hex chars) that keeps the
birthday-collision probability below `target_prob` for a namespace of
`n_items` distinct geometries, using `P ~ n² / 2^{b+1}` with `b = 4·length`
bits. `n_items <= 1` returns 1. The result is clamped to `[1, 64]`; when the
approximation asks for more than 64, a `UserWarning` is emitted and 64 is
returned. Raises `ValueError` if `target_prob` is not in `(0, 1)`.

```python
from hashmol3d import hash_length_for, hash_molecule

L = hash_length_for(10**9)  # 23 hex chars (1e9 items, p=1e-9)
res = hash_molecule(z, coords, length=L)
```

## Optional RDKit APIs

For optional `hash_rdkit`, `hash_file`, `read_rdkit`, and `canonical_smiles`
APIs, see [RDKit support](rdkit.md). The geometry-only invariance and
charge-independent suffix rules above apply when `include_smiles=False`.
S results use a version that includes the extension revision and RDKit version;
their `geometry_hash` field hashes geometry plus SMILES.

## Constants

- `DESCRIPTOR_VERSION`: the descriptor-format version baked into every
  geometry hash (`"10-FRAME-SHA256"`). Bumping this value invalidates
  previously computed hashes.
- `DEFAULT_LENGTH`: default geometry-hash length in hex characters (`32`).
- `__version__`: the package version (`"0.14.0"`).

## Deprecated

- `generate_hashmol3d(atomic_nums, coords, precision=1e-4, charge=0,
  multiplicity=None, hash_length=32)`: deprecated since 0.4.0; a wrapper for
  `hash_molecule` that emits a `DeprecationWarning` and does not expose
  `method` or `node_budget`. The old `hash_length=` keyword still works (it
  maps to `length=`). Note that since 0.5.0 the returned `hash_str` is
  the full readable identifier, not just the hex digest.
