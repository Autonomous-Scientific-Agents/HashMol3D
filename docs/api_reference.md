# HashMol3D API Reference

## Identifier format

A HashMol3D identifier has the form:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-bac9655753f489d6cbfdb299d59adbda`.

- **Hill formula**: carbon first if present, then hydrogen, then the
  remaining elements alphabetically. A count of 1 is omitted
  (`H2O`, `C6H6`, `CHBrClF`).
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
  of ten no greater than 1 Å (`1.0`, `1e-1`, `1e-2`, ...)
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

### `hash_length_for(n_items, target_prob=1e-9) -> int`

Return the minimum geometry-hash length (hex chars) that keeps the
birthday-collision probability below `target_prob` for a namespace of
`n_items` distinct geometries, using `P ~ n² / 2^{b+1}` with `b = 4·length`
bits. The result is clamped to `[1, 64]`.

```python
from hashmol3d import hash_length_for, hash_molecule

L = hash_length_for(10**9)  # 23 hex chars (1e9 items, p=1e-9)
res = hash_molecule(z, coords, length=L)
```

**Returns:** `HashMol3DResult` with fields:

- `hash_str`: full HashMol3D identifier (formula + state + `-` + hash)
- `formula`: Hill-order molecular formula
- `geometry_hash`: hex-only geometry portion (useful for grouping
  molecules that differ only in charge/multiplicity)
- `version`: descriptor version string (constant for a given release)
- `precision`: geometry-grid precision used (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `descriptor`: raw descriptor string that was hashed (for debugging)

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

## `hash_xyz(path, **kwargs)`

Convenience wrapper that reads an XYZ file and forwards the keyword
arguments to `hash_molecule`.

```python
from hashmol3d import hash_xyz

result = hash_xyz("water.xyz", precision=1e-3, charge=0)
print(result.hash_str)  # H2Oq0m1-...
print(result.geometry_hash)  # ...
```

## `read_xyz(path)`

Parse a standard XYZ file and return `(atomic_nums, coords)` as NumPy
arrays. Validates the atom count declared in the header and raises
`ValueError` on malformed input.

```python
from hashmol3d import read_xyz

atomic_nums, coords = read_xyz("molecule.xyz")
```

## Constants

- `DESCRIPTOR_VERSION`: the descriptor-format version baked into every
  geometry hash. Bumping this value invalidates previously computed hashes.

## Deprecated

- `generate_hashmol3d(...)`: thin alias for `hash_molecule` that emits a
  `DeprecationWarning`. The old `hash_length=` keyword still works (it
  maps to `length=`). Note that since 0.5.0 the returned `hash_str` is
  the full readable identifier, not just the hex digest.
