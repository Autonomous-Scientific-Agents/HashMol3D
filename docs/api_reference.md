# HashMol3D API Reference

## Identifier format

A HashMol3D identifier has the form:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-a1b28135d0c66...`.

- **Hill formula**: carbon first if present, then hydrogen, then the
  remaining elements alphabetically. A count of 1 is omitted
  (`H2O`, `C6H6`, `CHBrClF`).
- **State tag**: `q<sign><charge>m<mult>`. Zero charge is written `q0`;
  non-zero charges always carry an explicit sign (`q+1`, `q-2`).
- **Geometry hash**: hex truncation of SHA-256 over the geometry-only
  descriptor (atomic numbers + pairwise distances + precision +
  descriptor version). Charge and multiplicity are *not* hashed — they
  live in the readable prefix — so molecules that differ only in charge
  or multiplicity share the same `geometry_hash` and you can find them
  by suffix match.

## `hash_molecule(atomic_nums, coords, *, precision=1e-4, charge=0, multiplicity=None, length=None)`

Compute the HashMol3D identifier for a 3D molecular geometry.

All optional arguments are **keyword-only**.

**Parameters:**

- `atomic_nums`: integer array-like of atomic numbers, shape `(N,)`
- `coords`: float array-like of Cartesian coordinates in Å, shape `(N, 3)`
- `precision`: distance precision in Å (default `1e-4`)
- `charge`: total formal charge (default `0`)
- `multiplicity`: spin multiplicity. If `None`, inferred as singlet/doublet
  from electron parity.
- `length`: number of hex characters in the geometry hash. Must be in
  `[1, 64]`. If `None` (default), auto-scales with the number of atoms
  as `clip(N, 16, 64)` so collision risk stays roughly constant as
  molecules grow.

**Returns:** `HashMol3DResult` with fields:

- `hash_str`: full HashMol3D identifier (formula + state + `-` + hash)
- `formula`: Hill-order molecular formula
- `geometry_hash`: hex-only geometry portion (useful for grouping
  molecules that differ only in charge/multiplicity)
- `version`: descriptor version string (constant for a given release)
- `precision`: distance precision used (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `descriptor`: raw descriptor string that was hashed (for debugging)

`str(result)` returns `result.hash_str`.

**Invariance:** the `geometry_hash` (and therefore the full identifier,
at fixed charge and multiplicity) is invariant under permutation of
atoms, rigid translation, rigid rotation, spatial inversion (parity),
and sub-precision numerical noise.

## `hash_xyz(path, **kwargs)`

Convenience wrapper that reads an XYZ file and forwards the keyword
arguments to `hash_molecule`.

```python
from hashmol3d import hash_xyz

result = hash_xyz("water.xyz", precision=1e-3, charge=0)
print(result.hash_str)         # H2Oq0m1-...
print(result.geometry_hash)    # ...
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
