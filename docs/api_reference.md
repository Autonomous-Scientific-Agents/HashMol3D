# HashMol3D API Reference

## `hash_molecule(atomic_nums, coords, *, precision=1e-4, charge=0, multiplicity=None, length=32)`

Compute the HashMol3D identifier for a 3D molecular geometry.

All optional arguments are **keyword-only**.

**Parameters:**

- `atomic_nums`: integer array-like of atomic numbers, shape `(N,)`
- `coords`: float array-like of Cartesian coordinates in Å, shape `(N, 3)`
- `precision`: distance precision in Å (default `1e-4`)
- `charge`: total formal charge (default `0`)
- `multiplicity`: spin multiplicity. If `None`, inferred as singlet/doublet
  from electron parity.
- `length`: number of hex characters retained from the SHA-256 digest.
  Must be in `[1, 64]` (default `32`).

**Returns:** `HashMol3DResult` with fields:

- `hash_str`: canonical HashMol3D identifier (hex string)
- `version`: descriptor version string (constant for a given release)
- `precision`: distance precision used (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `descriptor`: raw descriptor string (for debugging)

`str(result)` returns `result.hash_str`.

**Invariance:** the identifier is invariant under permutation of atoms,
rigid translation, rigid rotation, spatial inversion (parity), and
sub-precision numerical noise.

## `hash_xyz(path, **kwargs)`

Convenience wrapper that reads an XYZ file and forwards the keyword
arguments to `hash_molecule`.

```python
from hashmol3d import hash_xyz

result = hash_xyz("water.xyz", precision=1e-3, charge=0)
print(result.hash_str)
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
  hash. Bumping this value invalidates previously computed hashes.

## Deprecated

- `generate_hashmol3d(...)`: kept as a thin alias for `hash_molecule`
  that emits a `DeprecationWarning`. The old `hash_length=` keyword still
  works; in the new API it is called `length=`.
