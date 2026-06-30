# HashMol3D API Reference

## `generate_hashmol3d(atomic_nums, coords, precision=1e-4, charge=0, multiplicity=None, hash_length=32, version="3-INV-SHA256")`

Compute the HashMol3D identifier for a 3D molecular geometry.

**Parameters:**

- `atomic_nums`: integer array-like of atomic numbers, shape `(N,)`
- `coords`: float array-like of Cartesian coordinates in Å, shape `(N, 3)`
- `precision`: distance precision in Å (default `1e-4`)
- `charge`: total formal charge (default `0`)
- `multiplicity`: spin multiplicity. If `None`, inferred as singlet/doublet
  from electron parity.
- `hash_length`: number of hex characters retained from the SHA-256 digest.
  Must be in `[1, 64]`.
- `version`: descriptor version tag (default `"3-INV-SHA256"`).
- `protocol` *(keyword-only, deprecated)*: alias for `version`.

**Returns:** `HashMol3DResult` with fields:

- `hash_str`: canonical HashMol3D identifier (hex string)
- `version`: descriptor + hash version string
- `precision`: distance precision used (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `descriptor`: raw descriptor string (for debugging)

The result also exposes `result.protocol` and `result.chiral_sign`
as backwards-compatible read-only properties.

**Invariance:** the identifier is invariant under permutation of atoms,
rigid translation, rigid rotation, spatial inversion (parity), and
sub-precision numerical noise.

## XYZ parsing (CLI helper)

```python
from hashmol3d.cli import parse_xyz
atomic_nums, coords = parse_xyz("molecule.xyz")
```

`parse_xyz` validates the atom count declared in the header and raises
`ValueError` on malformed input.
