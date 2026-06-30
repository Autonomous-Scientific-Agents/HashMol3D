# HashMol3D Specification v0.4.0

**Status:** Draft standard
**Canonical algorithm:** SHA-256
**Canonical output length:** 32 hex characters (128 bits)
**Canonical version tag:** `3-INV-SHA256`

HashMol3D is a deterministic identifier for 3D molecular conformers.
It is designed for reproducible identification of geometries in
computational-chemistry workflows and databases.

## 1. Invariance contract

The HashMol3D identifier is invariant under exactly those operations
that leave the eigenvalues of the non-relativistic molecular
Hamiltonian unchanged:

- rigid translation of the coordinates
- rigid rotation of the coordinates
- permutation (relabeling) of atom indices
- spatial inversion / reflection (parity)
- numerical noise smaller than the user-specified precision

It is **not** invariant under:

- changes in any atomic number Z
- changes in total charge
- changes in spin multiplicity
- changes in the descriptor version tag
- geometric distortions larger than the chosen precision

> **Note on chirality.** Two enantiomers share the eigenvalues of the
> non-relativistic Hamiltonian and therefore share the same HashMol3D
> identifier. If you need to distinguish enantiomers, combine HashMol3D
> with an external stereochemistry tag.

## 2. Inputs

The reference implementation takes:

1. `atomic_nums`: integer array of atomic numbers, shape `(N,)`
2. `coords`: float array of Cartesian coordinates in Å, shape `(N, 3)`
3. `precision`: distance precision in Å (default `1e-4`)
4. `charge`: total formal charge (default `0`)
5. `multiplicity`: spin multiplicity (default: inferred from electron parity)

## 3. Pair signature

Permutation, translation, rotation, and reflection invariance are all
achieved together by reducing the geometry to a multiset of pairwise
distances tagged by atomic numbers.

For every unordered pair of atoms `(i, j)` with `i < j`:

1. Compute the Euclidean distance `d_ij = ||r_i - r_j||`.
2. Round to `decimals = max(0, round(-log10(precision)))` decimal places.
3. Emit the triple `(min(Z_i, Z_j), max(Z_i, Z_j), d_ij_rounded)`.

The list of all such triples is sorted lexicographically. This sorted
list is invariant under any relabeling of atoms (it is a multiset
keyed only on Z and distance) and under any rigid motion or reflection
of the geometry (it depends only on pairwise distances).

The sorted list of atomic numbers is included as a separate
component so that empty-distance corner cases (single atom) still
distinguish elements.

## 4. Multiplicity inference

If `multiplicity` is not supplied:

    electrons   = sum(Z_i) - charge
    multiplicity = 1 if electrons is even else 2

This singlet/doublet default is appropriate when no other spin
information is available. Callers that know better should pass
`multiplicity` explicitly.

## 5. Canonical descriptor string

The descriptor is a UTF-8 string with the following pipe-separated
components, in this fixed order:

    V:<version>|P:<precision>|Z:<z_sorted>|D:<pairs>|Q:<charge>|M:<multiplicity>

Where:

- `<version>` is a string, e.g. `3-INV-SHA256`.
- `<precision>` is in scientific notation, e.g. `1.0e-04`.
- `<z_sorted>` is the sorted list of atomic numbers, comma-separated.
- `<pairs>` is the sorted list of triples, formatted as
  `Za-Zb:d.dddd`, comma-separated.
- `<charge>` is the integer formal charge.
- `<multiplicity>` is the integer spin multiplicity.

Example (water, `precision = 1e-4`):

    V:3-INV-SHA256|P:1.0e-04|Z:1,1,8|D:1-1:1.5144,1-8:0.9579,1-8:0.9579|Q:0|M:1

## 6. Hashing

1. Encode the descriptor as UTF-8 bytes.
2. Compute the SHA-256 digest.
3. Take the first `length` hex characters of the hex digest.

`length ∈ [1, 64]`. The recommended canonical value is 32
(128 bits). 16 (64 bits) is acceptable for small datasets; 64
(full 256 bits) is appropriate for archival.

## 7. Determinism and portability

To guarantee identical hashes across machines:

- Use the same version tag.
- Use the same precision.
- Format atomic numbers as decimal integers.
- Format distances with exactly `decimals` fractional digits.
- Encode the descriptor in UTF-8 before hashing.
- Use SHA-256 as defined in FIPS 180-4.

Any change to the descriptor format or semantics requires a new
version tag.

## 8. Dependencies

The reference implementation uses only NumPy and the Python standard
library; in particular it does **not** depend on RDKit or any
cheminformatics toolkit.

## 9. Reference API

```python
from hashmol3d import hash_molecule

result = hash_molecule(
    atomic_nums,
    coords,
    precision=1e-4,
    charge=0,
    multiplicity=None,   # inferred if None
    length=32,
)
print(result.hash_str)
```

A file-based convenience wrapper is also provided:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)
```
