# HashMol3D Specification v0.5.0

**Status:** Draft standard
**Canonical algorithm:** SHA-256
**Canonical version tag:** `4-GEOM-SHA256`

HashMol3D is a deterministic identifier for 3D molecular conformers.
It is designed for reproducible identification of geometries in
computational-chemistry workflows and databases.

## 1. Identifier format

A HashMol3D identifier is a single ASCII string with three parts:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-a1b28135d0c66ad0`.

- **Hill formula** — carbon first if present, then hydrogen, then the
  remaining elements alphabetically by symbol. A count of 1 is omitted.
  Examples: `H2O`, `C6H6`, `CHBrClF`, `H3N`.
- **State tag** — `q<charge>m<multiplicity>`. Zero and positive charges
  have no sign (`q0`, `q1`); only negative charges carry a leading `-`
  (`q-1`, `q-2`). Multiplicity is a positive integer with no sign.
- **`-`** — single hyphen separator, so the start of the geometry hash
  is unambiguous even though the formula and state tag contain no
  delimiters.
- **Geometry hash** — lowercase hex truncation of the SHA-256 digest of
  the geometry-only descriptor (see §3, §5, §6).

## 2. Invariance contract

The **geometry hash** is invariant under exactly those operations that
leave the eigenvalues of the non-relativistic molecular Hamiltonian
unchanged:

- rigid translation of the coordinates
- rigid rotation of the coordinates
- permutation (relabeling) of atom indices
- spatial inversion / reflection (parity)
- numerical noise smaller than the user-specified precision

It is **not** invariant under:

- changes in any atomic number Z
- changes in the descriptor version tag
- geometric distortions larger than the chosen precision

The **state tag** (and therefore the full identifier) additionally
changes with charge or multiplicity. Two states of the same geometry
share the same geometry hash but differ in their state tag, so they can
be grouped by suffix matching on the part after `-`.

> **Note on chirality.** Two enantiomers share the eigenvalues of the
> non-relativistic Hamiltonian and therefore share the same HashMol3D
> identifier. If you need to distinguish enantiomers, combine HashMol3D
> with an external stereochemistry tag.

## 3. Inputs

The reference implementation takes:

1. `atomic_nums`: integer array of atomic numbers, shape `(N,)`
2. `coords`: float array of Cartesian coordinates in Å, shape `(N, 3)`
3. `precision`: distance precision in Å (default `1e-4`)
4. `charge`: total formal charge (default `0`)
5. `multiplicity`: spin multiplicity (default: inferred from electron parity)

## 4. Pair signature

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

The sorted list of atomic numbers is included as a separate component
so single-atom corner cases still distinguish elements.

## 5. Multiplicity inference

If `multiplicity` is not supplied:

    electrons   = sum(Z_i) - charge
    multiplicity = 1 if electrons is even else 2

This singlet/doublet default is appropriate when no other spin
information is available. Callers that know better should pass
`multiplicity` explicitly.

## 6. Canonical descriptor string

The descriptor is a UTF-8 string with the following pipe-separated
components, in this fixed order:

    V:<version>|P:<precision>|Z:<z_sorted>|D:<pairs>

Where:

- `<version>` is a string, e.g. `4-GEOM-SHA256`.
- `<precision>` is in scientific notation, e.g. `1.0e-04`.
- `<z_sorted>` is the sorted list of atomic numbers, comma-separated.
- `<pairs>` is the sorted list of triples, formatted as
  `Za-Zb:d.dddd`, comma-separated.

Charge and multiplicity are **not** part of the descriptor; they are
written into the readable prefix of the identifier instead.

Example (water, `precision = 1e-4`):

    V:4-GEOM-SHA256|P:1.0e-04|Z:1,1,8|D:1-1:1.5144,1-8:0.9579,1-8:0.9579

## 7. Hashing

1. Encode the descriptor as UTF-8 bytes.
2. Compute the SHA-256 digest.
3. Take the first `length` hex characters of the hex digest.

`length ∈ [1, 64]`. The reference implementation auto-scales `length`
as `clip(N, 16, 64)` when not explicitly supplied, where `N` is the
number of atoms; this keeps birthday-collision risk roughly constant
as molecules grow. Callers may pin a fixed value (e.g. 16, 32, 64).

## 8. Determinism and portability

To guarantee identical identifiers across machines:

- Use the same descriptor version tag.
- Use the same precision.
- Format atomic numbers as decimal integers.
- Format distances with exactly `decimals` fractional digits.
- Encode the descriptor in UTF-8 before hashing.
- Use SHA-256 as defined in FIPS 180-4.
- Render the formula in Hill order and the state tag exactly as in §1.

Any change to the descriptor format or semantics requires a new
version tag.

## 9. Dependencies

The reference implementation uses only NumPy and the Python standard
library; in particular it does **not** depend on RDKit or any
cheminformatics toolkit.

## 10. Reference API

```python
from hashmol3d import hash_molecule

result = hash_molecule(
    atomic_nums,
    coords,
    precision=1e-4,
    charge=0,
    multiplicity=None,   # inferred if None
    length=None,         # auto-scaled if None
)
print(result.hash_str)        # H2Oq0m1-a1b28135d0c66ad0
print(result.geometry_hash)   # a1b28135d0c66ad0
```

A file-based convenience wrapper is also provided:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)
```
