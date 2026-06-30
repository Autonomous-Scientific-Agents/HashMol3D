# HashMol3D Design Notes

## Goal

A deterministic, rotation-, translation-, permutation-, and parity-
invariant identifier for 3D molecular conformers. The invariance set
matches the eigenvalues of the non-relativistic molecular Hamiltonian:
if those eigenvalues don't change, the hash doesn't change.

## Why a sorted multiset of `(Z_min, Z_max, d)` triples

Building a "canonical atom ordering" for a symmetric molecule is the
classical hard problem behind canonical SMILES / InChI. Naive greedy
schemes (sort by atomic number, then by sorted distance row, then by
some tiebreaker) silently fail for benzene, cubane, C60, and similar
high-symmetry geometries: equivalent atoms produce equal sort keys, and
the order chosen among them changes which distances appear at which
position of the flattened upper-triangular matrix.

Instead of canonicalizing the atom order at all, we hash a fingerprint
that is invariant under any relabeling **by construction**:

  * the sorted multiset of atomic numbers, and
  * the sorted multiset of `(Z_min, Z_max, distance)` triples
    over all unordered pairs of atoms.

Both objects depend only on Z and pairwise distances, so they inherit
translation, rotation, and parity invariance for free.

## Why we do not encode chirality

The non-relativistic Born-Oppenheimer molecular Hamiltonian commutes
with the spatial-inversion operator, so enantiomers share its
eigenvalue spectrum. Following the stated goal, the hash should not
change under reflection, and so HashMol3D does not attempt to
distinguish enantiomers. Users who need stereochemistry should record
it as a separate tag alongside the HashMol3D identifier.

This sidesteps a class of bugs that geometric chirality detectors are
prone to: degenerate principal axes (symmetric and spherical tops), the
sign ambiguity of `numpy.linalg.eigh`'s eigenvectors, and ordering-
dependent volume signs in greedy quartet pickers.

## Precision and rounding

Distances are rounded to `max(0, round(-log10(precision)))` decimal
places before hashing. This gives stable hashes under noise smaller
than `precision / 2`. Two conformers that differ by less than
`precision` may collide; two that differ by more will usually not.

There is a residual rounding-boundary risk: a distance of exactly
`x.xxxx5` may round to two different values depending on numerical
noise. Choose `precision` generously larger than the geometric noise
floor of your pipeline.

## Distance-multiset uniqueness ("homometric" sets)

In principle, two distinct geometries can share a distance multiset
(this is the "Patterson ambiguity" in crystallography, equivalently the
turnpike-problem non-uniqueness). For real molecular geometries with
atomic-number-tagged pairs and ~10–100 atoms, collisions are
vanishingly unlikely; we accept this as a tradeoff for rigorous
invariance and a dependency-free implementation.

## Dependencies

The implementation uses only NumPy and the Python standard library.
Dropping the RDKit dependency simplifies installation in HPC
environments and removes a non-trivial transitive-dependency surface.
