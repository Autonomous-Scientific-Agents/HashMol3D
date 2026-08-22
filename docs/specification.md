# HashMol3D Specification v0.7.0

**Status:** Draft standard
**Canonical algorithm:** SHA-256
**Canonical version tag:** `5-CANON-SHA256`

HashMol3D is a deterministic identifier for 3D molecular conformers.
It is designed for reproducible identification of geometries in
computational-chemistry workflows and databases.

## 1. Identifier format

A HashMol3D identifier is a single ASCII string with three parts:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-a4ba9da41d888939961ef77dae43b297`.

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

The signature is furthermore **complete**: two geometries share a
geometry hash *only if* their element-labeled distance matrices are
equal, after rounding, up to an atom relabeling — i.e. only if the
geometries are congruent at the chosen precision (or if the truncated
SHA-256 digests collide, §7). Homometric configurations — distinct
geometries with the same distance *multiset* — receive distinct hashes.

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
3. `precision`: distance precision in Å (default `1e-4`); it must be a
   power of ten no greater than 1 Å (`1.0`, `1e-1`, `1e-2`, ...)
4. `charge`: total formal charge (default `0`)
5. `multiplicity`: spin multiplicity (default: inferred from electron parity)

## 4. Canonical geometry signature

Translation, rotation, and reflection invariance come from using only
pairwise distances. Permutation invariance comes from writing the
element-labeled distance matrix in a **canonical atom order** that is a
pure function of the geometry. Because the full labeled distance matrix
determines the point set up to congruence, two geometries receive the
same signature **if and only if** they are congruent at the chosen
precision — homometric configurations (distinct geometries sharing a
distance multiset) do not collide.

### 4.1 Scaled distance matrix

For every atom pair `(i, j)` compute the Euclidean distance
`d_ij = ||r_i - r_j||` and quantize it to an integer number of grid
units:

    decimals = -log10(precision)                # a non-negative integer
    q_ij     = rint(d_ij * 10^decimals)        # round-half-to-even

`q` is a symmetric non-negative integer matrix with zero diagonal. All
subsequent steps operate on exact integers. (If any scaled distance
reaches 2^62 the input is rejected; choose a coarser precision.)

### 4.2 Color refinement (Weisfeiler-Leman)

Assign each atom an initial *color*: the rank of its atomic number
among the distinct Z values present (ascending). Then refine until
stable: in each round, recolor atom `i` by the pair

    (color_i, sorted multiset of (color_j, q_ij) over all j != i)

and replace colors by the lexicographic ranks of these signatures.
Colors only ever split (the old color is the primary sort key), and the
partition stabilizes in at most N rounds (1–3 in practice). The
resulting colors are independent of the input atom order.

### 4.3 Canonical order by individualization-refinement

If the stable partition assigns every atom a distinct color, sorting
atoms by color gives the canonical order directly. Otherwise, perform a
depth-first search:

1. Pick the target cell: the smallest color class with more than one
   atom (ties: smallest color).
2. For **each** atom in that cell, *individualize* it (give it a new
   color ordered immediately before its former cellmates), re-run
   refinement (§4.2), and recurse.
3. Each leaf (discrete partition) yields an atom order; evaluate the
   candidate string `upper-triangle of q in that order, row-major` and
   keep the lexicographically smallest.

The set of leaves explored is a function of the geometry alone, so the
winning order — and therefore the signature — is permutation-invariant.
The number of leaves equals the order of the geometry's rounded-distance
symmetry group (1 for generic molecules, 24 for a perfect tetrahedral
cluster, 2n for an ideal n-ring).

### 4.4 Degenerate-rounding fallback

The search visits at most **10,000** partition states (a normative
constant of this version). The tree size is permutation-invariant, so
exceeding the budget is a deterministic property of the geometry; it
requires rounding so coarse that many atoms become mutually
indistinguishable (e.g. a cluster hashed at a precision larger than its
diameter). Such inputs fall back to hashing the stable-WL per-atom
signature multiset: for each atom the triple
`(Z_i, color_i, sorted multiset of (color_j, q_ij))` with the stable
colors of §4.2, the triples sorted as a multiset. The fallback uses a
distinct descriptor section tag (`W` instead of `C`, §6), so the two
paths can never collide with each other.

### 4.5 Optional O(N) frame method (`method="frame"`)

For very large systems (proteins, clusters) where the O(N²) distance
matrix is prohibitive, an opt-in O(N log N) method hashes coordinates in
a canonical principal-axes frame instead:

1. Compute the Z-weighted centroid and the Z-weighted **gyration tensor**
   `T_ab = Σ_i Z_i (r_i − c)_a (r_i − c)_b`. All sums are computed over
   *sorted* addends so the result is exactly independent of the input
   atom order.
2. **Reliability check (normative):** with eigenvalues
   `λ1 ≤ λ2 ≤ λ3`, require `λ3 > 0` and both relative gaps
   `(λ2−λ1)/λ3` and `(λ3−λ2)/λ3` to be at least **0.05**. Below this the
   axes are degenerate or nearly so (symmetric tops, linear molecules)
   and reorient under arbitrarily small perturbations; the implementation
   emits a warning and falls back to the canonical method of §4.2–4.4.
3. Project the centered coordinates onto the eigenvectors (ascending
   eigenvalue order) and quantize each coordinate to the precision grid,
   `q = rint(u · 10^decimals)` (same overflow bound as §4.1).
4. **Axis signs need no convention:** evaluate all eight sign
   combinations of the three axes; for each, sort the `(Z, x, y, z)`
   rows lexicographically; keep the lexicographically smallest row list.
   Because the eight combinations include both parities, the signature is
   reflection-invariant by construction.

The result is serialized under a distinct section tag (`F`, §6), so
frame-method hashes can never collide with canonical-method hashes; the
two methods are, by the same token, **not comparable** — a corpus must
choose one method. When the frame is reliable, equal `F` descriptors
certify identical quantized coordinates in a canonical frame, i.e. the
method is complete in the same sense as §4. The gap threshold bounds the
frame's noise amplification (≈ 1/gap), and the residual floating-point
caveats of §8 apply with that factor.

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

    V:<version>|P:<precision>|Z:<z_ordered>|C:<distances>     (canonical path)
    V:<version>|P:<precision>|Z:<z_sorted>|W:<atom sigs>      (fallback path, §4.4)
    V:<version>|P:<precision>|Z:<z_rows>|F:<coordinates>      (frame method, §4.5)

Where:

- `<version>` is a string, e.g. `5-CANON-SHA256`.
- `<precision>` is in scientific notation, e.g. `1.0e-04`.
- `<z_ordered>` is the list of atomic numbers in canonical atom order,
  comma-separated (this always coincides with the ascending-sorted
  multiset, because the initial colors of §4.2 rank atoms by Z).
- `<distances>` is the upper triangle of the scaled integer distance
  matrix `q` (§4.1) in canonical atom order, row-major
  (`q_12, q_13, ..., q_1N, q_23, ...`), comma-separated. Empty for a
  single atom.
- `<atom sigs>` (fallback only) is the sorted multiset of per-atom
  signatures, each formatted as `Z,color:c1-q1,c2-q2,...` with the
  atom's stable color and its sorted `(color, q)` row, joined by `;`.
- `<coordinates>` (frame method only) is the winning sorted row list of
  §4.5, each row formatted as `Z:x,y,z` with quantized integer
  coordinates, joined by `;`; `<z_rows>` lists the atomic numbers in
  that row order.

Charge and multiplicity are **not** part of the descriptor; they are
written into the readable prefix of the identifier instead.

Example (water, `precision = 1e-4`); this descriptor's SHA-256 digest,
truncated to the default 32 hex characters, is the geometry hash
`a4ba9da41d888939961ef77dae43b297`:

    V:5-CANON-SHA256|P:1.0e-04|Z:1,1,8|C:15144,9575,9575

(The two hydrogens precede the oxygen; the first two entries are the
H–H and H–O rows: `q_HH = 15144`, `q_HO = q_H'O = 9575` grid units of
1e-4 Å.)

## 7. Hashing

1. Encode the descriptor as UTF-8 bytes.
2. Compute the SHA-256 digest.
3. Take the first `length` hex characters of the hex digest.

`length ∈ [1, 64]`. When not explicitly supplied, the reference
implementation uses a fixed default of 32 hex characters (128 bits).
Collision resistance is a property of the namespace, not of molecule
size: for `n` distinct geometries hashed into `b = 4·length` bits, the
expected number of birthday collisions is `~ n² / 2^{b+1}`. The 128-bit
default keeps that below one for corpora up to ~10¹⁶ geometries. Callers
who know their corpus size may pick `length` accordingly (the reference
implementation provides `hash_length_for(n_items, target_prob)`), or pin
any fixed value in `[1, 64]`.

## 8. Determinism and portability

To guarantee identical identifiers across machines:

- Use the same descriptor version tag.
- Use the same precision.
- Format atomic numbers and scaled distances as decimal integers with
  no leading zeros or sign.
- Quantize with round-half-to-even (IEEE 754 `rint`), as in §4.1.
- Use the search-node budget of 10,000 exactly (§4.4) and, for the frame
  method, the relative eigenvalue-gap threshold of 0.05 exactly (§4.5).
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
    multiplicity=None,  # inferred if None
    length=None,  # 32 hex (128-bit) if None
)
print(result.hash_str)  # H2Oq0m1-a4ba9da41d888939961ef77dae43b297
print(result.geometry_hash)  # a4ba9da41d888939961ef77dae43b297
```

A file-based convenience wrapper is also provided:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)
```
