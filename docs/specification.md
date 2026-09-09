# HashMol3D Specification v0.10.0

**Status:** Proposed standard (draft), developed using the HashMol3D library
**Canonical algorithm:** SHA-256
**Canonical version tag:** `7-FRAME-SHA256`

HashMol3D is a deterministic identifier for 3D molecular conformers.
It is designed for reproducible identification of geometries in
computational-chemistry workflows and databases.

## 1. Identifier format

A HashMol3D identifier is a single ASCII string with three parts:

    <Hill formula><state tag>-<geometry hash>

For example: `H2Oq0m1-b4db5388ff28342bdc809a83891e65ea`.

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

In **exact arithmetic** the mathematical descriptor is invariant under
rigid motions and atom relabeling:

- rigid translation of the coordinates
- rigid rotation of the coordinates
- permutation (relabeling) of atom indices
- spatial inversion / reflection (parity)

These are all rigid motions (plus parity) composed with relabeling; they
are a subset of the transformations that leave the eigenvalues of the
non-relativistic molecular Hamiltonian unchanged, but the descriptor does
**not** claim invariance under every spectrum-preserving operation.

Invariance to small numerical noise is **not** guaranteed. Perturbations
can change the descriptor when they cross quantization or frame-selection
boundaries (a near-degenerate principal axis, an anchor at its acceptance
threshold); their magnitude alone does not determine whether the
identifier changes. In particular a perturbation smaller than the
requested precision can still change the hash if it straddles a boundary,
and one larger than the precision can leave it unchanged if it does not.
See §7 and the manuscript's finite-grid and stability analysis for the
empirical rate.

The geometry hash is also **not** invariant under:

- changes in any atomic number Z (isotopes share Z and therefore hash alike)
- changes in the descriptor version tag
- geometric distortions that move the quantized representation to a
  different grid cell (a distortion larger than the chosen precision need
  not do so, and a smaller one can)

The default frame signature is furthermore **complete on its coordinate
grid**: equal descriptors contain the same sorted element-labelled
coordinates in an orthonormal canonical frame. The optional canonical
distance signature is complete on its rounded-distance grid. Completeness
concerns each method's specified *quantized* representation: quantization
can merge distinct unrounded geometries into one grid cell. Subject to
that quantization, homometric configurations — distinct geometries with
the same distance *multiset* — receive distinct hashes (except for a
truncated SHA-256 collision, §7). Search exhaustion produces no hash (§4.4).

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
3. `precision`: geometry-grid precision in Å (default `1e-4`); it must be a
   power of ten no greater than 1 Å (`1.0`, `1e-1`, `1e-2`, ...)
4. `charge`: total formal charge (default `0`)
5. `multiplicity`: spin multiplicity (default: inferred from electron parity)
6. `method`: `"frame"` (default) or `"canonical"`

## 4. Geometry signatures

The default frame method is specified in §4.5. Sections §4.1–4.4 specify
the optional canonical labelled-distance method and the final fallback used
when a stable bounded frame cannot be constructed.

For the canonical distance method, translation, rotation, and reflection
invariance come from using only pairwise distances. Permutation invariance
comes from writing the element-labelled distance matrix in a **canonical
atom order** that is a pure function of the geometry. Because the full
labelled distance matrix determines the point set up to congruence, two
geometries receive the same `C` signature if and only if they are congruent
on its rounded-distance grid — homometric configurations do not collide.

### 4.1 Scaled distance matrix

For every atom pair `(i, j)` compute the Euclidean distance
`d_ij = ||r_i - r_j||` and quantize it to an integer number of grid
units:

    decimals = round(-log10(precision))         # non-negative integer (precision is a power of ten)
    q_ij     = rint(d_ij * 10^decimals)         # round-half-to-even

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

### 4.4 Canonical search budget

The default search budget is **10,000** visited partition states; callers may
set a larger positive integer `node_budget`. If the full search cannot finish
within this budget, raise `SearchBudgetExceeded` and create no descriptor,
hash, or identifier. The CLI prints the error to stderr, exits with code 1,
and suggests increasing `--node-budget`. Partial best candidates must never
be serialized, and no weaker representation is substituted.

The search tree is determined by the labelled quantized matrix, so exhaustion
at a fixed budget is permutation-invariant. The budget controls availability,
not the bytes of a successful descriptor; increasing it does not require a new
version or change an already completed hash. This limit also applies to the
canonical search invoked after frame fallback.

### 4.5 Default canonical frame method (`method="frame"`)

The default method normally takes O(N log N) time and O(N) memory:

1. Compute the Z-weighted centroid and the Z-weighted **gyration tensor**
   `T_ab = Σ_i Z_i (r_i − c)_a (r_i − c)_b`. Tensor sums use sorted
   addends so atom permutation cannot change summation order. Let centered
   vectors be `v_i`, scale `s = 10^decimals`, and eigenvalues
   `λ1 ≤ λ2 ≤ λ3`.
2. If `max_i ||v_i|| s < 0.5`, serialize every coordinate as zero. This
   is the intrinsic point-like representation.
3. Otherwise require `λ3 > 0`. Before rejecting a cloud with
   `R s < 10`, where `R = max_i ||v_i||`, test collinearity about the
   largest-moment eigenvector `u3`. Let `rho = max_i ||v_i - (v_i·u3)u3||`.
   If `rho ≤ 64 * eps64 * R`, with `eps64 = 2^-52`, serialize the
   intrinsic line `(Z, 0, 0, rint((v_i·u3)s))`, minimizing over both axial
   signs. This accepts a line up to float64 roundoff, not an arbitrary
   sub-grid bend. Other clouds with `R s < 10` use §4.1–4.4 with a warning.
4. Define relative gaps `g1=(λ2−λ1)/λ3` and `g2=(λ3−λ2)/λ3`.
   If both are at least **0.05**, use the eigenvectors in ascending
   eigenvalue order, as in the v0.8 principal-frame method.
5. If exactly one gap is below 0.05, preserve the isolated eigenvector
   `u`. For every atom form its projection into the degenerate plane,
   `p_i = v_i − (v_i·u)u`. If the isolated axis is the largest-moment
   axis and `max_i ||p_i||s < 0.5`, serialize a one-dimensional line:
   `(Z, 0, 0, rint((v_i·u)s))`, considering both axial signs. Otherwise
   require `max_i ||p_i||s ≥ 10` and select the lexicographically largest
   invariant anchor key

       (rint(||p_i||s), Z_i, rint(|v_i·u|s)).

   Evaluate every tied anchor. Its normalized `p_i` resolves the ambiguous
   plane; a cross product supplies the remaining axis. The isolated axis
   stays in its ascending-eigenvalue slot.
6. If both gaps are below 0.05, choose first anchors by the largest key

       (rint(||v_i||s), Z_i).

   For every tied first anchor `i`, set `e1=v_i/||v_i||`, project every
   other atom perpendicular to it, and choose second anchors by the largest
   key

       (rint(||p_ij||s), Z_j, rint(|v_j·e1|s)).

   Evaluate every tied non-collinear ordered pair. Normalize `p_ij` as
   `e2` and set `e3=e1×e2`.
7. At most **10,000** tied candidate frames may be evaluated. Exceeding
   this normative budget, or finding an atom-derived axis shorter than 10
   grid units, deterministically selects the canonical method of §4.1–4.4
   with a `UserWarning`.
8. For every candidate basis, project the **original centered coordinates**
   (anchors never perturb the geometry), quantize with `rint`, evaluate all
   eight axis-sign combinations, sort `(Z,x,y,z)` rows lexicographically,
   and keep the globally smallest row list. The sign set includes both
   parities, making the descriptor reflection-invariant.

The result uses section tag `F` (§6). Generic inputs evaluate one frame;
an axial degeneracy can evaluate up to N tied anchors, and a fully
degenerate tensor can evaluate up to N(N−1) ordered pairs before the
normative budget intervenes. Equal `F` bodies contain identical quantized
element-labelled coordinates in an orthonormal frame and are therefore a
complete representation on that grid.

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
    V:<version>|P:<precision>|Z:<z_rows>|F:<coordinates>      (frame method, §4.5)

Where:

- `<version>` is a string, e.g. `7-FRAME-SHA256`.
- `<precision>` is in scientific notation, e.g. `1.0e-04`.
- `<z_ordered>` is the list of atomic numbers in canonical atom order,
  comma-separated (this always coincides with the ascending-sorted
  multiset, because the initial colors of §4.2 rank atoms by Z).
- `<distances>` is the upper triangle of the scaled integer distance
  matrix `q` (§4.1) in canonical atom order, row-major
  (`q_12, q_13, ..., q_1N, q_23, ...`), comma-separated. Empty for a
  single atom.
- `<coordinates>` (frame method only) is the winning sorted row list of
  §4.5, each row formatted as `Z:x,y,z` with quantized integer
  coordinates, joined by `;`; `<z_rows>` lists the atomic numbers in
  that row order.

Charge and multiplicity are **not** part of the descriptor; they are
written into the readable prefix of the identifier instead.

Example using the default frame method (water, `precision = 1e-4`); this
descriptor's SHA-256 digest, truncated to the default 32 hex characters,
is the geometry hash `b4db5388ff28342bdc809a83891e65ea`:

    V:7-FRAME-SHA256|P:1.0e-04|Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0

(The molecular-plane normal occupies the first axis. The two hydrogen rows
precede oxygen after lexicographic sorting.)

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
- Complete the canonical search (§4.4); the node budget only controls whether
  a result is available. For the frame method,
  use the relative eigenvalue-gap threshold 0.05, minimum atom-anchor length
  10 grid units, and candidate budget 10,000 exactly (§4.5).
- Encode the descriptor in UTF-8 before hashing.
- Use SHA-256 as defined in FIPS 180-4.
- Render the formula in Hill order and the state tag exactly as in §1.

Any change to the descriptor format or semantics requires a new
version tag.

Version 7 adds the early collinearity check for small nonpoint clouds. It
changes some coarse-grid frame requests from `C` to `F`. The version field is
hashed, so **all version-7 digests differ from their version-6 counterparts**,
even when their geometry bodies are unchanged. Recompute a corpus consistently
when migrating; version-6 and version-7 identifiers must not be mixed.

## 9. Future tagged extensions

New descriptor tags can extend the proposed standard to distinguish enantiomers
and isotopes. An enantiomer-sensitive extension needs a canonical handedness
representation; an isotope extension needs isotope labels associated with the
canonical atom order. Such extensions must define their serialization and
canonicalization rules and use a new descriptor version. The current library
implements neither extension and retains reflection invariance and atomic-number
labels only.

## 10. Dependencies

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
    method="frame",  # default; use "canonical" for the distance method
)
print(result.hash_str)  # H2Oq0m1-b4db5388ff28342bdc809a83891e65ea
print(result.geometry_hash)  # b4db5388ff28342bdc809a83891e65ea
```

A file-based convenience wrapper is also provided:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)
```
