# HashMol3D Design Notes

## Goal

A deterministic, rotation-, translation-, permutation-, and parity-
invariant identifier for 3D molecular conformers. The invariance set
matches the eigenvalues of the non-relativistic molecular Hamiltonian:
if those eigenvalues don't change, the hash doesn't change.

## Why a canonical labeled distance matrix (v5)

Versions up to 4 hashed the sorted multiset of `(Z_min, Z_max, d)`
triples. That object is invariant by construction, but it discards
*which distances share an atom*, so **homometric** configurations —
distinct geometries with the same distance multiset (the Patterson
ambiguity of crystallography) — collided. Boutin & Kemper (2004) proved
the distance multiset is a complete invariant only for *generic* point
clouds; the exceptions are measure-zero but structured: symmetric,
linear, and lattice-like arrangements. Concretely:

- the classic pair `{0,1,4,10,12,17}` / `{0,1,8,11,13,17}` on a line,
  and 3D grid products built from it, collide under v4
  (`tests/test_collisions.py` keeps these as fixtures);
- a scan of a 369,595-record THEMol shard found 5 v4 hash-collision
  groups; geometry-level verification showed all 5 to be *duplicate
  entries* (same coordinates to ~1e-6 Å under different SMILES
  annotations), i.e. **zero genuine homometric collisions** in that
  organic-conformer corpus — generic chemistry really is generic.

The practical risk therefore concentrates exactly where the theory says:
high-symmetry clusters, linear chains, lattice fragments, and
adversarial or machine-generated geometries. The v5 upgrade buys the
*guarantee* — "no geometric collisions by construction" is a theorem one
can cite for an archival identifier, where v4 could only say "none
observed so far".

The v5 descriptor removes the weakness at its root instead of patching
around it. It writes the **full element-labeled distance matrix in a
canonical atom order**:

1. **Quantize** all pairwise distances to integers on the precision
   grid (`q_ij = rint(d_ij · 10^decimals)`), so every later step is
   exact integer arithmetic.
2. **Color refinement** (Weisfeiler-Leman): iteratively recolor each
   atom by `(own color, sorted multiset of (neighbor color, q))` until
   the partition stabilizes. Converges in 1–3 rounds in practice.
3. **Individualization-refinement**: if symmetry-equivalent atoms
   remain, branch over the members of the smallest ambiguous cell,
   refine, recurse, and keep the lexicographically smallest distance
   matrix over all leaves. The leaf set is a function of the geometry
   alone, so the winner is permutation-invariant; the number of leaves
   equals the order of the rounded-distance symmetry group (1 for
   generic molecules, 12 for benzene, 24 for a perfect tetrahedral
   cluster).

A labeled distance matrix in a well-defined order determines the point
set up to congruence, so **equal descriptors now occur if and only if
the geometries are congruent at the chosen precision**: zero geometric
collisions by construction, rather than "no known collisions".

### Why not stop at per-atom distance signatures (1-WL)?

One round of refinement — hashing the multiset of per-atom sorted
`(Z_j, d_ij)` rows — already kills every classical homometric pair at
essentially the same cost as v4, and was the strongest cheap upgrade.
We went to the full canonical form because 1-WL is measurably not
complete: a 30,000-sample random search over small integer lattices
found two 5-point configurations with identical per-atom signature
multisets that are provably non-congruent (verified by exhaustive
search over all 120 atom pairings). Both are regression fixtures in
`tests/test_collisions.py`, and both are distinguished by the canonical
matrix. Since the canonical scheme subsumes the 1-WL scheme (refinement
is its first stage), shipping it directly also avoids a second hash
migration later.

### Rejected alternatives

- **Unanchored inertia frame** (translate to centroid, rotate to
  principal axes, sort atoms, hash coordinates): O(N) after a 3×3
  eigendecomposition and a complete invariant *when it works*, but the
  axes are undefined exactly where chemistry is interesting. In our
  battery it failed permutation/rotation invariance outright for
  benzene, methane, cubane, and rings (degenerate inertia tensors), and
  near-degeneracy is numerically explosive: a distorted benzene with a
  2·10⁻⁴ Å symmetry break flipped hashes for 97% of 10⁻⁷ Å
  perturbations. Distance noise stays local; frame noise is global.
- **Eigenvalue spectra** (distance/Coulomb matrix): permutation-
  invariant but O(N³), provably lossy (N eigenvalues cannot encode
  N(N−1)/2 distances), and noise is globalized — every eigenvalue moves
  when one atom moves, tripling the observed flip rate at 10⁻⁶ Å noise
  in our tests.
- **Moment/USR-style summaries**: built for similarity screening;
  collisions by design.

### Cost

The canonical search is pure NumPy and is *faster* than the v4 Python
tuple sort for generic molecules (≈1.6 ms vs 2.4 ms at N=100; 170 ms vs
486 ms at N=1000 on a laptop). Symmetric molecules pay for orbit
branching (benzene ≈1.4 ms, cubane ≈8 ms, a perfect 120-atom
monoelemental ring ≈300 ms) — acceptable for a per-geometry hash, and
real conformers rarely have exact rounded symmetry.

### Degenerate-rounding fallback

If rounding is so coarse that many atoms become mutually
indistinguishable (e.g. a cluster hashed at a precision exceeding its
diameter), the branching tree can explode combinatorially. The search
therefore visits at most 10,000 partition states — a normative constant
of the descriptor version. The tree size is permutation-invariant, so
the cutoff is a deterministic property of the geometry, and such inputs
fall back to hashing the stable-WL per-atom signature multiset under a
distinct descriptor tag (`W:` instead of `C:`), which prevents any
cross-path collision. The fallback is still strictly stronger than the
v4 multiset.

## The default canonical frame method

Version 0.8 introduced an opt-in principal frame for systems too large for
the O(N²) distance matrix. Version 0.9 makes `method="frame"` the default
and resolves its former symmetry failure without random perturbations.

The second moment is used only for the subspaces it determines reliably.
Three separated eigenvalues give the original one-frame O(N log N) path.
With a two-dimensional degenerate eigenspace, the isolated axis is retained
and the farthest canonically keyed projected atom anchors the ambiguous
plane. With three degenerate moments, a canonically keyed non-collinear atom
pair constructs the frame. Every invariantly tied anchor is evaluated and
the smallest sorted coordinate-row descriptor wins. Point-like and linear
sets are serialized in zero and one intrinsic dimensions, so arbitrary null
axes are never invented.

Anchors affect only frame construction: the original coordinates are always
projected and hashed. All eight axis signs are evaluated, retaining parity
invariance without handedness conventions. A 10-grid-unit minimum anchor
length prevents sub-precision noise from defining a global orientation, and
a 10,000-candidate budget bounds symmetry enumeration. Either condition
selects the canonical distance method with a warning.

Generic cost is O(N log N) time and O(N) memory. An axial degeneracy with M
tied anchors costs O(M N log N); a fully degenerate tensor with M tied atom
pairs has the same candidate-linear cost and can reach the budget. Exact
large symmetric shells are therefore the main performance failure mode.
Near conditioning thresholds and coordinate rounding boundaries remain the
main numerical failure modes; the distance fallback is retained for them.
Frame and distance bodies use distinct `F:` and `C:`/`W:` tags.

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

The precision must be a power of ten no greater than 1 Å. The frame method
quantizes canonical-frame coordinates; the distance method quantizes pair
distances. Both use `-log10(precision)` decimal places. Two conformers that
differ by less than `precision` may collide; two that differ by more will
usually not.

There is a residual rounding-boundary risk: a distance of exactly
`x.xxxx5` may round to two different values depending on numerical
noise. Choose `precision` generously larger than the geometric noise
floor of your pipeline. The canonical distance method keeps this damage
local — only distances sitting near a bin edge are at risk. A frame rotates
all coordinates together, so its noise is global; the eigenvalue-gap and
anchor-conditioning thresholds bound that risk, but callers needing maximum
rounding robustness can still request `method="canonical"`.

## Dependencies

The implementation uses only NumPy and the Python standard library.
Dropping the RDKit dependency simplifies installation in HPC
environments and removes a non-trivial transitive-dependency surface.
