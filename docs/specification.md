# HashMol3D Specification v0.15.0

**Status:** Proposed standard (draft), developed using the HashMol3D library
**Canonical algorithm:** SHA-256
**Descriptor version:** `11.0`

HashMol3D is a deterministic identifier for 3D molecular conformers.
It is designed for reproducible identification of geometries in
computational-chemistry workflows and databases.

## 1. Identifier format

A HashMol3D identifier is a single ASCII string with six `_`-separated
fields:

    hm3d_v<major version>_p<decimals>_<Hill formula>_<state tag>_<geometry hash>

For example: `hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc`.

- **`hm3d`** — fixed prefix.
- **Descriptor version** — `v` followed by the major number of the
  descriptor version (`v11` for `11.0`). The major number increases with any
  change that can alter a geometry hash; the minor number, which appears only
  in the stored descriptor (§6), with compatible changes that alter none, so
  it never changes an identifier.
- **Precision** — `p` followed by the number of decimals of the precision
  grid (`p4` for `10⁻⁴` Å, `p0` for 1 Å). The precision is also hashed
  (§6), so this field is confirmed by the geometry hash. Hashing it
  is necessary: water at 10⁻⁴ Å and the same atoms scaled tenfold at 10⁻³ Å
  have identical integer rows, and only the precision tells them apart.

- **Hill formula** — if carbon is present: carbon, then hydrogen, then the
  remaining elements alphabetically by symbol; if carbon is absent: all
  elements, hydrogen included, alphabetically. A count of 1 is omitted.
  Examples: `H2O`, `C6H6`, `CHBrClF`, `H3N`, `ClH`.
- **State tag** — `q<charge>m<multiplicity>`. Zero and positive charges
  have no sign (`q0`, `q1`); only negative charges carry a leading `-`
  (`q-1`, `q-2`). Multiplicity is a positive integer with no sign.
- **Geometry hash** — lowercase hex truncation of the SHA-256 digest of
  the hashed payload `P:...|Z:...|F:...` or `P:...|Z:...|D:...` (see §4,
  §6, §7).
- **`_`** — separator. No field contains `_`, so splitting at every `_`
  recovers the six fields. Identifiers use only letters, digits, `_` and `-`
  (in a negative charge), the portable file-name character set, so they are
  safe as file names. The geometry hash comes last, so states of one geometry
  can be grouped by suffix match.

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
See the manuscript's finite-grid and stability analysis for the empirical
rate, and §3 for the resolution floor below which float64 roundoff alone
changes identifiers.

The geometry hash is also **not** invariant under:

- changes in any atomic number Z (isotopes share Z and therefore hash alike)
- changes in the precision, or in the descriptor's geometry algorithm
  (which change the hashed integers; the version and precision fields of
  the identifier record both)
- geometric distortions that move the quantized representation to a
  different grid cell (a distortion larger than the chosen precision need
  not do so, and a smaller one can)

The default frame signature is furthermore **complete on its coordinate
grid**: equal descriptors contain the same sorted element-labelled
coordinates in an orthonormal canonical frame. The optional
distance signature is complete on its rounded-distance grid. Completeness
concerns each method's specified *quantized* representation: quantization
can merge distinct unrounded geometries into one grid cell. Subject to
that quantization, homometric configurations — distinct geometries with
the same distance *multiset* — receive distinct hashes (except for a
truncated SHA-256 collision, §7). Search exhaustion produces no hash (§4.4).

The **state tag** (and therefore the full identifier) additionally
changes with charge or multiplicity. Two states of the same geometry
share the same geometry hash but differ in their state tag, so they can
be grouped by suffix match on the geometry hash, the last field.

> **Note on chirality.** Two enantiomers share the eigenvalues of the
> non-relativistic Hamiltonian and therefore share the same HashMol3D
> identifier. If you need to distinguish enantiomers, store an external
> stereochemistry tag alongside the HashMol3D identifier; the identifier
> itself carries no connectivity or stereochemistry (§9).

## 3. Inputs

The reference implementation takes:

1. `atomic_nums`: integer array of atomic numbers, shape `(N,)`
2. `coords`: float array of Cartesian coordinates in Å, shape `(N, 3)`
3. `precision`: geometry-grid precision in Å (default `1e-4`); it must be a
   power of ten no greater than 1 Å (`1.0`, `1e-1`, `1e-2`, ...), and no
   finer than about `2^-35·R` (≈ 2.9×10⁻¹¹·R), where `R` is the largest
   distance of an atom from the Z-weighted centroid (the exact rule is below)
4. `charge`: total formal charge (default `0`)
5. `multiplicity`: spin multiplicity (default: inferred from electron parity)
6. `length`: geometry-hash length in hex characters, `[1, 64]` (default 32; §7)
7. `method`: `"frame"` (default) or `"distance"` (`"canonical"` is a
   deprecated alias)
8. `node_budget`: canonical search budget (default 10,000; §4.4)

All arguments after `coords` are keyword-only in the reference
implementation.

**Resolution floor.** Float64 coordinates of extent `R` carry about
`R·2^-52` of representation error, which the centroid, tensor, eigenvector
and distance computations amplify. Below a relative grid of about 10⁻¹¹
frame identifiers stop being invariant under rigid motions (0.5–3 % of
motions change the identifier at `precision/R ≈ 10⁻¹¹`, most of them near
10⁻¹³·⁵ on random 30-atom clouds of 1–500 Å); distance identifiers hold
to about 10⁻¹²·⁵. Inputs finer than about `2^-35·R` are therefore rejected
for both methods, and the error names the power of ten
`10^ceil(log10(2^-35·R))`. The decision itself is made exactly, on what is
hashed:

- For an `F` descriptor the decision is exact and uses the hashed rows: the
  input is rejected if any row `(x, y, z)`, in grid cells, has
  `x² + y² + z² > 2^70`. Every geometry with the same hashed rows, including
  a `descriptor_geometry` reconstruction (§6.1), therefore receives the same
  decision; a test on the raw coordinates would put a sharp boundary inside
  the cell that a reconstruction can cross.
- For a `D` descriptor the decision is likewise exact and uses the hashed
  distances `q_ij` (grid cells). With weights `w_i = Z_i` and `W = Σ w_i`,
  atom `i` lies `√A_i / W` cells from the weighted centroid, where
  `A_i = W · Σ_j w_j q_ij² − Σ_{j<k} w_j w_k q_jk²`; the input is rejected if
  any `A_i > W² · 2^70`. A stored `D` section can therefore be checked
  against the same rule (§6.1).
- Before either, inputs with `precision · 2^35 · (1 + 2^-10) < R` are
  rejected outright. No input that either exact rule accepts can reach this
  screen: a hashed row differs from the raw extent by less than one cell, and
  rounded distances change `A_i / W²` by at most about `3 · 2^35` cells²,
  far below the `2^-9 · 2^70` cells² of slack.

`R` is a rigid-motion and relabeling invariant; a single atom has no floor.
This rejects only grids that float64 cannot support; it does not change any
identifier. It also bounds every fine-grid frame integer (§4.5) below
`2^53`.

## 4. Geometry signatures

The default frame method is specified in §4.5. Sections §4.1–4.4 specify
the optional labelled-distance method (`method="distance"`) and the final fallback used
when a stable bounded frame cannot be constructed.

For the distance method, translation, rotation, and reflection
invariance come from using only pairwise distances. Permutation invariance
comes from writing the element-labelled distance matrix in a **canonical
atom order** that is a pure function of the geometry. Because the full
labelled distance matrix determines the point set up to congruence, two
geometries receive the same `D` signature if and only if they are congruent
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
3. Each leaf (discrete partition) yields an atom order; evaluate the upper
   triangle of `q` in that order, row-major, as a sequence of integers, and
   keep the one that is smallest element by element (numeric, not string,
   comparison).

The set of leaves explored is a function of the geometry alone, so the
winning order — and therefore the signature — is permutation-invariant.
There is no automorphism pruning, so the number of leaves is at least the
order of the geometry's rounded-distance symmetry group (1 for generic
molecules, 24 for a perfect tetrahedral cluster, 2n for an ideal n-ring),
and equals it when refinement separates every non-equivalent atom.

### 4.4 Canonical search budget

The default search budget is **10,000** visited partition states; callers may
set any positive integer `node_budget`. If the full search cannot finish
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
3. Otherwise require `λ3 > 0` (else select the distance method, as in
   step 7). Before rejecting a cloud with
   `R s < 10`, where `R = max_i ||v_i||`, test collinearity about the
   largest-moment eigenvector `u3`. Let `rho = max_i ||v_i - (v_i·u3)u3||`.
   If `rho ≤ 64 * eps64 * R`, with `eps64 = 2^-52`, serialize the
   intrinsic line `(Z, 0, 0, a_i)` with axial coordinate `a_i = v_i·u3`,
   quantized and rounded as in steps 8–9, minimizing over both axial
   signs. This accepts a line up to float64 roundoff, not an arbitrary
   sub-grid bend. Continue to the gap test for other clouds, including those
   with `R s < 10`.
4. Define relative gaps `g1=(λ2−λ1)/λ3` and `g2=(λ3−λ2)/λ3`.
   If both are at least **0.05**, use the eigenvectors in ascending
   eigenvalue order, regardless of `R s`: no atom anchor is needed.
   Otherwise, clouds with `R s < 10` use §4.1–4.4 with a warning. This
   whole-cloud size guard prevents a short finite bend from entering the
   half-grid intrinsic-line rule below. The factor 10 is a fixed policy,
   not an input-noise estimate or a mathematical requirement for a
   principal-axis frame.
5. If exactly one gap is below 0.05, preserve the isolated eigenvector
   `u`. For every atom form its projection into the degenerate plane,
   `p_i = v_i − (v_i·u)u`. If the isolated axis is the largest-moment
   axis and `max_i ||p_i||s < 0.5`, serialize a one-dimensional line:
   `(Z, 0, 0, v_i·u)`, quantized and rounded as in steps 8–9, considering
   both axial signs. Otherwise
   require `max_i ||p_i||s ≥ 1` and select the lexicographically largest
   invariant anchor key

       (rint(||p_i||s), Z_i, rint(|v_i·u|s)).

   Evaluate every tied anchor. Its normalized `p_i` resolves the ambiguous
   plane; a cross product supplies the remaining axis. The isolated axis
   stays in its ascending-eigenvalue slot. One grid unit is the smallest
   transverse extent that survives quantization; a shorter anchor does not
   amplify noise, because a perturbation `δ` rotates the frame about `u` by
   about `δ/rho` while every transverse coordinate is itself bounded by
   `rho`, so coordinates move by at most about `δ`.
6. If both gaps are below 0.05, choose first anchors by the largest key

       (rint(||v_i||s), Z_i).

   For every tied first anchor `i`, set `e1=v_i/||v_i||`, project every
   other atom perpendicular to it, and choose second anchors by the largest
   key

       (rint(||p_ij||s), Z_j, rint(|v_j·e1|s)).

   If, for any tied first anchor, `max_j ||p_ij||s < 1`, or no atom has
   `rint(||p_ij||s) > 0`, select the distance method (step 7). Otherwise
   evaluate every tied ordered pair, with second anchors restricted to
   `rint(||p_ij||s) > 0`. Normalize `p_ij` as `e2` and set `e3=e1×e2`.
7. At most **10,000** tied candidate frames may be evaluated. Exceeding
   this normative budget, or finding a required transverse extent below
   1 grid unit (steps 5–6), deterministically selects the distance method
   of §4.1–4.4 with a `UserWarning`.
8. For every candidate basis and for the intrinsic line, project the **original centered coordinates**
   (anchors never perturb the geometry) and quantize them **once**, on the
   fine grid `s' = 10^(decimals+4)`, with `rint`. Derive each hashed
   coordinate from its fine integer `f` by exact integer round-half-to-even
   division by `10^4` (never by rounding the coordinate at `s` directly:
   rounding twice differs from rounding once at exact half-cells).
   Evaluate all eight axis-sign combinations, sort the rows by their hashed
   values `(Z,x,y,z)` (fine values break ties between equal hashed rows),
   and keep the globally smallest row list, comparing hashed rows first and
   fine rows only when the hashed rows are equal. The sign set includes
   both parities, making the descriptor reflection-invariant. Anchor keys in
   steps 5–6 remain on the precision grid `s`.
9. Before serialization, move every stored fine value `f` with hashed value
   `h` into `[h·10^4 − 4997, h·10^4 + 4997]`. The stored geometry then keeps
   at least three fine units from every rounding boundary of the hashing grid,
   so a re-hash of it is not decided by an exact tie. This changes neither
   the hashed values nor the order of rows that differ in them.

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

    V:<version>|P:<precision>|Z:<z_ordered>|D:<distances>     (distance method)
    V:<version>|P:<precision>|Z:<z_rows>|F:<coordinates>      (frame method, §4.5)

Where:

- `<version>` is the descriptor version `<major>.<minor>`, e.g. `11.0`
  (versions 5–10 were written `<n>-CANON-SHA256` or `<n>-FRAME-SHA256`,
  see §8, and used `C` for the
  distance section).
- `<precision>` is in scientific notation, e.g. `1.0e-04`.
- `<z_ordered>` is the list of atomic numbers in canonical atom order,
  comma-separated (this always coincides with the ascending-sorted
  multiset, because the initial colors of §4.2 rank atoms by Z).
- `<distances>` is the upper triangle of the scaled integer distance
  matrix `q` (§4.1) in canonical atom order, row-major
  (`q_12, q_13, ..., q_1N, q_23, ...`), comma-separated. Empty for a
  single atom.
- `<coordinates>` (frame method only) is the winning sorted row list of
  §4.5, each row formatted as `Z:x,y,z` with the **fine** integer
  coordinates of §4.5 steps 8–9 (multiples of `precision·10^-4`), joined
  by `;`; `<z_rows>` lists the atomic numbers in that row order.

Charge and multiplicity are **not** part of the descriptor; they are
written into the state tag of the identifier instead.

This is the **stored descriptor** returned as `HashMol3DResult.descriptor`.
The **hashed payload** is the stored descriptor without its `V` field, with
every `F` coordinate replaced by its hashed value (exact round-half-to-even
division by `10^4`); `D` sections are hashed as stored:

    P:<precision>|Z:<z_rows>|F:<hashed coordinates>
    P:<precision>|Z:<z_ordered>|D:<distances>

The `V` field is not hashed: its major number appears in the identifier as
`v<major>`. The precision is hashed, so a bare geometry hash identifies a
geometry on one grid, and a stored descriptor whose `P` field was edited no
longer re-hashes to its digest. (Versions 5–10 hashed the full
`V|P|Z|<section>` string.)

Example using the default frame method (water, `precision = 1e-4`):

    V:11.0|P:1.0e-04|Z:1,1,8|F:1:0,-46880000,-75720000;1:0,-46880000,75720000;8:0,11720000,0

Its hashed payload, whose SHA-256 digest truncated to the default 32 hex
characters is the geometry hash `8e9eedf5032cf387541d637f56c682dc`:

    P:1.0e-04|Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0

The identifier is `hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc`.

(The molecular-plane normal occupies the first axis. The two hydrogen rows
precede oxygen after lexicographic sorting.)

### 6.1 Idempotent re-hashing

A stored descriptor determines its identifier without the source geometry:
re-hashing it (normalize, then §7) reproduces the geometry hash exactly,
for `F` and `D` descriptors and for stored descriptors of earlier versions
(which hash their full `V|P|Z|<section>` string, and whose `F` rows before
version 10 are already on the precision grid). The reference
implementation provides this as `hash_descriptor(descriptor, length=...)`.
The descriptor determines the identifier's version, precision and formula
fields; only the state tag (charge and multiplicity) is not in the
descriptor and must be stored alongside it.

A re-hashed descriptor must satisfy every rule its own version's generator
enforced; anything else is rejected rather than hashed. The checks are:

- a known version, with the sections it produced: `5-CANON-SHA256` and
  `6-FRAME-SHA256` with `F`, `C` or `W` (release 0.7.0 wrote only `C` and
  `W` under version 5; 0.8.0 added `F`), `7`–`10-FRAME-SHA256` with `F` or
  `C`, `11.0` with `F` or `D`;
- the precision written exactly as that version rendered it: any positive
  finite value for version 5 (which snapped its grid to the nearest power of
  ten, but no coarser than 1 Å, and recorded the precision as given, e.g.
  `P:5.0e-02`), a power of ten ≤ 1 Å from version 6;
- atomic numbers in `[1, 118]` and in ascending order;
- integers in canonical decimal form (no sign on zero, no `+`, no leading
  zeros) with absolute value below `2^62`;
- `F` rows: one per atom, elements equal to `Z`, in canonical row order;
- `D` (or `C`) sections: exactly `N(N−1)/2` non-negative distances;
- the RDKit `S` extension of versions 8–10 (§9): validated as its base
  descriptor plus a non-empty `S:` field;
- `W` sections (versions 5–6): `N ≥ 2` and one `Z,color:color-distance,...`
  row per atom, rows sorted, with elements equal to `Z` and, in each row, the
  sorted `(color, distance)` pairs of the other `N − 1` atoms.

From version 10, every `F` value must also lie within its rounding margin
(§4.5 step 9) and every descriptor must satisfy the exact resolution floor of
§3, on its `F` rows or `D`/`C` distances. These are necessary conditions, not
a proof of origin: they do not check that distances are
realizable in three dimensions or that `W` rows come from one geometry.

`descriptor_geometry` accepts only `F` descriptors of the current version; it
raises `ValueError` for `D` and `C` descriptors, which have no coordinates,
and for earlier versions (including their `S` descriptors).

The `F` rows are also a canonical-frame geometry, centered on the
Z-weighted centroid, on the fine grid. Hashing them as coordinates at the
descriptor's precision recomputes the frame, which shifts each value by
about one fine unit; a value that close to a rounding boundary of the
hashing grid can change cell. `descriptor_geometry(descriptor)` therefore
re-hashes the stored rows before returning them and, if the identifier
differs, settles them deterministically: in the frame the re-hash chose,
values within 50 fine units of a boundary are pinned 50 units inside their
target cell, and the other values take the minimum-norm change that keeps
the weighted centroid and gyration tensor unchanged (three Gauss–Newton
steps), for at most eight rounds. The result is a pure function of the
descriptor and is not part of the hashed payload, so it never affects an
identifier.

Measured on all 133,885 QM9 molecules at 10⁻², 10⁻³ and 10⁻⁴ Å, every
`descriptor_geometry` result hashed back to the original identifier, also
after a random rigid motion and relabeling and after a second round trip;
for 8–22 molecules per precision (at most 0.02 %) the raw stored rows alone
did not. Settling moved
coordinates by at most 5.4 % of a grid cell (0.5 % in all but one case).
For the version-9 rows read back as coordinates the round-trip rate is
54–56 %. Exactly symmetric inputs with tied anchor frames are the remaining
exception: a scan of 7,010 exactly symmetric `F` cases (benzene, methane, SF₆,
NH₃, C₅H₅, cubane, each at 301 scales and four precisions: 7,224 inputs,
of which 214 NH₃ inputs at 0.1 Å select the distance method) left 5 that
could not be settled, because the lexicographic minimum over tied anchors
selects whichever frame moves a value across a boundary. In that case
`descriptor_geometry` warns and returns the stored rows; `hash_descriptor`
remains exact.

## 7. Hashing

1. Encode the hashed payload (§6, `P:...|Z:...|<section>`) as UTF-8 bytes.
2. Compute the SHA-256 digest.
3. Take the first `length` hex characters of the hex digest.

`length ∈ [1, 64]`. When not explicitly supplied, the reference
implementation uses a fixed default of 32 hex characters (128 bits).
Collision resistance is a property of the namespace, not of molecule
size: for `n` distinct geometries hashed into `b = 4·length` bits, the
expected number of birthday collisions is `~ n² / 2^{b+1}`. The 128-bit
default keeps that below one for corpora up to ~2.6×10¹⁹ geometries, and
below 10⁻⁹ up to ~8.2×10¹⁴. Callers
who know their corpus size may pick `length` accordingly (the reference
implementation provides `hash_length_for(n_items, target_prob)`), or pin
any fixed value in `[1, 64]`.

## 8. Determinism and portability

To guarantee identical identifiers across machines:

- Use the same descriptor version and precision. Geometry hashes from
  different major versions can coincide when a geometry body is unchanged,
  so compare complete identifiers, which carry the version, across
  versions.
- Format atomic numbers, scaled distances, and `F` coordinates as decimal
  integers with no leading zeros, `+` sign, or signed zero; only negative
  `F` coordinates carry a leading `-`.
- Quantize with round-half-to-even (IEEE 754 `rint`), as in §4.1, and
  derive hashed frame coordinates from the fine integers in exact integer
  arithmetic (§4.5 step 8).
- Complete the canonical search (§4.4); the node budget only controls whether
  a result is available. For the frame method,
  use the relative eigenvalue-gap threshold 0.05, whole-cloud size guard
  10 grid units, minimum transverse-anchor extent 1 grid unit, and
  candidate budget 10,000 exactly (§4.5).
- Encode the descriptor in UTF-8 before hashing.
- Use SHA-256 as defined in FIPS 180-4.
- Render the formula in Hill order and the state tag exactly as in §1.

Any change to the descriptor format or semantics requires a new descriptor
version: a new major version if any geometry hash can change, otherwise a new
minor version.

Version 5 introduced the `V|P|Z|<section>` layout, with a `W`
(Weisfeiler–Lehman multiset) section when the canonical search exhausted its
budget; release 0.8.0 added the optional frame method (`F`). Version 6 made
the frame method the default and required a power-of-ten precision ≤ 1 Å.
Version 7 introduced the early collinearity check for small nonpoint clouds,
and replaced the `W` fallback with `SearchBudgetExceeded`.
Version 8 additionally accepts separated principal moments before applying the
minimum extent for atom anchors. This changes some coarse-grid requests from
`C` to `F` (including water at 0.1 and 1 angstrom); the anchor and gap thresholds
are unchanged.
Version 9 lowers the minimum transverse-anchor extent from 10 grid units to 1
(§4.5 steps 5–7). Nearly linear molecules whose bend lies between half a cell
and ten cells previously selected the distance method; they now anchor their
degenerate plane and produce `F` descriptors, which round 3N values instead of
N(N−1)/2 and are correspondingly less sensitive to coordinate noise. The
whole-cloud size guard (step 4), the gap threshold, the candidate budget, and
every serialized body of an unaffected geometry are unchanged. Because the
version field is hashed, **every version-9 digest differs from earlier
versions**, even when the geometry body is unchanged. Recompute a corpus
consistently when migrating; identifiers from different versions must not be
mixed.
Version 10 stores `F` rows on a grid `10^4` times finer than the precision
and hashes them rounded back to it (§4.5 steps 8–9, §6), so a stored
descriptor carries a canonical geometry that re-hashes to its own identifier
(§6.1). Deriving the hashed values from the fine integers, rather than
rounding at the precision directly, changes the hashed rows of about 0.3 % of
QM9 molecules (371 of 133,885 at 10⁻⁴ Å) relative to version 9; all other
hashed rows are identical. Version 10 also rejects precisions below the
float64 resolution floor (§3), decided exactly on the hashed values, and
validates stored descriptors before re-hashing them (§6.1). As with every
version, all digests change.
Version 11 (`11.0`) introduces the
`hm3d_v<major>_p<decimals>_<formula>_<state tag>_<geometry hash>` identifier
(§1). Its hashed payload omits the `V` field (§6): the version moves from the
hash into the identifier, while the precision stays hashed and is also shown
in the identifier. It renames the distance section from `C`
to `D` (`method="distance"`; `"canonical"` remains a deprecated alias) and
removes the RDKit S extension (§9). The geometry algorithm and every stored
`F`/`D` body are unchanged from version 10 except for the `V` field and the
section letter, but every geometry hash changes.

## 9. Connectivity and the historical S extension

The identifier carries no connectivity, stereochemistry, isotope or SMILES
information: the geometry hash covers only atomic numbers and geometry. RDKit
input adapters ([RDKit input](rdkit.md)) only read structures.

Versions 8–10 offered an opt-in RDKit `S` extension that appended canonical
isomeric SMILES to the hashed descriptor:
`V:<base>-S1-RDKIT-<rdkitVersion>|P|Z|<F or C>|S:<SMILES>`, hashed with its
`F` rows rounded as in the base hash. Version 11 removed it. Stored `S`
descriptors still re-hash to their original digests (§6.1); new identifiers
cannot be made with it.

## 10. Dependencies

The default implementation uses only NumPy and the Python standard library.
RDKit is an optional extra (the `rdkit` extra of the package) used only when
the caller explicitly requests an RDKit adapter or SMILES generation.

## 11. Reference API

```python
from hashmol3d import hash_molecule

result = hash_molecule(
    atomic_nums,
    coords,
    precision=1e-4,
    charge=0,
    multiplicity=None,  # inferred if None
    length=None,  # 32 hex (128-bit) if None
    method="frame",  # default; use "distance" for the distance method
    node_budget=10_000,  # canonical search budget (§4.4)
)
print(result.hash_str)  # hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc
print(result.geometry_hash)  # 8e9eedf5032cf387541d637f56c682dc
```

Re-hashing a stored descriptor (§6.1):

```python
from hashmol3d import descriptor_geometry, hash_descriptor

assert hash_descriptor(result.descriptor) == result.geometry_hash  # exact
z, xyz = descriptor_geometry(result.descriptor)  # canonical-frame geometry
assert hash_molecule(z, xyz).hash_str == result.hash_str
```

A file-based convenience wrapper is also provided:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)
```
