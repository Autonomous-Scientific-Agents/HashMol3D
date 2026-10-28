# Design note: an idempotent quantized geometry in the F descriptor

Status: implemented as descriptor version 10 (`10-FRAME-SHA256`, package
0.14.0); version 9 went to the near-linear anchor threshold. Rules 1–3 are
implemented as proposed, with additions measured on all of QM9 that are
described in §4. Nothing here applies to version 8, whose hashed form and
digests are frozen by the manuscript. Sections 1–3 are the original proposal.
Version 11 kept this design unchanged but renamed the distance section from
`C` to `D` and dropped `V` and `P` from the hashed payload (they moved into
the `hm3d_v<major>_p<decimals>_…` identifier); `C` below means the version
10 distance section.

Measurements below come from exploratory runs on random element-labeled
clouds and on a QM9 sample (about 5,000 molecules taken every 27th entry),
default grid ε = 10⁻⁴ Å, frame method. Sample sizes are given with each
number. They are indicative and have no harness in either repository yet; a
harness must be added before any of them is quoted in a publication.

## 1. The problem

A corpus reformatted into "descriptor form" wants, for every entry, a stored
representation from which the identifier can be re-derived. Two things are
currently true:

1. The identifier is `prefix-SHA256(descriptor)`. Re-hashing the stored
   descriptor string therefore reproduces the identifier exactly, for `F` and
   `C` entries alike. This is the only idempotent representative today.
2. The `F` descriptor's rows (canonical-frame coordinates quantized at ε) are
   a geometry, and one would like to be able to use them *as coordinates*:
   feed them back into `hash_molecule` and recover the same identifier. This
   fails about half the time.

| input re-hashed at ε = 10⁻⁴ Å | fraction giving the original identifier |
| --- | --- |
| original coordinates (any orientation, any atom order) | 1.00 |
| the descriptor's own rows × ε, random clouds N = 3 / 5 / 10 / 30 / 100 (200 each) | 0.78 / 0.43 / 0.35 / 0.30 / 0.21 |
| the descriptor's own rows × ε, QM9 sample (4,959 molecules, N 3–29) | 0.54 |

Mechanism. Rounding moves each atom by up to ε/2. The weighted centroid and
gyration tensor of the rounded rows therefore differ slightly from the
original's; the off-diagonal perturbation is of order √N·w·R·ε and rotates the
recomputed principal axes by θ ≈ ε / (g · R_rms · √N), where g is the relative
eigenvalue gap (measured ≈ 7×10⁻⁶ rad for N = 100). The outermost atoms move
by R_max·θ, typically 0.3–0.8 of a grid cell. Every row value sits exactly at a
bin centre, so a flip requires a shift above ε/2 on any of the 3N values; the
growing count of values (∝ N) and the shrinking per-value shift (∝ 1/√N)
cancel into the plateau seen above. Failures change 2–30 of the 3N integers
by exactly one grid unit; sign patterns and axis assignments are not involved.
Real molecules do better than random clouds because they are compact (smaller
R_max/R_rms) and have larger gaps.

Two variants were considered and rejected:

* **Quantize the input coordinates first, then compute the frame.** This makes
  the pre-quantized coordinates trivially idempotent but destroys rotation
  invariance: the grid is anchored to the input axes and commutes only with
  the 48 cubic symmetries. Under random rigid motions the descriptor was
  reproduced in 0 % of trials at both 10⁻² and 10⁻⁴ Å (standard construction:
  100 %).
* **Iterate rows → frame → rows until stationary** and hash the fixed point.
  On 1,500 QM9 molecules 99 % reach a fixed point (54 % immediately, 90 %
  within 3 steps, worst 34), 0.7 % enter a 2-cycle and 0.3 % do not settle
  within 40 steps. The fixed-point identifier has exactly the same noise
  sensitivity as the direct one (0.875 / 0.630 / 0.27 unchanged at σ =
  0.003 / 0.01 / 0.03 ε, identical for both). It buys idempotence at the cost
  of a version bump, up to a few dozen frame computations per geometry, and
  two extra rules (cycles, non-convergence), while the same idempotence is
  available for free through the descriptor string.

## 2. The proposed solution: extra digits in the F rows

Serialize the `F` rows with p + 4 decimal digits (10⁻⁸ Å for ε = 10⁻⁴ Å)
instead of p, and define the identifier as the hash of a *normalized* form in
which each row integer is rounded back to p digits. The stored descriptor then
carries a canonical fine geometry; re-hashing it through the algorithm
reproduces the identifier almost always, and normalizing-then-hashing
reproduces it exactly.

### Why four digits and not one or two

| extra digits k in the rows | random clouds N = 10 / 30 / 100 (150 each) | QM9 (1,500) |
| --- | --- | --- |
| 0 | 0.30 / 0.25 / 0.27 | 0.554 |
| 1 | 0.25 / 0.01 / 0.00 | 0.115 |
| 2 | 0.83 / 0.66 / 0.21 | 0.810 |
| 3 | 0.99 / 0.98 / 0.83 | 0.980 |
| 4 | 1.00 / 1.00 / 0.99 | 0.999 |
| 6 | 1.00 / 1.00 / 1.00 | 0.999 |

(fraction of algorithm-level re-hashes at ε reproducing the original
identifier when the input is the k-extra-digit rows read as coordinates)

One extra digit is worse than none: rows at ε sit at bin centres (margin
ε/2), whereas rows on the ε/10 sub-grid include every ε bin *edge* — one value
in ten ends in 5 and is an exact tie decided by the 10⁻¹⁶ residue of the
recomputed frame. At two digits it is one value in a hundred plus the ordinary
rounding shift M·(ε′/2)/ε; at four digits the failure rate is ≈ 0.5 % for a
100-atom cloud and ≈ 0.1 % on QM9. The residual at four and six digits is the
same, so it is not rounding; it is most likely the few molecules whose branch
or sign choice differs between the ε and ε′ constructions (anchor lengths are
measured in grid units).

### Three rules the specification must state

**Rule 1 — the hashed form is a normalization of the stored form.**
`identifier = prefix-SHA256(normalize(descriptor))`, where `normalize`
replaces each `F` row integer `v` (p + 4 digits) by `round_half_even(v / 10⁴)`
computed in exact integer arithmetic. No floating point is involved, so the
normalization is reproducible on every platform, and "re-hash the stored
descriptor" is exact by construction. The `V`, `P`, `Z`, and `C` fields are
unchanged by normalization. Implementations that hash descriptor bytes
directly must be updated; the `V` tag distinguishes the two forms.

**Rule 2 — the ε integers are derived from the fine integers, never computed
directly.** Rounding twice is not rounding once: `rint(x·10⁴)` and
`round(rint(x·10⁸) / 10⁴)` disagree whenever the fine rounding lands exactly on
a half, about 0.5×10⁻⁴ of values. Measured on 3,000 QM9 molecules, deriving
the ε rows from the 10⁻⁸ Å rows and comparing with the direct version-8 rows
disagreed for 97 molecules (3.2 %): 77 by sign pattern (Rule 3), 7 by a single
double-rounding tie, 13 by several values. The definition must therefore be
"quantize at ε′ = ε/10⁴, then derive the hashed integers by Rule 1", so that
two conforming implementations cannot disagree. This is what forces a version
bump: the hashed integers coincide with version 8 for about 97 % of molecules,
but the rule that produces them has changed.

**Rule 3 — sign and anchor minimization is performed on the normalized rows,
with the fine digits as tie-break.** The frame path selects among eight axis
sign patterns (and among tied anchors) by lexicographic minimum over the
sorted rows. Minimizing over fine rows chooses differently from minimizing
over ε rows whenever the ε comparison ties on its leading values — 2.6 % of
QM9 (the 77 above). Minimizing on the normalized rows keeps the hashed
integers identical to a direct-ε choice wherever version 8 had a unique
minimum; breaking exact ties with the fine digits makes the fine rows
canonical as well.

### What the proposal gives and what it does not

Gives:

* exact idempotence at the descriptor level (normalize, then hash);
* a canonical fine-resolution geometry for every `F` entry, usable as
  coordinates downstream, that re-hashes through the algorithm to the same
  identifier in ≈ 99.9 % of real molecules;
* no companion field; the hashed integers, digest length, and prefix are
  unchanged in form;
* cost of four characters per value (≈ 200 bytes for a QM9-sized molecule).

Does not give:

* anything for `C` entries, which have no rows at any precision; their
  idempotent form remains the descriptor string (or the source coordinates);
* exact algorithm-level idempotence — the fine rows still carry rounding error
  that perturbs the recomputed frame; ingestion must re-verify if it re-hashes
  coordinates rather than descriptors;
* any change in noise sensitivity of the identifier, since the hashed integers
  are still at ε (this is the point: the fine digits must never enter the
  hash, or the identifier would inherit 10⁴-times denser boundaries).

### Alternatives for comparison

| representative stored per entry | idempotence | canonical | covers C |
| --- | --- | --- | --- |
| descriptor string (version 8, today) | exact | yes | yes |
| source coordinates as read (never re-rounded) | exact | no | yes |
| rows at ε read as coordinates | ≈ 0.3–0.55 | yes | no |
| coordinates written with ≥ p + 8 decimals | ≈ 1 − M·10⁻ᵈ/2ε | no | yes |
| rows at p + 4 digits, this proposal | exact (normalize) / ≈ 0.999 (algorithm) | yes | no |

## 3. Open questions

1. *Resolved by implementation.* Is a canonical fine geometry needed at
   all, or is "store the descriptor string" sufficient for the corpus
   format? Both are now available: `hash_descriptor` re-hashes the string and
   `descriptor_geometry` returns the fine geometry.
2. *Resolved.* Timing of the digest change: it shipped as version 10 in
   0.14.0 (version 9 was used for the near-linear anchor change). The
   manuscript's corpus results remain version 8 and are labelled as such.
3. *Resolved.* The extra-digit count is the spec constant 4, independent of
   p. The float64 limit is relative, `2^-35·R` (§4), and is enforced
   directly.
4. Whether `C` descriptors should carry an analogous fine distance matrix for
   symmetry with `F`. It would not provide coordinates, so the benefit is
   unclear.
5. Harnesses: the rates above need scripts in the paper repository
   (conventions: `HM3D_DATA`, manifest check, CSV output) before they can be
   cited.

## 4. Implementation notes (version 10)

Measured on all 133,885 QM9 molecules (`gdb9.sdf`) at 10⁻², 10⁻³ and
10⁻⁴ Å unless stated otherwise. The harness is not yet in the paper
repository (open question 5 still applies).

* **Rules 1–3 as proposed.** `hash_descriptor` normalizes and hashes;
  normalization is exact for every molecule. Hashed rows differ from
  version 9 for 361–371 of 133,885 molecules per precision (0.3 %, less
  than the 3.2 % estimated above, because Rule 3 is applied). The normalized form shares the
  stored form's `V` tag, so it is internal (`_hashed_payload`) rather than
  public: normalizing it a second time would round twice.
* **Stored values keep a three-unit margin.** The residual failures at four
  digits were exact ties: stored values ending in `5000`, which a re-hash
  shift of one fine unit decides either way. Each stored value is moved at
  least three fine units inside its coarse cell (hashed values unchanged).
  Re-hash failures of the raw stored rows at 10⁻⁴ Å by margin: 0 → 66,
  1 → 52, 2 → 23, 3 → 22, 5 → 27, 10 → 37, 100 → 334. Large margins are
  worse: moving a value perturbs the recomputed frame by about as much.
* **`descriptor_geometry` verifies and settles.** The remaining failures
  (8–22 molecules per precision) cannot be fixed by clamping a value: the
  recomputed frame rotates to absorb the move, since the frame is defined
  by the same coordinates. The function therefore re-hashes the stored rows
  and, if needed, pins values within 50 fine units of a boundary 50 units
  inside their cell, while the other values take the minimum-norm change
  that keeps the weighted centroid and full gyration tensor unchanged
  (Gauss–Newton). Keeping, rather than zeroing, the tensor matters: in an
  anchored frame its off-diagonal terms are legitimately non-zero. Settling
  continues in the frame the re-hash chose, for at most eight rounds. With
  it, every QM9 round trip reproduced the identifier at all three
  precisions, also after a random rigid motion and relabeling and after a
  second round trip; so did 2,000 random clouds (N = 3–100) per precision.
  The largest settling move was 5.4 % of a cell. For comparison, version-9
  rows read back as coordinates reproduce the identifier for 54.4–55.8 %.
* **Known limit: exact symmetry with tied anchors.** Of 7,010 exactly
  symmetric `F` cases (benzene, methane, SF₆, NH₃, C₅H₅, cubane, each at 301
  scales and precisions 10⁻¹–10⁻⁴ Å; the other 214 of the 7,224 inputs, NH₃
  at 0.1 Å, select `C`) 5 could not be settled: the lexicographic minimum over tied anchor
  frames picks whichever frame moves a value across a boundary, and settling
  one frame does not constrain the others. `descriptor_geometry` then warns;
  `hash_descriptor` is exact regardless.
* **Fine digits are not invariant.** Under rigid motions the stored fine
  digits can differ by a unit when a frame is anchored on a short transverse
  extent (e.g. 1.5 cells), as expected from their 10⁴-times denser grid.
  Hashed values are unaffected; invariance tests compare hashed payloads.
* **Precision floor.** Fine rows must fit in int64 four decades sooner
  than coarse ones, so the finest accepted precision dropped by four
  decades. Nothing usable was lost: in version 9, at every precision now
  past the int64 limit, 0–5 % of rigid motions kept the identifier, because
  float64 roundoff (about `R·2^-52`, amplified by the frame computation)
  dominates there. The real limit is float64, so `hash_molecule` now rejects
  `precision < 2^-35·R` (`R` = largest distance from the Z-weighted
  centroid) for both methods. Measured on random 30-atom clouds of 1–500 Å,
  with and without a translation of 3R: frame identifiers change for
  0.5–3 % of motions at `precision/R ≈ 10⁻¹¹` and for most near 10⁻¹³·⁵;
  distance identifiers hold to about 10⁻¹²·⁵. At the finest accepted
  power of ten, 0 of 3,600 motions changed a frame or distance identifier,
  and 1,000 of 1,000 `descriptor_geometry` round trips reproduced it. The
  floor also keeps every fine frame integer below `2^53`.
  For `F` descriptors the decision is made exactly on the hashed rows
  (`x² + y² + z² ≤ 2^70` cells²), not on the raw extent. A raw-extent test is
  a sharp boundary that reconstruction can cross: in review, geometries
  placed within a few ulps of it hashed, reconstructed, and the
  reconstruction was then rejected (376 of 2,000). Decided on hashed rows,
  every verified reconstruction gets the same decision (0 of 10,533
  accepted boundary inputs failed). `C` descriptors are decided the same
  way, on the hashed distances: the squared distance of each atom from the
  weighted centroid follows exactly from them in integer arithmetic. A bound
  of twice the floor on single distances, used first, let unequal weights
  through (`Z:1,118|C:51539607552`, whose hydrogen lies past the floor). A
  raw screen with 2⁻¹⁰ slack still runs first, to bound work and integer
  sizes.
* **Descriptor validation.** `hash_descriptor` and `descriptor_geometry`
  check each descriptor against the rules of its own version (version table
  with the sections each release wrote, including 0.8.0's version-5 `F` and
  the `W` fallback of versions 5–6; the precision as each version wrote it;
  canonical integers; Z range and order; row/Z consistency; canonical row
  order; distance count; int64 bounds; for version 10 also the rounding
  margin and the exact floor). Before this, `P:invalid` or a `C:` body that
  was not a distance list hashed, and an `F` row could contradict `Z`.
  Descriptors emitted by all 17 revisions of `core.py` from version 5 to 9
  (QM9 sample, random clouds, symmetric and linear cases, odd version-5
  precisions, forced `W` fallbacks) re-hash to their original digests.
  Validation makes `hash_descriptor` about 2–3× slower (≈ 85 µs per QM9 `F`
  descriptor, ≈ 120 µs per `C` descriptor).
* **Cost.** Hashing is about 1.6× slower than version 9 on QM9-sized
  molecules (236 vs 146 µs) and 1.5× for a 5,000-atom cloud; verification in
  `descriptor_geometry` costs one more frame computation.

