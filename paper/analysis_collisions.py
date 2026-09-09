"""
Large-scale collision study for HashMol3D over popular public datasets.

We hash every geometry in two widely used datasets and check for hash
collisions:

  * QM9 -- 134k distinct small organic molecules (equilibrium geometries),
    exercising composition/size/element diversity across molecules.
  * MD17 -- molecular-dynamics trajectories of benzene, aspirin, and ethanol
    (up to ~10^6 conformers of a *single* molecule each), exercising the hard
    case of many near-duplicate geometries of the same molecule.

Collision bookkeeping. We compute the full 64-hex (256-bit) SHA-256 geometry
digest for every geometry. The current streaming analysis does not retain full
descriptors, so equal 256-bit digests could mean either equal descriptors or a
full SHA-256 collision; it cannot distinguish the two. No equal full digest was
observed in the reported corpus. A definite truncation collision is two
*distinct* full digests that share an L-hex prefix. Working on the set U of
distinct full digests, the count at length L is the *excess-item* count
n - |{ d[:L] : d in U }| (items beyond one per occupied prefix), where
n = |U|. The default is L = 32 (128 bits).

Expected value. For n items hashed into 2^b truncated slots by an ideal random
hash, the expected number of *occupied* slots is 2^b[1 - (1 - 2^-b)^n], so the
expected excess-item count is the exact occupancy expression

    E[n - U] = n - 2^b[1 - (1 - 2^-b)^n].

This is the statistic the observed n - U should be compared against at every
length. The sparse-limit birthday approximation n(n-1)/2^{b+1} counts colliding
*pairs*, a different quantity that agrees with the excess-item count only in the
sparse tail (few collisions) and diverges badly once slots saturate (short L).
We plot the exact occupancy expectation as the primary reference and show the
sparse-limit pair count as a secondary curve to make the distinction explicit.

Descriptor-path audit. The default ``frame`` run records the principal-axis,
one-axis/atom-anchor, two-atom-anchor, point, line, and canonical-fallback
branches as well as the emitted ``C`` / ``W`` / ``F`` descriptor tags.  This
makes the behavior at degenerate principal moments observable rather than
inferring it from successful hashes.  Set ``HM3D_METHOD=canonical`` to rerun
the retained distance-matrix alternative.

Reproducibility. In the default (reproduction) mode every dataset listed in
MANIFEST must be present, and -- when an expected geometry count is recorded --
the loaded count must match it; otherwise the run aborts. Partial collections
require the explicit ``--allow-partial`` flag and write to distinct
``*_partial`` output names so a partial run can never overwrite the published
artifacts. ``HM3D_PRECISION`` selects an accepted power-of-ten grid. Nondefault
precisions receive an automatic filename suffix unless ``HM3D_OUTPUT_SUFFIX``
is set explicitly. ``HM3D_AUDIT_BRANCHES=0`` skips the duplicate diagnostic
eigendecomposition, and ``HM3D_SKIP_FIGURE=1`` suppresses per-run plots; these
options do not change descriptor construction or the emitted F/C tags.

Dataset sources (download once into $HM3D_DATA, default /tmp/datasets):
  * QM9 (gdb9.sdf): https://doi.org/10.6084/m9.figshare.978904  (Ramakrishnan
    et al., Sci. Data 1, 140022 (2014); the "dsgdb9nsd" SDF release).
  * MD17 (md17_*.npz): http://www.sgdml.org/#datasets  (Chmiela et al.,
    Sci. Adv. 3, e1603015 (2017)); files md17_aspirin.npz,
    md17_benzene2017.npz, md17_ethanol.npz.

Outputs: a printed report, a CSV of the length sweep, and a PDF figure.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from collections import Counter

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))

import matplotlib

from hashmol3d import __version__
from hashmol3d.core import DESCRIPTOR_VERSION, hash_molecule

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("HM3D_DATA", "/tmp/datasets")
PRECISION = float(os.environ.get("HM3D_PRECISION", "1e-4"))
if not math.isfinite(PRECISION) or PRECISION <= 0.0:
    raise ValueError("HM3D_PRECISION must be a positive finite power of ten")
METHOD = os.environ.get("HM3D_METHOD", "frame")
if METHOD not in {"frame", "canonical"}:
    raise ValueError("HM3D_METHOD must be 'frame' or 'canonical'")


def _precision_label(value):
    if value == 1.0:
        return "1"
    exponent = int(round(-math.log10(value)))
    return f"1e-{exponent}"


PRECISION_LABEL = _precision_label(PRECISION)
METHOD_SUFFIX = "" if METHOD == "frame" else "_canonical"
PRECISION_SUFFIX = "" if PRECISION == 1e-4 else f"_precision_{PRECISION_LABEL}"
OUTPUT_SUFFIX = os.environ.get("HM3D_OUTPUT_SUFFIX", METHOD_SUFFIX + PRECISION_SUFFIX)
# Cap on MD17 frames per molecule; 0 (default) means use the complete
# trajectory, reproducing the counts reported in the paper. Set a positive
# value to subsample large trajectories for a quicker run.
MAX_PER_MD17 = int(os.environ.get("HM3D_MAX_MD17", "0"))
# The branch audit repeats the eigendecomposition solely for diagnostics. It
# can be disabled for multi-precision sweeps while retaining the authoritative
# descriptor tags.
AUDIT_BRANCHES = os.environ.get("HM3D_AUDIT_BRANCHES", "1") != "0"
SKIP_FIGURE = os.environ.get("HM3D_SKIP_FIGURE", "0") == "1"

# Expected inputs for reproduction mode. ``expected`` is the number of
# geometries successfully hashed (None = not pinned, only presence is checked).
# These counts are filled in from the canonical full-dataset run and let the
# script fail loudly on a truncated or altered collection.
MANIFEST = {
    "QM9": {
        "files": ["gdb9.sdf"],
        "kind": "qm9",
        "expected": 133885,
        "source": "https://doi.org/10.6084/m9.figshare.978904",
    },
    "MD17:aspirin": {
        "files": ["md17_aspirin.npz"],
        "kind": "md17",
        "expected": 211762,
        "source": "http://www.sgdml.org/#datasets",
    },
    "MD17:benzene2017": {
        "files": ["md17_benzene2017.npz"],
        "kind": "md17",
        "expected": 627983,
        "source": "http://www.sgdml.org/#datasets",
    },
    "MD17:ethanol": {
        "files": ["md17_ethanol.npz"],
        "kind": "md17",
        "expected": 555092,
        "source": "http://www.sgdml.org/#datasets",
    },
}

print(f"HashMol3D package version: {__version__}")
print(f"descriptor version: {DESCRIPTOR_VERSION}")
print(f"requested method: {METHOD}")
print(f"precision: {PRECISION:g} Angstrom")
print(f"branch audit: {AUDIT_BRANCHES}")


def _frame_branch(Z, coords):
    """Classify the deterministic frame branch used by descriptor v8.

    This mirrors the branch predicates in ``hashmol3d.core._frame_signature``;
    it is diagnostic only and does not participate in descriptor generation.
    The emitted descriptor tag remains the authority on whether a frame was
    accepted or canonical fallback occurred.
    """
    if METHOD != "frame":
        return "canonical-request"
    z = np.asarray(Z, dtype=np.int64)
    x = np.asarray(coords, dtype=float)
    w = z.astype(float)
    scale = 1.0 / PRECISION

    def ksum(a):
        return float(np.sum(np.sort(a)))

    centroid = np.array([ksum(w * x[:, k]) for k in range(3)]) / ksum(w)
    c = x - centroid
    tensor = np.empty((3, 3))
    for a in range(3):
        for b in range(a, 3):
            tensor[a, b] = tensor[b, a] = ksum(w * c[:, a] * c[:, b])
    lam, vec = np.linalg.eigh(tensor)
    radii = np.linalg.norm(c, axis=1)
    max_radius_grid = float(radii.max()) * scale
    if max_radius_grid < 0.5:
        return "point"
    if not lam[2] > 0.0:
        return "canonical-fallback"
    if max_radius_grid < 10.0:
        axis = vec[:, 2]
        projected = c - np.outer(c @ axis, axis)
        residual = float(np.linalg.norm(projected, axis=1).max())
        if residual <= 64.0 * np.finfo(np.float64).eps * float(radii.max()):
            return "line"

    gap0 = float((lam[1] - lam[0]) / lam[2])
    gap1 = float((lam[2] - lam[1]) / lam[2])
    if min(gap0, gap1) >= 0.05:
        return "principal"
    if max_radius_grid < 10.0:
        return "canonical-fallback"

    if (gap0 < 0.05) != (gap1 < 0.05):
        unique_slot = 2 if gap0 < 0.05 else 0
        axis = vec[:, unique_slot]
        axial = c @ axis
        projected = c - np.outer(axial, axis)
        max_projected_grid = float(np.linalg.norm(projected, axis=1).max()) * scale
        if unique_slot == 2 and max_projected_grid < 0.5:
            return "line"
        if max_projected_grid < 10.0:
            return "canonical-fallback"
        return "one-axis-anchor"

    # In the fully near-degenerate branch, a usable second atom must lie at
    # least ten grid units away from the first atom-derived axis.  The exact
    # candidate-budget outcome is confirmed below from the emitted tag.
    q_radii = np.rint(radii * scale).astype(np.int64)
    first_keys = [(int(q_radii[i]), int(z[i])) for i in range(len(z))]
    winning = max(first_keys)
    anchors = [i for i, key in enumerate(first_keys) if key == winning]
    if len(anchors) > 10_000:
        return "canonical-fallback"
    for i in anchors:
        first_axis = c[i] / radii[i]
        projected = c - np.outer(c @ first_axis, first_axis)
        if float(np.linalg.norm(projected, axis=1).max()) * scale < 10.0:
            return "canonical-fallback"
    return "two-atom-anchor"


def _digest_and_tag(Z, coords):
    """Return full digest, emitted tag, diagnostic branch, and hash time."""
    started = time.perf_counter()
    r = hash_molecule(
        np.asarray(Z, int),
        np.asarray(coords, float),
        precision=PRECISION,
        length=64,
        method=METHOD,
    )
    hash_seconds = time.perf_counter() - started
    # descriptor is "V:...|P:...|Z:...|<TAG>:<body>"; take the leading letter
    # of the final |-field.
    tag = r.descriptor.rsplit("|", 1)[1].split(":", 1)[0]
    branch = _frame_branch(Z, coords) if AUDIT_BRANCHES else "not-audited"
    if AUDIT_BRANCHES and METHOD == "frame" and tag != "F":
        branch = "canonical-fallback"
    return r.geometry_hash, tag, branch, hash_seconds


# --------------------------------------------------------------------------
def load_qm9(sdf_path, digests, tags, branches, meta):
    from rdkit import Chem

    supp = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    n = 0
    hash_seconds = 0.0
    for mol in supp:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        X = np.array(mol.GetConformer().GetPositions(), dtype=float)
        if Z.size == 0 or not np.all(np.isfinite(X)):
            continue
        dg, tag, branch, one_hash_seconds = _digest_and_tag(Z, X)
        hash_seconds += one_hash_seconds
        digests.append(dg)
        tags[tag] += 1
        branches[branch] += 1
        meta["natoms"].append(int(Z.size))
        meta["elements"].update(int(z) for z in Z)
        n += 1
        if n % 100_000 == 0:
            print(f"  QM9 progress: {n:,} geometries")
    return n, hash_seconds


def load_md17(npz_path, digests, tags, branches, meta, stride):
    d = np.load(npz_path)
    Z = np.asarray(d["z"], dtype=int)
    R = np.asarray(d["R"], dtype=float)  # (M, N, 3) Angstrom
    idx = range(0, R.shape[0], stride)
    n = 0
    hash_seconds = 0.0
    for i in idx:
        dg, tag, branch, one_hash_seconds = _digest_and_tag(Z, R[i])
        hash_seconds += one_hash_seconds
        digests.append(dg)
        tags[tag] += 1
        branches[branch] += 1
        n += 1
        if n % 100_000 == 0:
            print(f"  {os.path.basename(npz_path)} progress: {n:,} geometries")
    meta["natoms"].append(int(Z.size))
    meta["elements"].update(int(z) for z in Z)
    return n, hash_seconds


# --------------------------------------------------------------------------
def collision_sweep(unique_digests, lengths):
    """For each L, count truncation collisions among the DISTINCT digests."""
    U = unique_digests
    out = []
    for L in lengths:
        prefixes = {d[:L] for d in U}
        collisions = len(U) - len(prefixes)
        out.append((L, len(U), len(prefixes), collisions))
    return out


def exact_excess(n, b):
    """Expected excess-item count n - 2^b[1 - (1 - 2^-b)^n] (ideal random hash).

    Evaluated as n + 2^b * expm1(n * log1p(-2^-b)) for numerical stability
    across both the dense (short L) and sparse (long L) regimes.
    """
    return n + (2.0**b) * math.expm1(n * math.log1p(-(2.0**-b)))


def sparse_pairs(n, b):
    """Sparse-limit expected number of colliding pairs, n(n-1)/2^{b+1}."""
    return n * (n - 1) / 2.0 ** (b + 1)


def _resolve_datasets(allow_partial):
    """Return the list of (name, spec, paths) present, enforcing MANIFEST.

    In reproduction mode (default) every manifest entry must be present or the
    run aborts. With ``allow_partial`` the present subset is used.
    """
    present, missing = [], []
    for name, spec in MANIFEST.items():
        paths = [os.path.join(DATA, f) for f in spec["files"]]
        if all(os.path.exists(p) for p in paths):
            present.append((name, spec, paths))
        else:
            missing.append((name, [p for p in paths if not os.path.exists(p)]))
    if missing and not allow_partial:
        print("\nERROR: reproduction mode requires all manifest datasets under", DATA)
        for name, mp in missing:
            src = MANIFEST[name]["source"]
            print(f"  missing {name}: {mp}  (source: {src})")
        print("Re-run with --allow-partial to use only the datasets present;")
        print("partial runs write *_partial outputs and never overwrite the")
        print("published collision_results.csv / fig_collisions.pdf.")
        sys.exit(1)
    if missing:
        print("\nWARNING: --allow-partial: proceeding without", [m[0] for m in missing])
    return present


def main(allow_partial=False):
    datasets = _resolve_datasets(allow_partial)
    per_dataset = {}  # name -> dataset statistics
    all_digests = []
    all_tags = {"C": 0, "F": 0}
    all_branches = Counter()
    count_mismatch = False

    for name, spec, paths in datasets:
        digests, meta = [], {"natoms": [], "elements": set()}
        tags = {"C": 0, "F": 0}
        branches = Counter()
        started = time.perf_counter()
        if spec["kind"] == "qm9":
            n, hash_seconds = load_qm9(paths[0], digests, tags, branches, meta)
            elapsed = time.perf_counter() - started
            print(
                f"{name}: {n} geometries, {len(set(digests))} distinct, "
                f"N in [{min(meta['natoms'])},{max(meta['natoms'])}], "
                f"{len(meta['elements'])} elements, "
                f"tags C={tags['C']} F={tags['F']}, "
                f"{elapsed:.2f} s ({n / elapsed:.0f} geometries/s), "
                f"branches={dict(branches)}"
            )
        else:
            d = np.load(paths[0])
            M = d["R"].shape[0]
            stride = 1 if MAX_PER_MD17 <= 0 else max(1, M // MAX_PER_MD17)
            n, hash_seconds = load_md17(paths[0], digests, tags, branches, meta, stride)
            elapsed = time.perf_counter() - started
            print(
                f"{name}: {n} geometries (stride {stride} of {M}), "
                f"{len(set(digests))} distinct, N={meta['natoms'][0]}, "
                f"tags C={tags['C']} F={tags['F']}, "
                f"{elapsed:.2f} s ({n / elapsed:.0f} geometries/s), "
                f"branches={dict(branches)}"
            )
        exp = spec.get("expected")
        # Count checks only apply to the full (unsubsampled) trajectory.
        if exp is not None and MAX_PER_MD17 <= 0 and n != exp:
            print(f"  COUNT MISMATCH for {name}: loaded {n}, manifest expects {exp}")
            count_mismatch = True
        per_dataset[name] = {
            "count": n,
            "distinct": len(set(digests)),
            "meta": meta,
            "tags": tags,
            "branches": branches,
            "seconds": elapsed,
            "hash_seconds": hash_seconds,
        }
        all_digests.extend(digests)
        for k in all_tags:
            all_tags[k] += tags[k]
        all_branches.update(branches)

    if not all_digests:
        print("No datasets found under", DATA)
        sys.exit(1)

    if count_mismatch and not allow_partial:
        print("\nERROR: dataset geometry counts do not match the manifest; aborting.")
        print("Use --allow-partial to override (writes *_partial outputs).")
        sys.exit(1)

    # ---- combined report ----
    U_all = set(all_digests)
    total = len(all_digests)
    print(f"\nTOTAL: {total} geometries hashed; {len(U_all)} distinct full 256-bit digests.")
    print(f"Frame-branch audit: {dict(all_branches)}")

    # Per-dataset path and throughput statistics.  Wall times include dataset
    # iteration and the diagnostic eigendecomposition, so they are explicitly
    # end-to-end harness timings rather than isolated microbenchmarks.
    suffix = OUTPUT_SUFFIX + ("_partial" if allow_partial else "")
    stats_path = os.path.join(_here, f"dataset_stats{suffix}.csv")
    branch_names = [
        "principal",
        "one-axis-anchor",
        "two-atom-anchor",
        "line",
        "point",
        "canonical-fallback",
        "canonical-request",
        "not-audited",
    ]
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "dataset",
                "method",
                "geometries",
                "distinct_full_digests",
                "tag_F",
                "tag_C",
                *branch_names,
                "seconds",
                "geometries_per_second",
                "hash_seconds",
                "hashes_per_second",
            ]
        )
        for name, row in per_dataset.items():
            w.writerow(
                [
                    name,
                    METHOD,
                    row["count"],
                    row["distinct"],
                    row["tags"]["F"],
                    row["tags"]["C"],
                    *(row["branches"][b] for b in branch_names),
                    f"{row['seconds']:.6f}",
                    f"{row['count'] / row['seconds']:.3f}",
                    f"{row['hash_seconds']:.6f}",
                    f"{row['count'] / row['hash_seconds']:.3f}",
                ]
            )
        total_seconds = sum(row["seconds"] for row in per_dataset.values())
        total_hash_seconds = sum(row["hash_seconds"] for row in per_dataset.values())
        w.writerow(
            [
                "TOTAL",
                METHOD,
                total,
                len(U_all),
                all_tags["F"],
                all_tags["C"],
                *(all_branches[b] for b in branch_names),
                f"{total_seconds:.6f}",
                f"{total / total_seconds:.3f}",
                f"{total_hash_seconds:.6f}",
                f"{total / total_hash_seconds:.3f}",
            ]
        )
    print(f"Wrote {stats_path}")
    print(
        f"Descriptor-path audit over all {total} geometries: "
        f"C={all_tags['C']} (complete), F={all_tags['F']} (frame)."
    )

    lengths = list(range(4, 33, 2))
    sweep = collision_sweep(U_all, lengths)
    n_distinct = len(U_all)
    print("\nTruncation-collision sweep over the DISTINCT geometries:")
    print(
        f"{'L(hex)':>7}{'bits':>6}{'distinct':>12}{'unique_hash':>13}"
        f"{'excess(n-U)':>13}{'E[n-U]exact':>13}{'pairs(sparse)':>14}"
    )
    for L, nU, nP, c in sweep:
        b = 4 * L
        print(
            f"{L:>7}{b:>6}{nU:>12}{nP:>13}{c:>13}"
            f"{exact_excess(n_distinct, b):>13.1f}{sparse_pairs(n_distinct, b):>14.1f}"
        )

    default_c = dict((L, c) for L, _, _, c in sweep).get(32, 0)
    print(
        f"\nAt the default length L=32 (128 bits): {default_c} collisions "
        f"among {n_distinct} distinct geometries."
    )

    # ---- CSV ----
    csv_path = os.path.join(_here, f"collision_results{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "length_hex",
                "bits",
                "distinct_geometries",
                "unique_hashes",
                "collisions",
                "expected_excess",
                "expected_pairs_sparse",
            ]
        )
        for L, nU, nP, c in sweep:
            b = 4 * L
            w.writerow(
                [L, b, nU, nP, c, f"{exact_excess(nU, b):.6f}", f"{sparse_pairs(nU, b):.6f}"]
            )
    print(f"Wrote {csv_path}")

    if not SKIP_FIGURE:
        make_figure([(L, 4 * L, c) for L, _, _, c in sweep], n_distinct, suffix=suffix)


def make_figure(rows, n_distinct, suffix=""):
    """Plot observed excess-item collisions against the exact occupancy
    expectation (primary) and the sparse-limit pair count (secondary).

    rows: list of (length_hex, bits, collisions).
    """
    bits = [b for _, b, _ in rows]
    cs = [max(c, 0) for *_, c in rows]
    exp_excess = [max(exact_excess(n_distinct, b), 1e-3) for b in bits]
    exp_pairs = [max(sparse_pairs(n_distinct, b), 1e-3) for b in bits]
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(bits, [c + 0.1 for c in cs], "o-", label="observed excess items $n-U$ ($+0.1$)")
    plt.plot(
        bits,
        exp_excess,
        "s--",
        color="gray",
        label=r"exact occupancy $n-2^{b}[1-(1-2^{-b})^{n}]$",
    )
    plt.plot(
        bits,
        exp_pairs,
        "^:",
        color="darkorange",
        label=r"sparse-limit pairs $n(n{-}1)/2^{\,b+1}$",
    )
    plt.axhline(1.0, ls=":", color="gray", lw=1)
    plt.text(52, 1.35, "one collision", color="gray", fontsize=8)
    plt.axvline(128, ls="--", color="crimson")
    plt.text(
        125,
        1e5,
        "default 128-bit ",
        color="crimson",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
    )
    plt.xlim(10, 136)
    plt.ylim(5e-2, 4e7)
    plt.xlabel("hash length (bits)")
    plt.ylabel("excess items among distinct geometries ($n-U$)")
    plt.yscale("log")
    plt.title(f"Truncation collisions vs hash length ({n_distinct:,} distinct geometries)")
    plt.legend(loc="upper right", fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    fp = os.path.join(_here, f"fig_collisions{suffix}.pdf")
    plt.savefig(fp, bbox_inches="tight")
    print(f"\nWrote {fp}")


def replot_from_csv():
    """Regenerate the figure from collision_results.csv (no datasets needed)."""
    with open(os.path.join(_here, "collision_results.csv")) as f:
        r = list(csv.DictReader(f))
    rows = [(int(x["length_hex"]), int(x["bits"]), int(x["collisions"])) for x in r]
    make_figure(rows, int(r[0]["distinct_geometries"]))


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_from_csv()
    else:
        main(allow_partial="--allow-partial" in sys.argv)
