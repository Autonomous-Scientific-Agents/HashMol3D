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
digest for every geometry. Two geometries with the *same* full digest have the
same canonical descriptor -- i.e. they are the same geometry at the working
precision -- which is correct behaviour, not a collision. A genuine collision
is two *distinct* full digests that share a truncated L-hex prefix. Working on
the set U of distinct full digests, the number of truncation collisions at
length L is |U| - |{ d[:L] : d in U }|. The default is L = 32 (128 bits).

Outputs: a printed report, a CSV of the length sweep, and a PDF figure.
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))

import matplotlib

from hashmol3d.core import DESCRIPTOR_VERSION, hash_molecule

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("HM3D_DATA", "/tmp/datasets")
PRECISION = 1e-4
# Cap on MD17 frames per molecule; 0 (default) means use the complete
# trajectory, reproducing the counts reported in the paper. Set a positive
# value to subsample large trajectories for a quicker run.
MAX_PER_MD17 = int(os.environ.get("HM3D_MAX_MD17", "0"))

print(f"descriptor version: {DESCRIPTOR_VERSION}")


def full_digest(Z, coords):
    """Full 64-hex SHA-256 of the canonical descriptor (256 bits)."""
    return hash_molecule(
        np.asarray(Z, int), np.asarray(coords, float), precision=PRECISION, length=64
    ).geometry_hash


# --------------------------------------------------------------------------
def load_qm9(sdf_path, digests, meta):
    from rdkit import Chem

    supp = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    n = 0
    for mol in supp:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        X = np.array(mol.GetConformer().GetPositions(), dtype=float)
        if Z.size == 0 or not np.all(np.isfinite(X)):
            continue
        digests.append(full_digest(Z, X))
        meta["natoms"].append(int(Z.size))
        meta["elements"].update(int(z) for z in Z)
        n += 1
    return n


def load_md17(npz_path, digests, meta, stride):
    d = np.load(npz_path)
    Z = np.asarray(d["z"], dtype=int)
    R = np.asarray(d["R"], dtype=float)  # (M, N, 3) Angstrom
    idx = range(0, R.shape[0], stride)
    n = 0
    for i in idx:
        digests.append(full_digest(Z, R[i]))
        n += 1
    meta["natoms"].append(int(Z.size))
    meta["elements"].update(int(z) for z in Z)
    return n


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


def main():
    per_dataset = {}  # name -> (count_geoms, set_of_full_digests)
    all_digests = []

    # ---- QM9 ----
    sdf = None
    for cand in ("gdb9.sdf", "qm9.sdf"):
        p = os.path.join(DATA, cand)
        if os.path.exists(p):
            sdf = p
            break
    if sdf:
        digests, meta = [], {"natoms": [], "elements": set()}
        n = load_qm9(sdf, digests, meta)
        per_dataset["QM9"] = (n, set(digests), meta)
        all_digests.extend(digests)
        print(
            f"QM9: {n} geometries, {len(set(digests))} distinct, "
            f"N in [{min(meta['natoms'])},{max(meta['natoms'])}], "
            f"{len(meta['elements'])} elements"
        )
    else:
        print("QM9 SDF not found; skipping.")

    # ---- MD17 ----
    for npz in sorted(glob.glob(os.path.join(DATA, "md17_*.npz"))):
        name = os.path.basename(npz).replace("md17_", "").replace(".npz", "")
        d = np.load(npz)
        M = d["R"].shape[0]
        stride = 1 if MAX_PER_MD17 <= 0 else max(1, M // MAX_PER_MD17)
        digests, meta = [], {"natoms": [], "elements": set()}
        n = load_md17(npz, digests, meta, stride)
        per_dataset[f"MD17:{name}"] = (n, set(digests), meta)
        all_digests.extend(digests)
        print(
            f"MD17:{name}: {n} geometries (stride {stride} of {M}), "
            f"{len(set(digests))} distinct, N={meta['natoms'][0]}"
        )

    if not all_digests:
        print("No datasets found under", DATA)
        return

    # ---- combined report ----
    U_all = set(all_digests)
    total = len(all_digests)
    print(
        f"\nTOTAL: {total} geometries hashed; {len(U_all)} distinct geometries "
        f"(distinct canonical descriptors)."
    )

    lengths = list(range(4, 33, 2))
    sweep = collision_sweep(U_all, lengths)
    print("\nTruncation-collision sweep over the DISTINCT geometries:")
    print(f"{'L(hex)':>7}{'bits':>6}{'distinct':>12}{'unique_hash':>13}{'collisions':>12}")
    for L, nU, nP, c in sweep:
        print(f"{L:>7}{4 * L:>6}{nU:>12}{nP:>13}{c:>12}")

    # per-length collisions at the default and a couple references
    default_c = dict((L, c) for L, _, _, c in sweep).get(32, 0)
    print(
        f"\nAt the default length L=32 (128 bits): {default_c} collisions "
        f"among {len(U_all)} distinct geometries."
    )

    # ---- CSV ----
    with open(os.path.join(_here, "collision_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length_hex", "bits", "distinct_geometries", "unique_hashes", "collisions"])
        for L, nU, nP, c in sweep:
            w.writerow([L, 4 * L, nU, nP, c])

    make_figure([(L, 4 * L, c) for L, _, _, c in sweep], len(U_all))


def make_figure(rows, n_distinct):
    """Plot observed truncation collisions against the birthday estimate.

    rows: list of (length_hex, bits, collisions).
    """
    bits = [b for _, b, _ in rows]
    cs = [max(c, 0) for *_, c in rows]
    expected = [n_distinct**2 / 2 ** (b + 1) for b in bits]
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(bits, [c + 0.1 for c in cs], "o-", label="observed ($+0.1$ to show zero)")
    plt.plot(bits, expected, "s--", color="gray", label="birthday estimate $n^2/2^{b+1}$")
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
    plt.ylim(5e-2, 4e6)
    plt.xlabel("hash length (bits)")
    plt.ylabel("collisions among distinct geometries")
    plt.yscale("log")
    plt.title(f"Truncation collisions vs hash length ({n_distinct:,} distinct geometries)")
    plt.legend(loc="center right", fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    fp = os.path.join(_here, "fig_collisions.pdf")
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
        main()
