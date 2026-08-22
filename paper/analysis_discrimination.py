"""
Does discrimination require FINER precision for longer alkene chains?

The reviewer's concern (discrimination form): a long chain packs ~N^2/2
pairwise distances into a bounded range [~1, L] Angstrom. At precision eps there
are ~L/eps distance "bins"; when N^2/2 approaches L/eps the distance multiset
saturates and distinct geometries are more likely to collide. This predicts a
saturation precision eps* ~ L/(N^2/2) that gets FINER as the chain grows.

We test this operationally: for polyenes of increasing length we generate an
ensemble of genuinely distinct conformers and ask, at each precision, whether
the HashMol3D geometry hash keeps them all distinct (no false merges). We then
compare the empirical onset of collisions to the saturation bound and to the
default precision (1e-4 Angstrom).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))

import matplotlib
from rdkit import Chem
from rdkit.Chem import AllChem

from hashmol3d.core import hash_molecule

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# eps grid from coarse to fine, powers of ten only (HashMol3D requires the
# precision to be a power of ten <= 1 Å); coarse end wide enough to trigger
# saturation, and 1e-4 is the shipped default checked in the interpretation.
EPS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
TAU_SAME = 1e-3  # two geometries with class-gap below this are the "same"
K_CONF = 120  # conformers embedded per chain


def polyene_smiles(nd):
    return "C=C" + "C=C" * (nd - 1)


def build_conformers(nd, k=K_CONF):
    mol = Chem.AddHs(Chem.MolFromSmiles(polyene_smiles(nd)))
    p = AllChem.ETKDGv3()
    p.randomSeed = 20260814
    p.pruneRmsThresh = -1.0  # keep all; we prune ourselves by distance multiset
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=k, params=p))
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    confs = [np.array(mol.GetConformer(c).GetPositions(), float) for c in cids]
    return Z, confs


def class_multiset(Z, X):
    """Sorted distance vectors keyed by unordered element pair (Zi<=Zj)."""
    n = len(Z)
    iu, ju = np.triu_indices(n, k=1)
    d = np.linalg.norm(X[iu] - X[ju], axis=-1)
    za = np.minimum(Z[iu], Z[ju])
    zb = np.maximum(Z[iu], Z[ju])
    out = {}
    for a, b, dd in zip(za, zb, d):
        out.setdefault((int(a), int(b)), []).append(float(dd))
    return {k: np.sort(np.array(v)) for k, v in out.items()}


def class_gap(msA, msB):
    """L-inf difference between two same-molecule class multisets."""
    g = 0.0
    for k in msA:
        g = max(g, float(np.max(np.abs(msA[k] - msB[k]))))
    return g


def distinct_clusters(mss, tau=TAU_SAME):
    """Single-linkage cluster conformers whose class-gap < tau -> distinct reps."""
    n = len(mss)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if class_gap(mss[i], mss[j]) < tau:
                parent[find(i)] = find(j)
    reps = {}
    for i in range(n):
        reps.setdefault(find(i), i)
    return list(reps.values())


def main():
    n_doubles = [2, 3, 4, 6, 8, 12, 16, 20, 24]
    rows = []
    print(
        f"{'mol':>10} {'N':>4} {'pairs':>6} {'L(Ang)':>8} {'nConf':>6} "
        f"{'nDist':>6} {'dMin(Ang)':>10} {'eps*':>9} " + " ".join(f"u@{e:.0e}" for e in EPS)
    )

    for nd in n_doubles:
        Z, confs = build_conformers(nd)
        n = len(Z)
        nC = int(np.sum(Z == 6))
        name = f"C{nC}H{n - nC}"
        mss = [class_multiset(Z, X) for X in confs]

        reps = distinct_clusters(mss)
        rep_ms = [mss[i] for i in reps]
        rep_conf = [confs[i] for i in reps]
        n_dist = len(reps)

        # closest genuinely-distinct pair
        dmin = np.inf
        for i in range(n_dist):
            for j in range(i + 1, n_dist):
                dmin = min(dmin, class_gap(rep_ms[i], rep_ms[j]))
        dmin = float(dmin) if np.isfinite(dmin) else float("nan")

        n_pairs = n * (n - 1) // 2
        L = max(
            float(
                np.max(
                    np.linalg.norm(
                        X[np.triu_indices(n, 1)[0]] - X[np.triu_indices(n, 1)[1]], axis=-1
                    )
                )
            )
            for X in rep_conf
        )
        eps_star = L / n_pairs  # saturation heuristic

        # unique full-length hashes among distinct reps at each precision
        uniq = {}
        for e in EPS:
            hs = {hash_molecule(Z, X, precision=e, length=64).geometry_hash for X in rep_conf}
            uniq[e] = len(hs)

        rows.append(
            dict(
                name=name,
                n=n,
                pairs=n_pairs,
                L=L,
                nconf=len(confs),
                ndist=n_dist,
                dmin=dmin,
                eps_star=eps_star,
                uniq=uniq,
            )
        )
        print(
            f"{name:>10} {n:>4} {n_pairs:>6} {L:>8.2f} {len(confs):>6} "
            f"{n_dist:>6} {dmin:>10.2e} {eps_star:>9.2e} " + " ".join(f"{uniq[e]:>5}" for e in EPS)
        )

    # ---- interpretation ----
    print("\nInterpretation:")
    print(f" - 'nDist' = # genuinely-distinct conformers (class-gap >= {TAU_SAME:.0e} Ang).")
    print(" - u@eps  = # unique hashes among those reps at precision eps.")
    print("   A false merge has occurred whenever u@eps < nDist.")
    print(" - eps* = L / (N choose 2): the distance-bin saturation scale.")
    default_ok = all(r["uniq"][1e-4] == r["ndist"] for r in rows)
    print(
        f" - At the DEFAULT precision 1e-4: "
        f"{'zero false merges for every chain.' if default_ok else 'SOME false merges occurred!'}"
    )

    # ---- figure: required precision vs chain length ----
    Ns = [r["n"] for r in rows]
    eps_star = [r["eps_star"] for r in rows]
    # closest distinct-conformer gap: eps must be finer than this to resolve
    dmin_N = [r["n"] for r in rows if np.isfinite(r["dmin"])]
    dmin_v = [r["dmin"] for r in rows if np.isfinite(r["dmin"])]
    # coarsest eps in the grid with no false merge (>= this value works)
    onset = []
    for r in rows:
        good = [e for e in EPS if r["uniq"][e] == r["ndist"]]
        onset.append(max(good) if good else min(EPS))

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(
        Ns, eps_star, "s-", label=r"saturation scale $\epsilon^*=L/\binom{N}{2}$ ($\propto 1/N$)"
    )
    plt.plot(dmin_N, dmin_v, "o-", label="closest distinct-conformer gap $\\Delta_{\\min}$")
    plt.plot(
        Ns,
        onset,
        "D--",
        color="green",
        alpha=0.7,
        label="coarsest tested $\\epsilon$ with 0 false merges",
    )
    plt.axhline(1e-4, ls="--", color="gray", lw=1)
    plt.text(Ns[0], 1.25e-4, "default $\\epsilon=10^{-4}$ Å", color="gray", fontsize=9)
    plt.xlabel("number of atoms $N$")
    plt.ylabel("precision $\\epsilon$ (Å)")
    plt.yscale("log")
    plt.ylim(5e-5, 3.0)
    plt.title("Precision needed for discrimination vs molecule size")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    f = os.path.join(_here, "fig_discrimination.pdf")
    plt.savefig(f, bbox_inches="tight")
    print(f"\nWrote {f}")


if __name__ == "__main__":
    main()
