"""
Hash-flip stability and end-to-end round-trip analysis for HashMol3D.

This addresses the question a stable *identifier* must answer directly: with
what probability does the geometry hash change under coordinate perturbations,
and how does that probability depend on the noise-to-precision ratio, the atom
count, and the molecular class? It complements the per-distance amplification
study (analysis_precision.py) with the descriptor-level flip probability, plus
the quantization-boundary margins that control it and a realistic round-trip
duplicate test that reports both false-negative and false-merge rates.

Three parts:

  A. Flip-probability sweep. For a panel of molecules spanning size and class,
     apply isotropic Gaussian coordinate noise of magnitude sigma and measure
     the fraction of trials whose default-precision hash differs from the
     reference. We sweep the ratio sigma/epsilon and report Wilson 95% binomial
     confidence intervals with explicit trial counts.

  B. Boundary margins. A descriptor flips when *any* pairwise distance crosses
     a bin boundary. For each molecule we report the distribution of margins
     (distance of each quantized pair distance to its nearest bin edge) and the
     minimum margin, which is the single most fragile distance.

  C. Round-trip duplicate test. For copies that *should* be recognized as the
     same geometry -- random rigid motions, atom permutations, and fixed-decimal
     file round-trips -- we report the false-negative rate (hash changed). For
     copies that should *not* be merged -- distinct molecules, and distinct
     conformers -- we report the false-merge rate at several precisions.

Geometries are MMFF-optimized (RDKit ETKDGv3); this is a geometric-stability
study, so the force field used to generate the structures is immaterial. All
randomness is seeded. Outputs: printed tables, CSVs, and a figure.
"""

from __future__ import annotations

import csv
import math
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

from hashmol3d import __version__
from hashmol3d.core import DESCRIPTOR_VERSION, hash_molecule

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250822)
DEFAULT_EPS = 1e-4


# --------------------------------------------------------------------------
# Molecule panel: size axis (polyenes) + class axis (rigid/aromatic/cage/
# branched/heterogeneous). SMILES -> MMFF-optimized 3D geometry.
# --------------------------------------------------------------------------
PANEL = [
    ("C2H4", "C=C", "chain"),
    ("C6H8", "C=CC=CC=C", "chain"),
    ("C10H12", "C=CC=CC=CC=CC=C", "chain"),
    ("C16H18", "C=C" + "C=C" * 7, "chain"),
    ("benzene", "c1ccccc1", "aromatic"),
    ("naphthalene", "c1ccc2ccccc2c1", "aromatic"),
    ("neopentane", "CC(C)(C)C", "branched"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3", "cage"),
    ("cubane", "C1C2C3C1C4C2C3C4", "cage"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "heterogeneous"),
]


def build(smiles: str, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    X = np.array(mol.GetConformer().GetPositions(), dtype=float)
    return Z, X, mol


def gh(Z, X, eps=DEFAULT_EPS):
    return hash_molecule(Z, X, precision=eps, method="canonical").geometry_hash


def gh_desc(Z, X, eps=DEFAULT_EPS):
    """Return (geometry_hash, full canonical descriptor string).

    The descriptor is the pre-hash canonical form of the quantized element-
    labeled distance matrix; two structures with different descriptors are
    genuinely distinct at this grid (Proposition 2), so a shared hash between
    them would be a true false merge (a truncation/collision event) rather than
    a legitimate coarse-grid deduplication of quantized-identical geometries.
    """
    r = hash_molecule(Z, X, precision=eps, method="canonical")
    return r.geometry_hash, r.descriptor


def tag_of(Z, X, eps=DEFAULT_EPS):
    r = hash_molecule(Z, X, precision=eps, method="canonical")
    return r.descriptor.rsplit("|", 1)[1].split(":", 1)[0]


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def pair_distances(X):
    iu, ju = np.triu_indices(X.shape[0], k=1)
    return np.linalg.norm(X[iu] - X[ju], axis=-1)


def boundary_margins(X, eps):
    """Distance (Angstrom) of every pair distance to its nearest bin edge.

    Bins are centered at integer multiples of eps (round-half-to-even), so the
    edges sit at half-integer multiples. margin = eps*(0.5 - |u - round(u)|),
    u = d/eps. Zero margin = exactly on an edge (maximally fragile).
    """
    d = pair_distances(X)
    u = d / eps
    return eps * (0.5 - np.abs(u - np.round(u)))


# --------------------------------------------------------------------------
def part_A_flip_sweep():
    print("\n=== Part A: flip-probability sweep (eps = 1e-4, Wilson 95% CI) ===")
    ratios = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # sigma / eps
    n_trials = 500
    eps = DEFAULT_EPS
    rows = []
    curves = {}  # name -> (N, [p per ratio])
    for name, smi, cls in PANEL:
        Z, X0, _ = build(smi)
        n = len(Z)
        npair = n * (n - 1) // 2
        h0 = gh(Z, X0, eps)
        ps = []
        for ratio in ratios:
            sigma = ratio * eps
            flips = 0
            for _ in range(n_trials):
                Xn = X0 + rng.normal(0.0, sigma, X0.shape)
                if gh(Z, Xn, eps) != h0:
                    flips += 1
            p, lo, hi = wilson(flips, n_trials)
            ps.append(p)
            rows.append(
                dict(
                    name=name, cls=cls, N=n, npair=npair, eps=eps, sigma=sigma,
                    ratio=ratio, trials=n_trials, flips=flips,
                    p=p, ci_lo=lo, ci_hi=hi,
                )
            )
        curves[name] = (n, cls, ps)
        print(
            f"  {name:>12} ({cls:>13}, N={n:>2}, pairs={npair:>4}): "
            + " ".join(f"{r:.2f}:{p:.3f}" for r, p in zip(ratios, ps))
        )

    with open(os.path.join(_here, "flip_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote flip_sweep.csv")

    # figure: flip prob vs sigma/eps, one line per molecule
    plt.figure(figsize=(6.4, 4.4))
    styles = {"chain": "o-", "aromatic": "s-", "branched": "^-",
              "cage": "D-", "heterogeneous": "v-"}
    for name, (n, cls, ps) in curves.items():
        plt.plot(ratios, ps, styles.get(cls, "o-"), ms=4,
                 label=f"{name} (N={n}, {cls})")
    plt.xscale("log")
    plt.xlabel(r"noise / precision  $\sigma/\varepsilon$")
    plt.ylabel("hash-flip probability")
    plt.title(r"Descriptor flip probability vs $\sigma/\varepsilon$ (500 trials each)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=2, fontsize=7, frameon=False)
    plt.tight_layout()
    fp = os.path.join(_here, "fig_flips.pdf")
    plt.savefig(fp, bbox_inches="tight")
    print(f"  wrote {fp}")
    return rows


# --------------------------------------------------------------------------
def part_B_margins():
    print("\n=== Part B: quantization-boundary margins (eps = 1e-4 Angstrom) ===")
    print(f"  {'mol':>12} {'N':>3} {'pairs':>5} {'min margin(A)':>13} "
          f"{'median(A)':>10} {'frac<0.1eps':>11}")
    rows = []
    eps = DEFAULT_EPS
    for name, smi, cls in PANEL:
        Z, X0, _ = build(smi)
        m = boundary_margins(X0, eps)
        frac_fragile = float(np.mean(m < 0.1 * eps))
        rows.append(dict(name=name, cls=cls, N=len(Z), npair=len(m),
                         min_margin=float(m.min()), median_margin=float(np.median(m)),
                         frac_below_0p1eps=frac_fragile))
        print(f"  {name:>12} {len(Z):>3} {len(m):>5} {m.min():>13.2e} "
              f"{np.median(m):>10.2e} {frac_fragile:>11.3f}")
    with open(os.path.join(_here, "boundary_margins.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote boundary_margins.csv")
    return rows


# --------------------------------------------------------------------------
def random_rotation():
    """Uniform random rotation matrix via QR of a Gaussian matrix."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def part_C_roundtrip():
    print("\n=== Part C: round-trip duplicate test ===")
    eps = DEFAULT_EPS
    n_rep = 200

    # ---- C1: transforms that MUST preserve the hash (false-negative rate) ----
    print("\n  C1 false-negative rate (should stay equal), eps=1e-4:")
    print(f"  {'transform':>22} {'trials':>7} {'false_neg':>10} {'rate':>8}")
    fn_rows = []
    for label, kind in [
        ("rigid motion (rot+trans)", "rigid"),
        ("atom permutation", "perm"),
        ("file round-trip 6 dp", "rt6"),
        ("file round-trip 4 dp", "rt4"),
        ("file round-trip 3 dp", "rt3"),
        ("sub-eps noise (s=eps/10)", "subnoise"),
    ]:
        fneg = trials = 0
        for name, smi, cls in PANEL:
            Z, X0, _ = build(smi)
            h0 = gh(Z, X0, eps)
            for _ in range(n_rep // len(PANEL) + 1):
                if kind == "rigid":
                    Xt = X0 @ random_rotation().T + rng.normal(0, 5, 3)
                elif kind == "perm":
                    perm = rng.permutation(len(Z))
                    Xt, Zt = X0[perm], Z[perm]
                    if gh(Zt, Xt, eps) != h0:
                        fneg += 1
                    trials += 1
                    continue
                elif kind == "rt6":
                    Xt = np.round(X0, 6)
                elif kind == "rt4":
                    Xt = np.round(X0, 4)
                elif kind == "rt3":
                    Xt = np.round(X0, 3)
                elif kind == "subnoise":
                    Xt = X0 + rng.normal(0, eps / 10, X0.shape)
                if gh(Z, Xt, eps) != h0:
                    fneg += 1
                trials += 1
        rate = fneg / trials
        fn_rows.append(dict(transform=label, trials=trials, false_neg=fneg, rate=rate))
        print(f"  {label:>22} {trials:>7} {fneg:>10} {rate:>8.3f}")

    # ---- C2: distinct structures that must NOT merge (false-merge rate) ----
    # Distinct conformers of each flexible molecule at several precisions.
    # Ground truth for "distinct at this grid" is the canonical descriptor, not
    # the RDKit conformer id: rigid molecules (e.g. C2H4) yield many embeddings
    # of the *same* geometry, which SHOULD merge. A false merge is therefore a
    # pair with DIFFERENT descriptors but the SAME geometry hash (a truncation
    # collision). We also report how many pairs are quantized-identical (a
    # legitimate coarse-grid merge) to characterize deduplication behavior.
    print("\n  C2 false-merge (distinct descriptor, shared hash) + legitimate merges:")
    print(f"  {'eps(A)':>10} {'distinct_pairs':>14} {'false_merge':>12} "
          f"{'rate':>10} {'quant_ident':>12}")
    fm_rows = []
    conf_sets = []
    for name, smi, cls in [p for p in PANEL if p[2] in ("chain", "heterogeneous")]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=12, params=params)
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        confs = [np.array(mol.GetConformer(c).GetPositions(), float) for c in cids]
        conf_sets.append((name, Z, confs))
    for eps_t in [1e-1, 1e-2, 1e-3, 1e-4]:
        distinct_pairs = false_merge = quant_ident = 0
        for name, Z, confs in conf_sets:
            hd = [gh_desc(Z, c, eps_t) for c in confs]
            for i in range(len(hd)):
                for j in range(i + 1, len(hd)):
                    (hi, di), (hj, dj) = hd[i], hd[j]
                    if di == dj:
                        quant_ident += 1          # same quantized geometry: legit merge
                        continue
                    distinct_pairs += 1           # genuinely distinct at this grid
                    if hi == hj:
                        false_merge += 1          # different descriptor, same hash: collision
        rate = false_merge / distinct_pairs if distinct_pairs else 0.0
        fm_rows.append(dict(eps=eps_t, distinct_pairs=distinct_pairs,
                            false_merge=false_merge, rate=rate, quant_ident=quant_ident))
        print(f"  {eps_t:>10.0e} {distinct_pairs:>14} {false_merge:>12} "
              f"{rate:>10.4f} {quant_ident:>12}")

    with open(os.path.join(_here, "roundtrip.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "trials_or_pairs", "count", "rate"])
        for r in fn_rows:
            w.writerow(["false_negative", r["transform"], r["trials"], r["false_neg"], r["rate"]])
        for r in fm_rows:
            w.writerow(["false_merge", f"eps={r['eps']:.0e}", r["distinct_pairs"],
                        r["false_merge"], r["rate"]])
            w.writerow(["quant_identical", f"eps={r['eps']:.0e}", r["distinct_pairs"] + r["quant_ident"],
                        r["quant_ident"], r["quant_ident"] / (r["distinct_pairs"] + r["quant_ident"])
                        if (r["distinct_pairs"] + r["quant_ident"]) else 0.0])
    print("  wrote roundtrip.csv")
    return fn_rows, fm_rows


def main():
    print(f"HashMol3D package version: {__version__}")
    print(f"descriptor version: {DESCRIPTOR_VERSION}")
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "A"):
        part_A_flip_sweep()
    if only in ("all", "B"):
        part_B_margins()
    if only in ("all", "C"):
        part_C_roundtrip()


if __name__ == "__main__":
    main()
