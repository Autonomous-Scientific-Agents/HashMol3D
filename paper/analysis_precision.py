"""
Precision-sensitivity analysis for HashMol3D on alkene (polyene) chains.

Motivation
----------
An expert reviewer questioned whether *high geometric precision* is required
for longer alkene chains. We test this quantitatively by relating three
things as a function of chain length N:

  1. how coordinate perturbations propagate into pairwise-distance changes
     (the quantity HashMol3D actually rounds and hashes),
  2. the energetic significance of those distance changes, measured by the
     nuclear-repulsion energy V_NN = sum_{i<j} Z_i Z_j / r_ij (an exact
     function of the pairwise distances) and by HF total energy (PySCF),
  3. whether the HashMol3D geometry hash changes at a given precision.

Two perturbation modes are studied:

  * RANDOM per-atom Gaussian noise  -> models numerical / round-off noise.
  * CONSTRUCTED END-HINGED BEND -> rotate one side of a terminal C-C bond
    about an axis through the terminal carbon. This is a geometric stress
    test, not a normal mode or a molecular-dynamics sample.

Outputs: printed tables, a CSV, and two PDF figures used in the paper.
"""

from __future__ import annotations

import csv
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
from pyscf import gto, scf
from pyscf.data.nist import BOHR  # Angstrom per Bohr

rng = np.random.default_rng(12345)
PRECISIONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


# --------------------------------------------------------------------------
# Geometry construction: all-trans conjugated polyenes  CH2=CH-(CH=CH)_{k}
# --------------------------------------------------------------------------
def polyene_smiles(n_double: int) -> str:
    """All-trans polyene with n_double C=C units, e.g. C=C, C=CC=C, ..."""
    return "C=C" + "C=C" * (n_double - 1)


def build_polyene(n_double: int):
    """Return (Z, coords[Ang], rdkit_mol) for the most-extended (all-trans),
    MMFF-optimized polyene conformer.

    We embed several conformers, optimize each, and keep the one with the
    largest end-to-end extent. This reliably selects the extended all-trans
    minimum, giving straight, near-stationary geometries so that the
    lever-arm bend in Experiment 2 scales cleanly with chain length.
    """
    mol = Chem.MolFromSmiles(polyene_smiles(n_double))
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    n_conf = 1 if n_double == 1 else 24
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)

    best_cid, best_extent = None, -1.0
    for cid in cids:
        c = np.array(mol.GetConformer(cid).GetPositions(), dtype=float)
        extent = pair_distances(c).max()
        if extent > best_extent:
            best_extent, best_cid = extent, cid
    coords = np.array(mol.GetConformer(best_cid).GetPositions(), dtype=float)
    # Keep only the chosen conformer so downstream GetConformer() is unambiguous.
    keep = coords.copy()
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, keep[i].tolist())
    mol.AddConformer(conf, assignId=True)
    return Z, coords, mol


def nuclear_repulsion(Z, coords_ang) -> float:
    """Exact nuclear repulsion energy (Hartree); coords in Angstrom."""
    r = coords_ang / BOHR  # -> Bohr
    d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(Z), k=1)
    return float(np.sum(Z[iu] * Z[ju] / d[iu, ju]))


def hf_energy(Z, coords_ang, basis="sto-3g") -> float:
    """RHF total energy (Hartree) at the given geometry."""
    atom = [[int(z), tuple(xyz)] for z, xyz in zip(Z, coords_ang)]
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    return float(mf.kernel())


def pair_distances(coords):
    iu, ju = np.triu_indices(coords.shape[0], k=1)
    d = np.linalg.norm(coords[iu] - coords[ju], axis=-1)
    return d


def hash_at(Z, coords, precision):
    return hash_molecule(Z, coords, precision=precision, method="canonical").geometry_hash


# --------------------------------------------------------------------------
# Soft-mode central BEND (lever arm): rotate one half about an axis
# perpendicular to the chain, hinged at the central backbone bond. This is a
# single bond-angle change (all other internal coordinates preserved), so its
# energy cost is small and roughly length-independent, while the far-atom
# displacement it produces grows with the chain's arm length.
# --------------------------------------------------------------------------
def end_backbone_bond(mol):
    """Pick the terminal C-C backbone bond (nearest a chain end).

    Hinging the bend here makes the lever arm equal to (nearly) the full
    molecular length, so the far-atom displacement scales cleanly with chain
    length rather than with a variable interior fragment size.
    """
    ccbonds = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
            i, j = a1.GetIdx(), a2.GetIdx()
            ccbonds.append((min(i, j), max(i, j)))
    if not ccbonds:
        return None
    # smallest-index C-C bond = the one at the start of the backbone; return
    # it oriented so the pivot atom `a` is the terminal one.
    a, b = min(ccbonds, key=lambda ij: ij[0] + ij[1])
    return a, b


def long_axis(coords):
    """Principal (longest) axis of the geometry via PCA."""
    c = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0]


def perp_axes(coords, k=8):
    """k unit vectors perpendicular to the long axis, evenly spread in azimuth."""
    la = long_axis(coords)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(la, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(la, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(la, e1)
    e2 /= np.linalg.norm(e2)
    return [np.cos(t) * e1 + np.sin(t) * e2 for t in np.linspace(0, np.pi, k, endpoint=False)]


def side_atoms(mol, a, b):
    """Atoms on b's side after cutting bond a-b (BFS on the bond graph)."""
    adj = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if {i, j} == {a, b}:
            continue
        adj[i].add(j)
        adj[j].add(i)
    seen, stack = {b}, [b]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(a)
    return sorted(seen)


def rotate_about_axis(coords, pivot, axis, moving, angle_rad):
    """Rotate `moving` atoms about (pivot, axis) by angle_rad (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    out = coords.copy()
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    for idx in moving:
        v = coords[idx] - pivot
        out[idx] = pivot + v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
    return out


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def main():
    print(f"HashMol3D package version: {__version__}")
    print(f"descriptor version: {DESCRIPTOR_VERSION}")
    n_doubles = list(range(1, 9))  # ethene ... hexadecaoctaene (C16H18)
    rows = []
    print(f"{'chain':>16} {'N':>4} {'C':>3} {'NRE(Ha)':>12} {'E_HF(Ha)':>14} {'maxD(Ang)':>10}")

    # storage for figures
    fig_N, fig_rand_dmax, fig_lever_dmax = [], [], []
    fig_rand_dNRE, fig_lever_dNRE = [], []
    fig_lever_dE, fig_rand_dE = [], []

    SIGMA = 1e-4  # random noise magnitude (Ang), ~ tight-opt noise floor
    N_TRIALS = 200
    # HF evaluations under noise are costly. Five draws give only a descriptive
    # mean and do not support a useful uncertainty estimate.
    N_E_TRIALS = 5
    DTHETA = np.deg2rad(0.5)  # small soft-mode torsion (degrees)

    for nd in n_doubles:
        Z, X0, mol = build_polyene(nd)
        n = len(Z)
        nC = int(np.sum(Z == 6))
        d0 = pair_distances(X0)
        nre0 = nuclear_repulsion(Z, X0)
        e0 = hf_energy(Z, X0)
        name = f"C{nC}H{n - nC}"
        print(f"{name:>16} {n:>4} {nC:>3} {nre0:>12.4f} {e0:>14.6f} {d0.max():>10.3f}")

        # ---- Experiment 1: random per-atom Gaussian noise ----
        rand_dmax, rand_dnre, rand_dE = [], [], []
        flip_frac = {p: 0 for p in PRECISIONS}
        for t in range(N_TRIALS):
            Xn = X0 + rng.normal(0.0, SIGMA, X0.shape)
            dn = pair_distances(Xn)
            rand_dmax.append(np.abs(dn - d0).max())
            rand_dnre.append(abs(nuclear_repulsion(Z, Xn) - nre0))
            if t < N_E_TRIALS:
                # Computed after the RNG draw so the seeded coordinate-noise
                # stream is unchanged by whether an energy is evaluated.
                rand_dE.append(abs(hf_energy(Z, Xn) - e0))
            for p in PRECISIONS:
                if hash_at(Z, Xn, p) != hash_at(Z, X0, p):
                    flip_frac[p] += 1
        rand_dmax_mean = float(np.mean(rand_dmax))
        rand_dnre_mean = float(np.mean(rand_dnre))
        rand_dE_mean = float(np.mean(rand_dE))
        flip_frac = {p: flip_frac[p] / N_TRIALS for p in PRECISIONS}

        # ---- Experiment 2: soft end-hinged bend (lever arm) ----
        # Hinge at a terminal carbon and bend the rest of the chain about an
        # axis perpendicular to the long axis. The maximal atomic displacement
        # is ~ (arm length) * sin(dtheta), i.e. it scales with chain length,
        # while the energy cost (a single bond-angle change) is small and
        # roughly length-independent.
        bond = end_backbone_bond(mol)
        lever_disp = lever_dmax = lever_dnre = lever_dE = float("nan")
        lever_flip = {p: None for p in PRECISIONS}
        if bond is not None:
            a, b = bond
            moving = side_atoms(mol, a, b)  # b-side atoms
            pivot = X0[a]  # hinge at the terminal atom
            axis = perp_axes(X0, k=8)[0]  # one perpendicular bend axis
            Xt = rotate_about_axis(X0, pivot, axis, moving, DTHETA)
            # max per-atom Euclidean displacement
            lever_disp = float(np.linalg.norm(Xt - X0, axis=1).max())
            lever_dmax = float(np.abs(pair_distances(Xt) - d0).max())
            lever_dnre = abs(nuclear_repulsion(Z, Xt) - nre0)
            lever_dE = abs(hf_energy(Z, Xt) - e0)
            lever_flip = {p: (hash_at(Z, Xt, p) != hash_at(Z, X0, p)) for p in PRECISIONS}

        # ---- Experiment 3: single central-bond stretch datum ----
        # Stretch one central backbone C-C bond by a fixed DR and measure the
        # energy change. A one-sided change from an MMFF reference cannot be
        # converted into a force constant or an identifier energy resolution.
        DR = 1e-2  # Angstrom
        cb = end_backbone_bond(mol)  # reuse: (a interior-ish, b)
        stretch_dE = stretch_dnre = float("nan")
        if cb is not None:
            # pick the most central C-C bond for the stretch
            ccb = [
                (
                    min(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()),
                    max(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()),
                )
                for bd in mol.GetBonds()
                if bd.GetBeginAtom().GetAtomicNum() == 6 and bd.GetEndAtom().GetAtomicNum() == 6
            ]
            a2, b2 = min(ccb, key=lambda ij: abs((ij[0] + ij[1]) / 2 - n / 2))
            u = X0[b2] - X0[a2]
            u = u / np.linalg.norm(u)
            moving2 = side_atoms(mol, a2, b2)
            Xs = X0.copy()
            Xs[moving2] += DR * u
            stretch_dE = abs(hf_energy(Z, Xs) - e0)
            stretch_dnre = abs(nuclear_repulsion(Z, Xs) - nre0)

        rows.append(
            dict(
                name=name,
                n=n,
                nC=nC,
                nre0=nre0,
                e0=e0,
                dmax0=float(d0.max()),
                rand_dmax=rand_dmax_mean,
                rand_dnre=rand_dnre_mean,
                rand_dE=rand_dE_mean,
                flip_1e2=flip_frac[1e-2],
                flip_1e3=flip_frac[1e-3],
                flip_1e4=flip_frac[1e-4],
                flip_1e5=flip_frac[1e-5],
                lever_disp=lever_disp,
                lever_dmax=lever_dmax,
                lever_dnre=lever_dnre,
                lever_dE=lever_dE,
                lever_flip_1e2=lever_flip[1e-2],
                lever_flip_1e3=lever_flip[1e-3],
                lever_flip_1e4=lever_flip[1e-4],
                stretch_dr=DR,
                stretch_dE=stretch_dE,
                stretch_dnre=stretch_dnre,
            )
        )
        fig_N.append(n)
        fig_rand_dmax.append(rand_dmax_mean)
        fig_lever_dmax.append(lever_disp)
        fig_rand_dNRE.append(rand_dnre_mean)
        fig_rand_dE.append(rand_dE_mean)
        fig_lever_dNRE.append(lever_dnre)
        fig_lever_dE.append(lever_dE)

    # ---- write CSV ----
    csv_path = os.path.join(_here, "analysis_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ---- printed summary tables ----
    print(
        f"\n=== Experiment 1: random per-atom noise, sigma = {SIGMA:.0e} Ang, {N_TRIALS} trials ==="
    )
    print(
        f"{'mol':>10} {'N':>4} {'mean max|dd|(Ang)':>18} "
        f"{'mean|dNRE|(Ha)':>15} {'flip@1e-3':>10} {'flip@1e-4':>10}"
    )
    for r in rows:
        print(
            f"{r['name']:>10} {r['n']:>4} {r['rand_dmax']:>18.2e} "
            f"{r['rand_dnre']:>15.2e} {r['flip_1e3']:>10.2f} "
            f"{r['flip_1e4']:>10.2f}"
        )

    print(f"\n=== Experiment 2: constructed rigid bend, dtheta = {np.rad2deg(DTHETA):.2f} deg ===")
    print(
        f"{'mol':>10} {'N':>4} {'maxDisp(Ang)':>13} {'max|dd|(Ang)':>13} "
        f"{'|dE_HF|(Ha)':>12} {'flip@1e-2':>10} {'flip@1e-3':>10}"
    )
    for r in rows:
        print(
            f"{r['name']:>10} {r['n']:>4} {r['lever_disp']:>13.2e} "
            f"{r['lever_dmax']:>13.2e} {r['lever_dE']:>12.2e} "
            f"{str(r['lever_flip_1e2']):>10} {str(r['lever_flip_1e3']):>10}"
        )

    print(
        f"\n=== Experiment 3 (raw stretch datum): single central C-C bond "
        f"stretch, dr = {rows[0]['stretch_dr']:.0e} Ang ==="
    )
    print("    NOTE: the one-sided finite difference below is reported only as")
    print("    raw geometry-to-energy sensitivity. analysis_energy.py performs")
    print("    symmetric local-curvature fits, but those fits are not a global")
    print("    energy-resolution bound for descriptor equality.")
    print(f"{'mol':>10} {'N':>4} {'|dE_HF|(Ha)':>12} {'|dNRE|(Ha)':>12}")
    for r in rows:
        print(f"{r['name']:>10} {r['n']:>4} {r['stretch_dE']:>12.2e} {r['stretch_dnre']:>12.2e}")

    # ---- Figure 1: distance-change amplification vs chain length ----
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(
        fig_N,
        fig_rand_dmax,
        "o-",
        label=f"random noise: max $|\\Delta d|$ ($\\sigma={SIGMA:.0e}$ Å)",
    )
    plt.plot(
        fig_N,
        fig_lever_dmax,
        "s-",
        label=f"constructed bend ({np.rad2deg(DTHETA):.1f}$^\\circ$): max atomic displacement",
    )
    plt.axhline(1e-4, ls="--", color="gray", lw=1)
    plt.text(fig_N[0], 1.15e-4, "precision $=10^{-4}$ Å", color="gray", fontsize=9)
    plt.xlabel("number of atoms $N$")
    plt.ylabel("geometric change (Å)")
    plt.yscale("log")
    plt.title("Distance sensitivity vs molecule size")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    f1 = os.path.join(_here, "fig_sensitivity.pdf")
    plt.savefig(f1, bbox_inches="tight")
    print(f"Wrote {f1}")

    # ---- Figure 2: energy significance vs chain length ----
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(fig_N, fig_rand_dNRE, "o-", label="|$\\Delta V_{NN}$| random noise")
    plt.plot(fig_N, fig_lever_dNRE, "s-", label="|$\\Delta V_{NN}$| constructed bend")
    plt.plot(fig_N, fig_rand_dE, "v--", label="|$\\Delta E_{HF}$| random noise")
    plt.plot(fig_N, fig_lever_dE, "^-", label="|$\\Delta E_{HF}$| constructed bend")
    plt.axhline(1.6e-3, ls="--", color="crimson", lw=1, label="chemical accuracy (1.6 mHa)")
    plt.xlabel("number of atoms $N$")
    plt.ylabel("energy change (Hartree)")
    plt.yscale("log")
    plt.title("Energetic significance of a hash-changing perturbation")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    f2 = os.path.join(_here, "fig_energy.pdf")
    plt.savefig(f2, bbox_inches="tight")
    print(f"Wrote {f2}")


if __name__ == "__main__":
    main()
