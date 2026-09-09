"""
Exploratory local-curvature probes for HashMol3D precision studies.

An earlier version estimated the energy implied by a distance precision by
taking a *single one-sided* 1e-2 Angstrom bond stretch of an MMFF geometry and
scaling the RHF/STO-3G energy difference by (1e-4/1e-2)^2. That is invalid:
quadratic scaling requires (i) the reference to be a stationary point of the
same electronic-structure method, so the linear gradient term vanishes, and
(ii) the response to be harmonic over the fitted interval. Neither was shown,
and at a nonstationary geometry a linear term dominates.

This script replaces that invalid one-sided estimate with a controlled local
calculation:

  1. Optimize each polyene at RHF/STO-3G (geomeTRIC), and *verify* it is a
     stationary point by reporting SCF convergence and the max analytic
     gradient component at the optimized geometry.
  2. Probe two selected single-atom displacements with SYMMETRIC +/-delta
     displacements at several delta. The first moves a terminal hydrogen along
     the bond to its nearest neighbor (a carbon), i.e. a C-H stretch. The
     second moves a central carbon along the bond to ITS nearest neighbor;
     that neighbor is not restricted to carbon and is in practice an attached
     hydrogen, so this is a carbon-displacement probe (which also perturbs
     several other internal coordinates), not a C=C/C-C bond force constant.
  3. Fit E(delta) = E0 + a*delta + (1/2) k*delta^2 and report the linear term,
     fitted curvature k with its uncertainty, and R^2. We also tabulate
     (1/2)k*eps^2 as a local scale for that imposed displacement.
  4. Spot-check two systems at B3LYP/def2-SVP.

The larger of these two sampled curvatures is not the largest Hessian
eigenvalue. Moreover, equality of quantized distances does not bound Cartesian
displacements by eps. The reported local quadratic scale is therefore neither
an upper bound on unresolved energy nor a universal energy resolution for the
identifier.

Outputs: printed report, energy_results.csv, and fig_energy_resolution.pdf (the
legacy figure filename is retained for workflow compatibility). The
supplementary material uses the CSV values, not the figure, and states their
limited interpretation. All electronic-structure work is optional
analysis-only tooling (pyscf/geomeTRIC/rdkit); the hashmol3d package itself
remains NumPy-only.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
# Make the NumPy-only hashmol3d package importable when this script is run
# from the repository's paper/ directory without a prior `pip install`.
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))

import matplotlib
from rdkit import Chem
from rdkit.Chem import AllChem

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf import dft, gto, scf
from pyscf.data.nist import BOHR  # Angstrom per Bohr
from pyscf.geomopt.geometric_solver import optimize

EPS_REF = 1e-4  # default distance precision (Angstrom)

# Symmetric displacement grid (Angstrom) for the force-constant fit.
DELTAS = np.array([-0.02, -0.016, -0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012, 0.016, 0.02])

# geomeTRIC convergence criteria (tighter than the geomeTRIC defaults);
# reported verbatim in the paper. Units follow the geomeTRIC/pyscf convention:
# energy in Ha, gradients in Ha/Bohr, displacements in Angstrom.
CONV = dict(
    convergence_energy=1e-7,  # Ha
    convergence_grms=1e-4,  # Ha/Bohr
    convergence_gmax=2e-4,  # Ha/Bohr
    convergence_drms=8e-4,  # Angstrom
    convergence_dmax=1.2e-3,  # Angstrom
)


def polyene_smiles(n_double: int) -> str:
    return "C=C" + "C=C" * (n_double - 1)


def build_mmff(n_double: int):
    """Extended all-trans polyene, MMFF pre-optimized (starting geometry)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(polyene_smiles(n_double)))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    n_conf = 1 if n_double == 1 else 24
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    best, best_ext = None, -1.0
    for c in cids:
        X = np.array(mol.GetConformer(c).GetPositions(), float)
        ext = _pair(X).max()
        if ext > best_ext:
            best_ext, best = ext, X
    return Z, best


def _pair(X):
    iu, ju = np.triu_indices(X.shape[0], k=1)
    return np.linalg.norm(X[iu] - X[ju], axis=-1)


def _pyscf_mol(Z, X, basis, spin=0):
    atom = [[int(z), tuple(xyz)] for z, xyz in zip(Z, X)]
    return gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0, spin=spin)


def make_mf(Z, X, level):
    """Return an SCF/DFT object for the given level of theory."""
    basis = {"hf/sto-3g": "sto-3g", "b3lyp/def2-svp": "def2-svp"}[level]
    mol = _pyscf_mol(Z, X, basis)
    if level.startswith("b3lyp"):
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    return mf


def optimize_geometry(Z, X0, level):
    """Optimize at `level`; return (X_opt, converged, max_grad_component)."""
    mf = make_mf(Z, X0, level)
    mol_eq = optimize(mf, maxsteps=200, **CONV)
    X_opt = mol_eq.atom_coords() * BOHR  # Bohr -> Angstrom
    mf2 = make_mf(Z, X_opt, level)
    e = mf2.kernel()
    g = mf2.nuc_grad_method().kernel()  # Ha/Bohr, shape (N,3)
    return X_opt, bool(mf2.converged), float(np.abs(g).max()), float(e)


def energy(Z, X, level):
    mf = make_mf(Z, X, level)
    e = mf.kernel()
    return float(e), bool(mf.converged)


def nearest_neighbor(X, i):
    d = np.linalg.norm(X - X[i], axis=1)
    d[i] = np.inf
    return int(np.argmin(d))


def scan_coordinate(Z, X0, level, atom_idx, e0):
    """Symmetric +/-delta scan: displace `atom_idx` along the bond to its
    nearest neighbor. Return dict with fit (a, k, sigmas, R^2), the max
    distance-change slope, and the local quadratic scale at EPS_REF.
    """
    j = nearest_neighbor(X0, atom_idx)
    u = X0[atom_idx] - X0[j]
    u = u / np.linalg.norm(u)
    d0 = _pair(X0)
    es, dmax, conv_all = [], [], True
    for dl in DELTAS:
        X = X0.copy()
        X[atom_idx] = X0[atom_idx] + dl * u
        if abs(dl) < 1e-15:
            es.append(e0)
            dmax.append(0.0)
            continue
        e, ok = energy(Z, X, level)
        conv_all &= ok
        es.append(e)
        dmax.append(float(np.abs(_pair(X) - d0).max()))
    es = np.array(es)
    # relative energies in Hartree
    de = es - e0
    # fit de = a*delta + 0.5*k*delta^2  (no constant: reference is delta=0)
    # use full quadratic with intercept to let the fit expose any offset/noise
    coef, cov = np.polyfit(DELTAS, de, 2, cov=True)
    c2, c1, c0 = coef
    k = 2.0 * c2
    sig_k = 2.0 * float(np.sqrt(cov[0, 0]))
    a = c1
    sig_a = float(np.sqrt(cov[1, 1]))
    # R^2
    pred = np.polyval(coef, DELTAS)
    ss_res = float(np.sum((de - pred) ** 2))
    ss_tot = float(np.sum((de - de.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # slope of max distance-change vs |delta| (should be ~1 for a bond stretch)
    nz = np.array(dmax) > 0
    slope = float(np.mean(np.array(dmax)[nz] / np.abs(DELTAS)[nz]))
    eres = 0.5 * k * EPS_REF**2
    return dict(
        atom=atom_idx,
        neighbor=j,
        k=k,
        sig_k=sig_k,
        a=a,
        sig_a=sig_a,
        r2=r2,
        dmax_slope=slope,
        eres=eres,
        conv_all=conv_all,
        deltas=DELTAS.tolist(),
        de=de.tolist(),
    )


def stiff_atom_choices(Z, X):
    """Return (H_index, C_index): a terminal H and a central C to displace.

    The H is displaced along its C-H bond (a genuine C-H stretch). The C is
    displaced along the bond to ITS nearest neighbor, which for these polyenes
    is an attached hydrogen, so the second probe is a carbon displacement, not
    a C=C/C-C stretch.
    """
    Hs = [i for i, z in enumerate(Z) if z == 1]
    Cs = [i for i, z in enumerate(Z) if z == 6]
    # terminal H: the H whose nearest neighbor is a C (all of them) -- pick one
    # near a chain end (extremal along principal axis)
    c = X - X.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    t = c @ vt[0]
    hi = Hs[int(np.argmax(np.abs(t[Hs])))]
    ci = Cs[int(np.argmin(np.abs(t[Cs])))]  # central-ish C
    return hi, ci


def main():
    import hashmol3d

    print(f"HashMol3D package version: {hashmol3d.__version__}")
    print("energy calibration levels: HF/STO-3G (all), B3LYP/def2-SVP (subset)")
    print(f"geomeTRIC criteria: {CONV}")
    n_doubles = [1, 2, 3, 4, 5, 6, 7, 8]  # C2H4 .. C16H18 (matches tab:precision)
    subset_check = {1, 4}  # which n_double get the B3LYP/def2-SVP cross-check

    rows = []
    fig_N, fig_eres, fig_kmax = [], [], []
    # The B3LYP/def2-SVP cross-check series is collected separately below.
    fig_Nb, fig_eresb = [], []
    for nd in n_doubles:
        Z, X_mmff = build_mmff(nd)
        n = len(Z)
        nC = int(np.sum(Z == 6))
        name = f"C{nC}H{n - nC}"
        try:
            X_opt, scf_ok, gmax, e0 = optimize_geometry(Z, X_mmff, "hf/sto-3g")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] HF/STO-3G optimization FAILED: {exc}")
            continue
        hi, ci = stiff_atom_choices(Z, X_opt)
        sH = scan_coordinate(Z, X_opt, "hf/sto-3g", hi, e0)
        sC = scan_coordinate(Z, X_opt, "hf/sto-3g", ci, e0)
        # Retain the larger of the two sampled curvatures for a compact summary.
        # This is not claimed to be the stiffest molecular coordinate.
        stiff = sH if sH["k"] >= sC["k"] else sC
        print(
            f"\n[{name}] N={n} HF/STO-3G opt: SCF converged={scf_ok}, "
            f"max|grad|={gmax:.2e} Ha/Bohr, E0={e0:.6f} Ha"
        )
        for lab, s in (("C-H", sH), ("C-disp", sC)):
            print(
                f"   {lab:>7} probe: k={s['k']:.4f}+/-{s['sig_k']:.4f} Ha/A^2, "
                f"a={s['a']:+.2e}+/-{s['sig_a']:.1e} Ha/A (stationary=>~0), "
                f"R^2={s['r2']:.5f}, dmax/|d|={s['dmax_slope']:.3f}, "
                f"local-scale@1e-4={s['eres']:.2e} Ha, SCFok={s['conv_all']}"
            )

        row = dict(
            name=name,
            n=n,
            nC=nC,
            scf_converged=scf_ok,
            max_grad=gmax,
            e0=e0,
            k_CH=sH["k"],
            sig_k_CH=sH["sig_k"],
            a_CH=sH["a"],
            r2_CH=sH["r2"],
            nbr_CH=sH["neighbor"],
            k_Cdisp=sC["k"],
            sig_k_Cdisp=sC["sig_k"],
            a_Cdisp=sC["a"],
            r2_Cdisp=sC["r2"],
            nbr_Cdisp=sC["neighbor"],
            k_probe_max=stiff["k"],
            local_scale_1e4=stiff["eres"],
        )

        # ---- subset cross-check at B3LYP/def2-SVP ----
        if nd in subset_check:
            try:
                Xb, ok_b, gmax_b, e0b = optimize_geometry(Z, X_mmff, "b3lyp/def2-svp")
                hib, cib = stiff_atom_choices(Z, Xb)
                sHb = scan_coordinate(Z, Xb, "b3lyp/def2-svp", hib, e0b)
                sCb = scan_coordinate(Z, Xb, "b3lyp/def2-svp", cib, e0b)
                stiffb = sHb if sHb["k"] >= sCb["k"] else sCb
                row["k_probe_max_b3lyp"] = stiffb["k"]
                row["local_scale_1e4_b3lyp"] = stiffb["eres"]
                row["max_grad_b3lyp"] = gmax_b
                print(
                    f"   [B3LYP/def2-SVP] SCF conv={ok_b}, max|grad|={gmax_b:.2e}, "
                    f"k_probe={stiffb['k']:.4f} Ha/A^2, "
                    f"local-scale@1e-4={stiffb['eres']:.2e} Ha"
                )
                fig_Nb.append(n)
                fig_eresb.append(stiffb["eres"])
            except Exception as exc:  # noqa: BLE001
                print(f"   [B3LYP/def2-SVP] cross-check FAILED: {exc}")

        rows.append(row)
        fig_N.append(n)
        fig_eres.append(stiff["eres"])
        fig_kmax.append(stiff["k"])

    # ---- CSV ----
    csv_path = os.path.join(_here, "energy_results.csv")
    keys = sorted({k for r in rows for k in r})
    # keep a stable, readable leading order
    lead = [
        "name",
        "n",
        "nC",
        "scf_converged",
        "max_grad",
        "e0",
        "k_CH",
        "sig_k_CH",
        "a_CH",
        "r2_CH",
        "nbr_CH",
        "k_Cdisp",
        "sig_k_Cdisp",
        "a_Cdisp",
        "r2_Cdisp",
        "nbr_Cdisp",
        "k_probe_max",
        "local_scale_1e4",
        "k_probe_max_b3lyp",
        "local_scale_1e4_b3lyp",
        "max_grad_b3lyp",
    ]
    fields = [k for k in lead if k in keys] + [k for k in keys if k not in lead]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {csv_path}")

    # ---- figure: selected local quadratic scale vs N ----
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(
        fig_N,
        fig_eres,
        "o-",
        label=r"HF/STO-3G: $\frac{1}{2}k_{\rm probe}\varepsilon^2$ at $\varepsilon=10^{-4}$ Å",
    )
    if fig_Nb:
        plt.plot(fig_Nb, fig_eresb, "s--", color="seagreen", label=r"B3LYP/def2-SVP cross-check")
    plt.xlabel("number of atoms $N$")
    plt.ylabel("selected local quadratic scale (Hartree)")
    plt.yscale("log")
    plt.title("Local curvature scale for two selected displacement directions")
    plt.legend(loc="best", fontsize=8, frameon=False)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    fp = os.path.join(_here, "fig_energy_resolution.pdf")
    plt.savefig(fp, bbox_inches="tight")
    print(f"Wrote {fp}")


if __name__ == "__main__":
    main()
