"""
Does the default 1e-4 A precision keep the closely spaced geometries of routine
quantum-chemistry workflows distinct?

Two workflows probe the coarse (merging) limit of the precision directly:
  * finite-difference Hessian: central displacements +/- delta on every
    Cartesian degree of freedom;
  * geometry optimization: successive iterates converging to a minimum.

We hash each geometry at the default precision and count distinct hashes.
Reference system: ethanol, HF/STO-3G, optimized with geomeTRIC. Produces the
numbers behind Table (Section: precision for QC workflows).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))

from pyscf import gto, scf
from pyscf.data.nist import BOHR
from pyscf.geomopt.geometric_solver import optimize

from hashmol3d import hash_molecule

rng = np.random.default_rng(0)


def gh(Z, X, precision=1e-4):
    return hash_molecule(Z, X, precision=precision).geometry_hash


Z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=int)
X_eq = np.array(
    [
        [1.1687, -0.4055, 0.0000],
        [0.0000, 0.5518, 0.0000],
        [-1.1978, -0.2286, 0.0000],
        [2.1094, 0.1447, 0.0300],
        [1.1512, -1.0480, 0.8800],
        [1.1512, -1.0980, -0.8300],
        [0.0308, 1.1980, 0.8850],
        [0.0308, 1.1980, -0.8850],
        [-1.9216, 0.4085, 0.0100],
    ],
    dtype=float,
)
N = len(Z)


def fd_test():
    # Break symmetry so every displacement is inequivalent; any merge is then a
    # pure precision effect rather than a (correct) symmetry coincidence.
    X0 = X_eq + rng.normal(0, 0.03, X_eq.shape)
    print("=== Finite-difference Hessian displacements (central diff) ===")
    print(f"{'delta(A)':>10} {'#distinct':>10} {'#total':>7}")
    for delta in (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5):
        geoms = [X0]
        for a in range(N):
            for k in range(3):
                for s in (+1.0, -1.0):
                    g = X0.copy()
                    g[a, k] += s * delta
                    geoms.append(g)
        nd = len({gh(Z, g) for g in geoms})
        print(f"{delta:>10.0e} {nd:>10} {len(geoms):>7}")


def opt_test():
    print("\n=== Geometry-optimization trajectory (HF/STO-3G) ===")
    conv_sets = {
        "default": dict(
            convergence_energy=1e-6,
            convergence_grms=3e-4,
            convergence_gmax=4.5e-4,
            convergence_drms=1.2e-3,
            convergence_dmax=1.8e-3,
        ),
        "tight": dict(
            convergence_energy=1e-6,
            convergence_grms=1e-5,
            convergence_gmax=1.5e-5,
            convergence_drms=4e-5,
            convergence_dmax=6e-5,
        ),
    }
    for label, conv in conv_sets.items():
        start = X_eq + rng.normal(0, 0.08, X_eq.shape)
        atom = [[int(z), tuple(xyz)] for z, xyz in zip(Z, start)]
        mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        traj = []
        mf._traj = traj

        def cb(envs, traj=traj):
            traj.append(envs["mol"].atom_coords() * BOHR)

        with contextlib.redirect_stdout(io.StringIO()):  # silence geomeTRIC
            optimize(mf, callback=cb, maxsteps=100, **conv)
        traj = np.array(traj)
        hs = [gh(Z, g) for g in traj]
        steps = [np.abs(traj[i] - traj[i - 1]).max() for i in range(1, len(traj))]
        coll = sum(1 for i in range(1, len(hs)) if hs[i] == hs[i - 1])
        print(
            f"[{label:>7}] {len(traj):>2} iterates, {len(set(hs)):>2} distinct, "
            f"{coll} consecutive collisions, "
            f"smallest step = {min(steps):.1e} A"
        )


if __name__ == "__main__":
    fd_test()
    opt_test()
