"""Verify each HashMol3D collision group from the THEMol shard.

For every colliding group we ask two independent questions:

1. Are the 3D geometries actually the same structure?  We compare the
   sorted multiset of *full-precision* pairwise distances between the two
   records. If the max element-wise difference is ~0, the two records hold
   the same geometry (a duplicate entry), so the shared hash is correct.
   If it is large, we have a genuine homometric collision: two distinct
   geometries with the same rounded fingerprint.

2. Are they the same molecule?  We canonicalize both SMILES with RDKit
   after stripping atom-map numbers. Equal canonical SMILES => the
   differing raw strings were only cosmetic (atom renumbering / resonance).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

H5 = "data/MBIS/mbis_6.h5"


def sorted_dists(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu, ju = np.triu_indices(xyz.shape[0], k=1)
    return np.sort(d[iu, ju])


def canon(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles, sanitize=True)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def main() -> None:
    rep = json.load(open("themol_collision_report.json"))
    groups = rep["homometric_true_collisions"] + rep["same_constitution_examples"]
    h = h5py.File(H5, "r")

    print(f"{len(groups)} collision group(s)\n" + "=" * 70)
    n_same_mol = 0
    n_true = 0
    for gi, grp in enumerate(groups, 1):
        uuids = grp["uuids"]
        geoms = []
        for u in uuids:
            xyz = np.asarray(h[u]["coords"][()], dtype=float).reshape(-1, 3)
            geoms.append(xyz)
        # pairwise geometry difference (sorted full-precision distances)
        ref = sorted_dists(geoms[0])
        max_geom_diff = 0.0
        for xyz in geoms[1:]:
            v = sorted_dists(xyz)
            if v.shape == ref.shape:
                max_geom_diff = max(max_geom_diff, float(np.max(np.abs(v - ref))))
            else:
                max_geom_diff = float("inf")
        # canonical SMILES set
        canon_set = set()
        raw = grp["nonisomeric_smiles"]
        for s in raw:
            c = canon(s)
            canon_set.add(c if c is not None else f"<unparsable:{s}>")

        same_geometry = max_geom_diff < 1e-3
        same_molecule = len(canon_set) == 1
        verdict = (
            "SAME MOLECULE (duplicate entry)"
            if same_molecule and same_geometry
            else "SAME GEOMETRY, SMILES differ"
            if same_geometry
            else "*** TRUE HOMOMETRIC COLLISION ***"
        )
        if same_geometry:
            n_same_mol += 1
        else:
            n_true += 1

        print(f"[{gi}] formula={grp['formulas']} count={grp['count']}")
        print(f"    max geometry diff (sorted dists): {max_geom_diff:.2e} Angstrom")
        print(f"    distinct canonical SMILES: {len(canon_set)}")
        for c in sorted(canon_set):
            print(f"      {c}")
        print(f"    VERDICT: {verdict}\n")

    h.close()
    print("=" * 70)
    print(f"same-geometry duplicate groups: {n_same_mol}")
    print(f"true homometric collisions:     {n_true}")


if __name__ == "__main__":
    main()
