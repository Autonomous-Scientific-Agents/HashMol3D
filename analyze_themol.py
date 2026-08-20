"""Collision analysis of HashMol3D over a THEMol HDF5 shard.

Reads every record (uuid -> atomic_numbers, coords, SMILES) from an HDF5
shard, computes the HashMol3D geometry hash, and reports collisions.

We hash at length=64 (full SHA-256) so that hash truncation can never
manufacture a collision; identical 64-hex tails therefore mean identical
*descriptors* (same Z-multiset + same distance multiset at the chosen
precision). We separately count collisions at the shipped default
length=32 (128-bit).

For each colliding group we inspect the SMILES to classify it:
  - one distinct non-isomeric SMILES  -> same constitution (legit duplicate
    structure, or enantiomer/geometry re-run)
  - several distinct non-isomeric SMILES -> genuine homometric collision:
    chemically different molecules with the same fingerprint.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import h5py
import numpy as np

from hashmol3d.core import hash_molecule

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/MBIS/mbis_6.h5"
PRECISION = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4


def decode(v):
    v = v[()] if hasattr(v, "__getitem__") else v
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def main() -> None:
    h = h5py.File(PATH, "r")
    keys = list(h.keys())
    n = len(keys)
    print(f"shard={PATH} records={n} precision={PRECISION:.0e}", flush=True)

    # hash (len 64) -> list of record indices
    groups64: dict[str, list[int]] = defaultdict(list)
    hash32_seen: dict[str, int] = {}
    hash32_collisions = 0

    uuids: list[str] = []
    smi_noniso: list[str] = []
    smi_iso: list[str] = []
    formulas: list[str] = []
    natoms: list[int] = []

    t0 = time.time()
    for i, k in enumerate(keys):
        g = h[k]
        z = np.asarray(g["atomic_numbers"][()]).reshape(-1).astype(int)
        xyz = np.asarray(g["coords"][()], dtype=float).reshape(-1, 3)
        res = hash_molecule(z, xyz, precision=PRECISION, length=64)
        full = res.geometry_hash
        short = full[:32]

        idx = len(uuids)
        uuids.append(k)
        smi_noniso.append(decode(g["mapped_nonisomeric_smiles"]))
        smi_iso.append(decode(g["mapped_isomeric_smiles"]))
        formulas.append(res.formula)
        natoms.append(int(z.size))

        groups64[full].append(idx)
        if short in hash32_seen:
            hash32_collisions += 1
        else:
            hash32_seen[short] = idx

        if (i + 1) % 20000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n}  ({rate:.0f}/s)", flush=True)
    h.close()

    n_total = len(uuids)
    n_unique64 = len(groups64)
    colliding = {hh: idxs for hh, idxs in groups64.items() if len(idxs) > 1}

    # Classify colliding groups.
    same_constitution = []   # 1 distinct non-isomeric SMILES
    homometric = []          # >1 distinct non-isomeric SMILES (true collision)
    for hh, idxs in colliding.items():
        noniso = {smi_noniso[j] for j in idxs}
        iso = {smi_iso[j] for j in idxs}
        rec = {
            "hash": hh,
            "count": len(idxs),
            "n_distinct_nonisomeric_smiles": len(noniso),
            "n_distinct_isomeric_smiles": len(iso),
            "formulas": sorted({formulas[j] for j in idxs}),
            "natoms": sorted({natoms[j] for j in idxs}),
            "nonisomeric_smiles": sorted(noniso),
            "isomeric_smiles": sorted(iso),
            "uuids": [uuids[j] for j in idxs],
        }
        if len(noniso) > 1:
            homometric.append(rec)
        else:
            same_constitution.append(rec)

    dup_records = sum(len(idxs) - 1 for idxs in colliding.values())

    summary = {
        "shard": PATH,
        "precision": PRECISION,
        "descriptor_version": __import__("hashmol3d").DESCRIPTOR_VERSION,
        "n_records": n_total,
        "n_distinct_hashes_len64": n_unique64,
        "n_colliding_groups_len64": len(colliding),
        "n_duplicate_records_len64": dup_records,
        "hash32_extra_collisions_vs_hash64": hash32_collisions - dup_records,
        "hash32_total_collisions": hash32_collisions,
        "n_same_constitution_groups": len(same_constitution),
        "n_homometric_groups_true_collisions": len(homometric),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out = {
        "summary": summary,
        "homometric_true_collisions": sorted(homometric, key=lambda r: -r["count"]),
        "same_constitution_examples": sorted(
            same_constitution, key=lambda r: -r["count"]
        )[:50],
    }
    with open("themol_collision_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote themol_collision_report.json")
    print(f"same-constitution collision groups: {len(same_constitution)}")
    print(f"TRUE (homometric) collision groups:  {len(homometric)}")
    if homometric:
        print("\n-- true collision examples (up to 10) --")
        for r in sorted(homometric, key=lambda r: -r["count"])[:10]:
            print(f"  hash={r['hash'][:16]}.. formulas={r['formulas']} "
                  f"smiles={r['nonisomeric_smiles']}")


if __name__ == "__main__":
    main()
