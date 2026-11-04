"""
Boundary-aware matching for HashMol3D: margins, probe keys, RMSD confirmation.

An exact geometry hash splits two copies of a geometry whenever noise moves a
hashed value across a rounding boundary. This script measures how well the
lookup-side helpers recover such splits without changing any identifier:

  A. Noise panel. For each fixed reference geometry of the flip study
     (noise_panel.npz, written by analysis_flips.py) and each noise ratio
     sigma/epsilon, perturb the coordinates and record whether the noisy
     copy matches the reference
       - on its exact geometry hash,
       - on an exact hash OR a half-cell offset-grid hash (the alternative
         considered and rejected: one grid is rarely safe for all 3N values),
       - through boundary_probes() keys (at most 1 or 2 flipped values),
       - by same_geometry() RMSD confirmation (tolerance = precision),
     and whether either side has min_margin below the probe margin. Timings
     are per call.

  B. Corpus margins (optional; needs $HM3D_DATA/gdb9.sdf). For QM9 at several
     precisions, the fraction of geometries whose min_margin is below each
     margin, and the number of values within the probe margin, which sets
     the probe-set size.

Outputs: boundary_matching.csv and, with QM9, boundary_margin_corpus.csv.
HM3D_TRIALS (default 500) sets trials per point; HM3D_QM9_STRIDE (default 1)
subsamples QM9. All randomness is seeded.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
import warnings

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

from hashmol3d import __version__, boundary_probes, hash_molecule, same_geometry
from hashmol3d.core import _ALL_AXIS_SIGNS, _FINE_FACTOR, _FINE_MARGIN, _parse_descriptor

EPS = 1e-4
RATIOS = (0.01, 0.03, 0.1)
TRIALS = int(os.environ.get("HM3D_TRIALS", "500"))
PANEL = os.environ.get("HM3D_NOISE_PANEL", os.path.join(_here, "noise_panel.npz"))
DATA = os.environ.get("HM3D_DATA", "/tmp/datasets")
QM9_STRIDE = int(os.environ.get("HM3D_QM9_STRIDE", "1"))
CORPUS_PRECISIONS = (1e-2, 1e-3, 1e-4)
CORPUS_MARGINS = (0.01, 0.05, 0.1)
rng = np.random.default_rng(20261103)
warnings.simplefilter("ignore", UserWarning)  # frame fallbacks are counted via the tag


def probe_margin(ratio):
    """Probe values within about three noise standard deviations of a boundary."""
    return min(0.5, max(0.05, 3 * ratio))


def offset_key(descriptor):
    """Canonical rows on a grid shifted by half a cell, as odd half-cell indices."""
    rows = np.array(_parse_descriptor(descriptor).rows, dtype=np.int64)
    z, fine = rows[:, :1], rows[:, 1:]
    half = 2 * np.floor_divide(fine, _FINE_FACTOR) + 1
    best = None
    for signs in _ALL_AXIS_SIGNS:
        k = np.column_stack([z, half * np.array(signs)])
        k = k[np.lexsort(k[:, ::-1].T)].tobytes()
        best = k if best is None or k < best else best
    return best


def timed(f, *args, **kwargs):
    t = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t


def probes_or_none(descriptor, margin, max_flips):
    try:
        return boundary_probes(descriptor, margin=margin, max_flips=max_flips)
    except ValueError:  # too many probes (max_keys) or a D descriptor
        return None


def noise_panel():
    panel = np.load(PANEL)
    names = sorted({k[:-2] for k in panel.files})
    out = []
    for name in names:
        Z, X = panel[name + "_Z"], panel[name + "_X"]
        ref = hash_molecule(Z, X, precision=EPS)
        ref_offset = offset_key(ref.descriptor) if "|F:" in ref.descriptor else None
        for ratio in RATIOS:
            m = probe_margin(ratio)
            c = dict.fromkeys(
                [
                    "exact",
                    "exact_or_offset",
                    "probe1",
                    "probe2",
                    "rmsd",
                    "flagged_split",
                    "probe_overflow",
                ],
                0,
            )
            t_hash = t_rmsd = 0.0
            t_probe = {1: 0.0, 2: 0.0}
            n_probe = {1: 0, 2: 0}
            probe_sizes = []
            for _ in range(TRIALS):
                res, dt = timed(
                    hash_molecule, Z, X + rng.normal(0.0, ratio * EPS, X.shape), precision=EPS
                )
                t_hash += dt
                if res.geometry_hash == ref.geometry_hash:
                    for key in ("exact", "exact_or_offset", "probe1", "probe2", "rmsd"):
                        c[key] += 1
                    continue
                c["flagged_split"] += min(ref.min_margin, res.min_margin) < m
                is_f = "|F:" in res.descriptor
                c["exact_or_offset"] += (
                    is_f and ref_offset is not None and offset_key(res.descriptor) == ref_offset
                )
                for j in (1, 2):
                    probes, dt = (
                        timed(probes_or_none, res.descriptor, m, j) if is_f else (None, 0.0)
                    )
                    if probes is None:
                        c["probe_overflow"] += j == 2
                    else:
                        t_probe[j] += dt
                        n_probe[j] += 1
                        if j == 2:
                            probe_sizes.append(len(probes))
                        c[f"probe{j}"] += ref.geometry_hash in probes
                same, dt = timed(same_geometry, ref.descriptor, res.descriptor)
                t_rmsd += dt
                c["rmsd"] += same
            splits = TRIALS - c["exact"]
            row = {
                "name": name,
                "N": len(Z),
                "ratio": ratio,
                "trials": TRIALS,
                "probe_margin": m,
                "ref_min_margin": f"{ref.min_margin:.6f}",
                "splits": splits,
                **{
                    f"match_{k}": c[k]
                    for k in ("exact", "exact_or_offset", "probe1", "probe2", "rmsd")
                },
                "flagged_splits": c["flagged_split"],
                "probe2_overflow": c["probe_overflow"],
                "median_probe2_keys": int(np.median(probe_sizes)) if probe_sizes else 0,
                "ms_per_hash": 1e3 * t_hash / TRIALS,
                **{f"ms_per_probe{j}_call": 1e3 * t_probe[j] / max(n_probe[j], 1) for j in (1, 2)},
                "ms_per_rmsd": 1e3 * t_rmsd / max(splits, 1),
            }
            out.append(row)
            print(
                f"{name:12s} N={len(Z):3d} s/e={ratio:<5g} split={splits / TRIALS:6.3f}  recovered: offset={row['match_exact_or_offset'] - c['exact']:4d} probe1={row['match_probe1'] - c['exact']:4d} probe2={row['match_probe2'] - c['exact']:4d} rmsd={row['match_rmsd'] - c['exact']:4d} of {splits:4d}  flagged={c['flagged_split']}",
                flush=True,
            )
    return out


def corpus_margins(path):
    from rdkit import Chem

    supp = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    mols = []
    for i, mol in enumerate(supp):
        if mol is None or mol.GetNumConformers() == 0 or i % QM9_STRIDE:
            continue
        mols.append(
            (
                np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
                mol.GetConformer().GetPositions(),
            )
        )
    out = []
    for precision in CORPUS_PRECISIONS:
        margins, at_risk, n_f = [], [], 0
        for Z, X in mols:
            res = hash_molecule(Z, X, precision=precision)
            margins.append(res.min_margin)
            if "|F:" in res.descriptor:
                n_f += 1
                fine = np.array(_parse_descriptor(res.descriptor).rows, dtype=np.int64)[:, 1:]
                off = np.abs(fine - np.rint(fine / _FINE_FACTOR).astype(np.int64) * _FINE_FACTOR)
                at_risk.append(
                    int(
                        np.sum(
                            _FINE_FACTOR // 2 - off <= max(_FINE_MARGIN, round(0.05 * _FINE_FACTOR))
                        )
                    )
                )  # boundary_probes() rule
        margins = np.array(margins)
        at_risk = np.array(at_risk)
        row = {
            "dataset": "QM9",
            "stride": QM9_STRIDE,
            "precision": precision,
            "geometries": len(mols),
            "F": n_f,
            **{f"frac_min_margin_below_{m:g}": float(np.mean(margins < m)) for m in CORPUS_MARGINS},
            "median_values_within_0.05": float(np.median(at_risk)),
            "median_probe2_keys_at_0.05": float(np.median([k + math.comb(k, 2) for k in at_risk])),
        }
        out.append(row)
        print(row, flush=True)
    return out


def write(rows, stem):
    path = os.path.join(_here, f"{stem}.csv")
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    print(f"HashMol3D {__version__}; trials per point: {TRIALS}")
    write(noise_panel(), "boundary_matching")
    qm9 = os.path.join(DATA, "gdb9.sdf")
    if os.path.exists(qm9):
        write(corpus_margins(qm9), "boundary_margin_corpus")
    else:
        print(f"skipping corpus margins: {qm9} not found")
