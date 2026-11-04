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

  B. Corpus margins (needs $HM3D_DATA/gdb9.sdf and/or tmQM_X{1,2,3}.xyz.gz).
     For QM9 and tmQM at several precisions, the fraction of geometries whose
     min_margin is below each margin, and the number of values within the
     probe margin, which sets the probe-set size.

  C. False merges (needs $HM3D_DATA/md17_*.npz and gdb9.sdf). same_geometry()
     with its default tolerance accepts two distinct geometries whenever their
     RMSD is within the precision. MD17 frames keep one atom order, so the
     exact optimal RMSD of two frames (rotations and reflections, identity
     correspondence) is known. Even consecutive frames are ~0.01 A apart,
     beyond every tolerance here, so pairs near the tolerance are made by
     stepping from a frame toward the next one until the exact RMSD is a given
     multiple of the precision (RMSD_RATIOS); consecutive frames ("lag1") are
     kept for context. For each pair and precision, record whether the hashes
     are equal, whether same_geometry() accepts the pair, the exact RMSD, and
     the RMSD upper bound it found. An acceptance with exact RMSD above the
     tolerance would be a false merge; a rejection with exact RMSD within it is
     a match missed by the correspondence heuristic. Random pairs of QM9
     constitutional isomers (same formula) check the far end.

Outputs: boundary_matching.csv (A), boundary_margin_corpus.csv (B) and
boundary_false_merges.csv (C). HM3D_PARTS (default "ABC") selects parts;
HM3D_TRIALS (default 500) sets trials per point; HM3D_QM9_STRIDE and
HM3D_TMQM_STRIDE (default 1) subsample the corpora; HM3D_PAIRS (default 500)
sets start frames per MD17 trajectory. Missing datasets are skipped. All randomness is
seeded.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import sys
import time
import warnings

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

from hashmol3d import (
    __version__,
    boundary_probes,
    hash_descriptor,
    hash_molecule,
    same_geometry,
)
from hashmol3d.core import (
    _ALL_AXIS_SIGNS,
    _FINE_DIGITS,
    _FINE_FACTOR,
    _FINE_MARGIN,
    _parse_descriptor,
    _rmsd_upper_bound,
)
from hashmol3d.periodic_table import get_atomic_num

EPS = 1e-4
RATIOS = (0.01, 0.03, 0.1)
TRIALS = int(os.environ.get("HM3D_TRIALS", "500"))
PANEL = os.environ.get("HM3D_NOISE_PANEL", os.path.join(_here, "noise_panel.npz"))
DATA = os.environ.get("HM3D_DATA", "/tmp/datasets")
QM9_STRIDE = int(os.environ.get("HM3D_QM9_STRIDE", "1"))
TMQM_STRIDE = int(os.environ.get("HM3D_TMQM_STRIDE", "1"))
TMQM_FILES = ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
PARTS = os.environ.get("HM3D_PARTS", "ABC")
PAIRS = int(os.environ.get("HM3D_PAIRS", "500"))
MD17 = ("aspirin", "benzene2017", "ethanol")
# Exact RMSD of the synthetic MD17 pairs, in units of the precision.
RMSD_RATIOS = (0.25, 0.5, 0.9, 1.1, 2.0, 4.0)
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


def load_qm9(path):
    from rdkit import Chem

    supp = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    for mol in supp:
        if mol is not None and mol.GetNumConformers():
            yield (
                np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
                mol.GetConformer().GetPositions(),
            )


def load_tmqm(paths):
    """tmQM multi-structure XYZ, as read by the paper's analysis_panel.py."""
    for path in paths:
        with gzip.open(path, "rt") as f:
            while True:
                head = f.readline()
                if not head:
                    break
                if not head.strip():
                    continue
                n = int(head.split()[0])
                f.readline()  # CSD code, charge, spin
                Z = np.empty(n, dtype=int)
                X = np.empty((n, 3))
                for i in range(n):
                    sym, x, y, z = f.readline().split()[:4]
                    Z[i] = get_atomic_num(sym)
                    X[i] = (float(x), float(y), float(z))
                yield Z, X


def at_risk_count(descriptor, margin):
    """Values within ``margin`` grid cells of a boundary, by the boundary_probes() rule."""
    fine = np.array(_parse_descriptor(descriptor).rows, dtype=np.int64)[:, 1:]
    off = np.abs(fine - np.rint(fine / _FINE_FACTOR).astype(np.int64) * _FINE_FACTOR)
    return int(np.sum(_FINE_FACTOR // 2 - off <= max(_FINE_MARGIN, round(margin * _FINE_FACTOR))))


def corpus_margins(dataset, mols, stride):
    out = []
    for precision in CORPUS_PRECISIONS:
        margins, at_risk, n_f = [], [], 0
        for Z, X in mols:
            res = hash_molecule(Z, X, precision=precision)
            margins.append(res.min_margin)
            if "|F:" in res.descriptor:
                n_f += 1
                at_risk.append(at_risk_count(res.descriptor, 0.05))
        margins = np.array(margins)
        row = {
            "dataset": dataset,
            "stride": stride,
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


def exact_rmsd(a, b):
    """Optimal RMSD over rotations and reflections, for a known correspondence."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    return float(np.sqrt(np.mean(np.sum((a @ (u @ vt) - b) ** 2, axis=1))))


def stored_coords(descriptor):
    parsed = _parse_descriptor(descriptor)
    rows = np.array(parsed.rows, dtype=np.int64)
    return rows[:, 0], rows[:, 1:] / 10.0 ** (parsed.decimals + _FINE_DIGITS)


def pair_row(source, kind, precision, Z, Xa, Xb):
    a = hash_molecule(Z, Xa, precision=precision)
    b = hash_molecule(Z, Xb, precision=precision)
    bound = float("nan")
    if "|F:" in a.descriptor and "|F:" in b.descriptor:
        za, ca = stored_coords(a.descriptor)
        bound = _rmsd_upper_bound(za, ca, stored_coords(b.descriptor)[1])
    return {
        "source": source,
        "kind": kind,
        "precision": precision,
        "equal_hash": a.geometry_hash == b.geometry_hash,
        "accepted": same_geometry(a.descriptor, b.descriptor),
        "exact_rmsd": exact_rmsd(Xa, Xb),
        "bound_rmsd": bound,
    }


def false_merges():
    pairs = []
    for name in MD17:
        path = os.path.join(DATA, f"md17_{name}.npz")
        if not os.path.exists(path):
            print(f"skipping {path}: not found")
            continue
        d = np.load(path)
        Z, R = np.asarray(d["z"], dtype=int), np.asarray(d["R"], dtype=float)
        starts = rng.choice(R.shape[0] - 1, size=PAIRS, replace=False)
        for precision in CORPUS_PRECISIONS:
            for i in starts:
                Xa, step = R[i], R[i + 1] - R[i]
                pairs.append(pair_row(f"MD17:{name}", "lag1", precision, Z, Xa, Xa + step))
                full = exact_rmsd(Xa, Xa + step)
                for c in RMSD_RATIOS:
                    # The RMSD of a short step along a fixed direction is linear in its length.
                    Xb = Xa + step * (c * precision / full)
                    pairs.append(pair_row(f"MD17:{name}", f"rmsd_{c:g}eps", precision, Z, Xa, Xb))
    qm9 = os.path.join(DATA, "gdb9.sdf")
    if os.path.exists(qm9):
        by_formula = {}
        for Z, X in load_qm9(qm9):
            by_formula.setdefault(tuple(sorted(Z)), []).append((Z, X))
        groups = [g for g in by_formula.values() if len(g) > 1]
        for _ in range(PAIRS):
            g = groups[rng.integers(len(groups))]
            i, j = rng.choice(len(g), size=2, replace=False)
            (Za, Xa), (Zb, Xb) = g[i], g[j]
            for precision in CORPUS_PRECISIONS:
                a = hash_molecule(Za, Xa, precision=precision).descriptor
                b = hash_molecule(Zb, Xb, precision=precision).descriptor
                bound = float("nan")
                if "|F:" in a and "|F:" in b:
                    za, ca = stored_coords(a)
                    bound = _rmsd_upper_bound(za, ca, stored_coords(b)[1])
                # Isomers have no common atom order, so no exact RMSD.
                pairs.append(
                    {
                        "source": "QM9:isomers",
                        "kind": "random",
                        "precision": precision,
                        "equal_hash": hash_descriptor(a) == hash_descriptor(b),
                        "accepted": same_geometry(a, b),
                        "exact_rmsd": float("nan"),
                        "bound_rmsd": bound,
                    }
                )
    summary = []
    keys = list(dict.fromkeys((r["source"], r["kind"], r["precision"]) for r in pairs))
    for source, kind, precision in keys:
        rows = [
            r
            for r in pairs
            if r["source"] == source and r["kind"] == kind and r["precision"] == precision
        ]
        exact = np.array([r["exact_rmsd"] for r in rows])
        bound = np.array([r["bound_rmsd"] for r in rows])
        acc = np.array([r["accepted"] for r in rows])
        row = {
            "source": source,
            "kind": kind,
            "precision": precision,
            "pairs": len(rows),
            "equal_hash": sum(r["equal_hash"] for r in rows),
            "accepted": int(acc.sum()),
            "exact_rmsd_median": float(np.nanmedian(exact)) if np.isfinite(exact).any() else "",
            "exact_within_tol": int(np.sum(exact <= precision)),
            # Acceptance beyond the tolerance; the stored rows carry 1e-4 cell of rounding.
            "accepted_exact_above_tol": int(np.sum(acc & (exact > precision * (1 + 1e-3)))),
            "missed_exact_within_tol": int(np.sum(~acc & (exact <= precision * (1 - 1e-3)))),
            "bound_rmsd_min": float(np.nanmin(bound)) if np.isfinite(bound).any() else "",
            "bound_over_exact_max": float(np.nanmax(bound / exact))
            if np.isfinite(exact).any()
            else "",
        }
        summary.append(row)
        print(row, flush=True)
    return summary


def write(rows, stem):
    path = os.path.join(_here, f"{stem}.csv")
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    print(f"HashMol3D {__version__}; trials per point: {TRIALS}")
    if "A" in PARTS:
        write(noise_panel(), "boundary_matching")
    if "B" in PARTS:
        corpus = []
        qm9 = os.path.join(DATA, "gdb9.sdf")
        if os.path.exists(qm9):
            mols = [m for i, m in enumerate(load_qm9(qm9)) if i % QM9_STRIDE == 0]
            corpus += corpus_margins("QM9", mols, QM9_STRIDE)
        tmqm = [os.path.join(DATA, f) for f in TMQM_FILES]
        if all(os.path.exists(p) for p in tmqm):
            mols = [m for i, m in enumerate(load_tmqm(tmqm)) if i % TMQM_STRIDE == 0]
            corpus += corpus_margins("tmQM", mols, TMQM_STRIDE)
        if corpus:
            write(corpus, "boundary_margin_corpus")
        else:
            print(f"skipping corpus margins: no datasets in {DATA}")
    if "C" in PARTS:
        write(false_merges(), "boundary_false_merges")
