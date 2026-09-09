"""Reproduce QM9 principal-axis and near-line path decisions without a dataset download.

The JSON retains four-decimal SDF coordinates from the checksum-pinned corpus.
Gap-threshold sweeps are diagnostics, not alternative version-8 settings.
Run: python paper/analysis_frame_cases.py (requires HashMol3D and NumPy).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hashmol3d import core


def diagnose(case, precision, threshold):
    z = np.asarray(case["atomic_numbers"], dtype=np.int64)
    coords = np.asarray(case["coordinates"], dtype=float)
    w = z.astype(float)

    def ksum(values):
        return np.sum(np.sort(values))

    c = coords - np.array([ksum(w * coords[:, k]) for k in range(3)]) / ksum(w)
    tensor = np.array([[ksum(w * c[:, a] * c[:, b]) for b in range(3)] for a in range(3)])
    lam, vec = np.linalg.eigh(tensor)
    axis = vec[:, 2]
    radius = float(np.linalg.norm(c, axis=1).max())
    transverse = float(np.linalg.norm(c - np.outer(c @ axis, axis), axis=1).max())
    original = core._FRAME_GAP_MIN
    try:
        core._FRAME_GAP_MIN = threshold
        signature = core._frame_signature(z, coords, round(-np.log10(precision)))
    finally:
        core._FRAME_GAP_MIN = original
    tag = "C" if signature is None else signature[0]
    return {
        "qm9_id": case["qm9_id"],
        "formula": case["formula"],
        "descriptor_version": core.DESCRIPTOR_VERSION,
        "precision_angstrom": f"{precision:g}",
        "gap_threshold": f"{threshold:g}",
        "max_radius_angstrom": f"{radius:.12g}",
        "max_transverse_angstrom": f"{transverse:.12g}",
        "small_relative_gap": f"{(lam[1] - lam[0]) / lam[2]:.12g}",
        "version_7_path": case["version_7_paths"][f"{precision:g}"] if threshold == 0.05 else "",
        "path": tag,
    }


def main():
    here = Path(__file__).resolve().parent
    data = json.loads((here / "qm9_frame_cases.json").read_text())
    rows = []
    for case in data["cases"]:
        for precision in (1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6):
            rows.append(diagnose(case, precision, 0.05))
        for threshold in (0.01, 0.001, 1e-6, 1e-8, 1e-10):
            rows.append(diagnose(case, 1e-4, threshold))
    output = here / "qm9_frame_cases.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} diagnostics to {output}")


if __name__ == "__main__":
    main()
