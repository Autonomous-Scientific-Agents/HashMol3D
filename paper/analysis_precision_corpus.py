"""Run and aggregate the full-corpus HashMol3D precision sweep.

Each precision is evaluated by ``analysis_collisions.py`` on the complete
manifest.  Full-digest repetitions and truncated-prefix collisions are kept as
separate statistics.  Use ``--aggregate-only`` after existing per-precision
runs to rebuild only the two compact CSV products used by the paper.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRECISIONS = (
    ("1", "1"),
    ("1e-1", "1e-1"),
    ("1e-2", "1e-2"),
    ("1e-3", "1e-3"),
    ("1e-4", "1e-4"),
    ("1e-5", "1e-5"),
)
BITS = (24, 32, 40, 48, 64, 128)


def _read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_sweep():
    for precision, suffix in PRECISIONS:
        env = os.environ.copy()
        env.update(
            HM3D_METHOD="frame",
            HM3D_PRECISION=precision,
            HM3D_AUDIT_BRANCHES="0",
            HM3D_SKIP_FIGURE="1",
            HM3D_OUTPUT_SUFFIX=f"_precision_{suffix}",
        )
        print(f"\n=== precision {precision} Angstrom ===", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / "analysis_collisions.py")],
            cwd=HERE,
            env=env,
            check=True,
        )


def aggregate():
    summary_fields = [
        "precision_angstrom",
        "geometries",
        "distinct_full_digests",
        "precision_merge_excess",
        "precision_merge_percent",
        "tag_F",
        "tag_C",
        "tag_W",
        *(f"truncation_collisions_{bits}bit" for bits in BITS),
    ]
    dataset_fields = [
        "precision_angstrom",
        "dataset",
        "geometries",
        "distinct_full_digests",
        "precision_merge_excess",
        "precision_merge_percent",
    ]

    with (HERE / "precision_sweep.csv").open("w", newline="") as summary_file, (
        HERE / "precision_sweep_by_dataset.csv"
    ).open("w", newline="") as dataset_file:
        summary_writer = csv.DictWriter(
            summary_file, fieldnames=summary_fields, lineterminator="\n"
        )
        dataset_writer = csv.DictWriter(
            dataset_file, fieldnames=dataset_fields, lineterminator="\n"
        )
        summary_writer.writeheader()
        dataset_writer.writeheader()

        for precision, suffix in PRECISIONS:
            stats = _read_rows(HERE / f"dataset_stats_precision_{suffix}.csv")
            total = next(row for row in stats if row["dataset"] == "TOTAL")
            collisions = {
                int(row["bits"]): row
                for row in _read_rows(
                    HERE / f"collision_results_precision_{suffix}.csv"
                )
            }
            n_total = int(total["geometries"])
            n_distinct = int(total["distinct_full_digests"])
            excess = n_total - n_distinct
            if any(
                int(collisions[bits]["distinct_geometries"]) != n_distinct
                for bits in BITS
            ):
                raise ValueError(f"inconsistent digest count at precision {precision}")

            summary_writer.writerow(
                {
                    "precision_angstrom": precision,
                    "geometries": n_total,
                    "distinct_full_digests": n_distinct,
                    "precision_merge_excess": excess,
                    "precision_merge_percent": f"{100 * excess / n_total:.6f}",
                    "tag_F": total["tag_F"],
                    "tag_C": total["tag_C"],
                    "tag_W": total["tag_W"],
                    **{
                        f"truncation_collisions_{bits}bit": collisions[bits][
                            "collisions"
                        ]
                        for bits in BITS
                    },
                }
            )

            for row in stats:
                if row["dataset"] == "TOTAL":
                    continue
                count = int(row["geometries"])
                distinct = int(row["distinct_full_digests"])
                merge_excess = count - distinct
                dataset_writer.writerow(
                    {
                        "precision_angstrom": precision,
                        "dataset": row["dataset"],
                        "geometries": count,
                        "distinct_full_digests": distinct,
                        "precision_merge_excess": merge_excess,
                        "precision_merge_percent": (
                            f"{100 * merge_excess / count:.6f}"
                        ),
                    }
                )

    print("Wrote precision_sweep.csv")
    print("Wrote precision_sweep_by_dataset.csv")


def main():
    unknown = set(sys.argv[1:]) - {"--aggregate-only"}
    if unknown:
        raise SystemExit(f"unknown arguments: {sorted(unknown)}")
    if "--aggregate-only" not in sys.argv:
        run_sweep()
    aggregate()


if __name__ == "__main__":
    main()
