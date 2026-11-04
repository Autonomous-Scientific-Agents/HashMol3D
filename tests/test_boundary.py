"""Boundary-aware matching: rounding margins, probe keys and RMSD confirmation.

``hash_molecule`` reports how close the hashed values come to a rounding
boundary (``min_margin``). ``boundary_probes`` lists the geometry hashes of
the neighboring cells of near-boundary values in a stored ``F`` descriptor,
and ``same_geometry`` confirms a match from two stored descriptors. None of
them changes an identifier.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

import hashmol3d.core as core
from hashmol3d import (
    DESCRIPTOR_VERSION,
    boundary_probes,
    hash_descriptor,
    hash_molecule,
    same_geometry,
)

PRECISION = 1e-4
FACTOR = core._FINE_FACTOR


@pytest.fixture
def cloud():
    """An asymmetric 12-atom cloud on the principal-axes frame path."""
    rng = np.random.default_rng(7)
    z = np.array([6, 6, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1])
    return z, rng.normal(scale=1.5, size=(12, 3)) * [1.0, 0.8, 0.6]


def _motion(z, coords, rng):
    perm = rng.permutation(len(z))
    rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return z[perm], (coords @ rot.T + rng.uniform(-5, 5, size=3))[perm]


def _stored_rows(descriptor):
    return np.array(core._parse_frame_rows(descriptor.split("|F:", 1)[1]), dtype=np.int64)


def _with_rows(descriptor, rows):
    """``descriptor`` with its F rows replaced, re-sorted into canonical order."""
    coarse = core._coarsen(rows[:, 1:])
    keys = np.column_stack([rows[:, 0], coarse, rows[:, 1:]])
    rows = rows[np.lexsort(keys[:, ::-1].T)]
    body = ";".join(f"{r[0]}:{r[1]},{r[2]},{r[3]}" for r in rows.tolist())
    return descriptor.split("|F:", 1)[0] + "|F:" + body


def _across_nearest_boundary(descriptor, inside=10):
    """``descriptor`` with its value nearest a boundary moved just across it,
    ``inside`` fine units into the neighboring cell."""
    rows = _stored_rows(descriptor)
    fine = rows[:, 1:]
    offset = fine - core._coarsen(fine) * FACTOR
    i, j = np.unravel_index(np.argmax(np.abs(offset)), offset.shape)
    step = 1 if offset[i, j] > 0 else -1
    moved = rows.copy()
    moved[i, 1 + j] = (core._coarsen(fine)[i, j] + step) * FACTOR - step * (FACTOR // 2 - inside)
    return _with_rows(descriptor, moved)


class TestMinMargin:
    def test_frame_margin_matches_the_stored_rows(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        assert "|F:" in res.descriptor
        fine = _stored_rows(res.descriptor)[:, 1:]
        offset = fine - core._coarsen(fine) * FACTOR
        stored = (FACTOR // 2 - np.abs(offset)).min() / FACTOR
        # Stored values are clamped _FINE_MARGIN fine units inside their cell.
        assert 0.0 <= res.min_margin <= 0.5
        assert stored == max(res.min_margin, core._FINE_MARGIN / FACTOR)

    def test_distance_margin_is_that_of_the_scaled_distances(self, cloud):
        z, coords = cloud
        res = hash_molecule(z, coords, precision=PRECISION, method="distance")
        iu = np.triu_indices(len(z), k=1)
        scaled = np.linalg.norm(coords[:, None] - coords[None], axis=-1)[iu] / PRECISION
        assert res.min_margin == pytest.approx(0.5 - np.abs(scaled - np.rint(scaled)).max())

    @pytest.mark.parametrize("method", ["frame", "distance"])
    def test_a_single_atom_has_no_value_near_a_boundary(self, method):
        res = hash_molecule([6], [[1.23456789, 2.0, -3.0]], method=method)
        assert res.min_margin == 0.5

    def test_margin_does_not_enter_result_equality(self, cloud):
        res = hash_molecule(*cloud)
        assert replace(res, min_margin=0.123) == res
        assert hash(replace(res, min_margin=0.123)) == hash(res)


class TestBoundaryProbes:
    def test_probes_recover_a_value_moved_across_a_boundary(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        moved = _across_nearest_boundary(res.descriptor)
        assert hash_descriptor(moved) != res.geometry_hash
        assert res.geometry_hash in boundary_probes(moved, margin=0.01, max_flips=1)

    @pytest.mark.parametrize("margin", [1e-4, core._FINE_MARGIN / FACTOR, 1e-3])
    def test_clamped_values_are_probed_at_any_margin(self, cloud, margin):
        # A value on the clamp limit, the closest a stored value can be to a
        # boundary, belongs to the neighboring cell of the original.
        res = hash_molecule(*cloud, precision=PRECISION)
        moved = _across_nearest_boundary(res.descriptor, inside=core._FINE_MARGIN)
        assert res.geometry_hash in boundary_probes(moved, margin=margin, max_flips=1)

    def test_margin_is_inclusive_in_fine_units(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        moved = _across_nearest_boundary(res.descriptor, inside=290)
        assert res.geometry_hash in boundary_probes(moved, margin=0.029, max_flips=1)
        assert res.geometry_hash not in boundary_probes(moved, margin=0.0289, max_flips=1)

    def test_probes_exclude_the_own_hash_and_duplicates(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        probes = boundary_probes(res.descriptor, margin=0.3, max_flips=2)
        assert probes
        assert res.geometry_hash not in probes
        assert len(set(probes)) == len(probes)

    def test_probe_count_follows_the_values_within_the_margin(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        fine = _stored_rows(res.descriptor)[:, 1:]
        offset = fine - core._coarsen(fine) * FACTOR
        k = int(np.sum(FACTOR // 2 - np.abs(offset) <= 0.2 * FACTOR))
        # One flip moves one value: every result is a distinct cell.
        assert len(boundary_probes(res.descriptor, margin=0.2, max_flips=1)) == k

    def test_length(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION, length=12)
        moved = _across_nearest_boundary(res.descriptor)
        probes = boundary_probes(moved, margin=0.01, max_flips=1, length=12)
        assert res.geometry_hash in probes
        assert all(len(p) == 12 for p in probes)

    def test_too_many_probes_is_an_error(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        with pytest.raises(ValueError, match="max_keys"):
            boundary_probes(res.descriptor, margin=0.5, max_flips=3, max_keys=100)

    def test_distance_descriptors_cannot_be_probed(self, cloud):
        res = hash_molecule(*cloud, method="distance")
        with pytest.raises(ValueError, match="F descriptor"):
            boundary_probes(res.descriptor)

    def test_other_versions_cannot_be_probed(self, cloud):
        res = hash_molecule(*cloud)
        old = res.descriptor.replace(f"V:{DESCRIPTOR_VERSION}", "V:11.0")
        with pytest.raises(ValueError, match="version"):
            boundary_probes(old)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"margin": 0.0},
            {"margin": 0.6},
            {"margin": True},
            {"margin": "0.1"},
            {"max_flips": 0},
            {"max_flips": 1.5},
            {"max_keys": 0},
            {"length": 0},
        ],
    )
    def test_invalid_options(self, cloud, kwargs):
        res = hash_molecule(*cloud)
        with pytest.raises(ValueError):
            boundary_probes(res.descriptor, **kwargs)


class TestSameGeometry:
    def test_equal_payloads_are_the_same_geometry(self, cloud):
        rng = np.random.default_rng(1)
        a = hash_molecule(*cloud).descriptor
        b = hash_molecule(*_motion(*cloud, rng)).descriptor
        assert same_geometry(a, b)

    def test_a_split_by_a_boundary_is_confirmed(self, cloud):
        res = hash_molecule(*cloud, precision=PRECISION)
        moved = _across_nearest_boundary(res.descriptor)
        assert hash_descriptor(moved) != res.geometry_hash
        assert same_geometry(res.descriptor, moved)

    def test_noisy_copies_are_confirmed(self, cloud):
        z, coords = cloud
        rng = np.random.default_rng(3)
        ref = hash_molecule(z, coords, precision=PRECISION)
        split = 0
        for _ in range(40):
            noisy = coords + rng.normal(scale=0.03 * PRECISION, size=coords.shape)
            other = hash_molecule(*_motion(z, noisy, rng), precision=PRECISION)
            split += other.geometry_hash != ref.geometry_hash
            assert same_geometry(ref.descriptor, other.descriptor)
        assert split  # the noise does cross boundaries

    def test_distinct_geometries_are_not_the_same(self, cloud):
        z, coords = cloud
        moved = coords.copy()
        moved[0] += [0.01, 0.0, 0.0]
        a = hash_molecule(z, coords, precision=PRECISION).descriptor
        b = hash_molecule(z, moved, precision=PRECISION).descriptor
        assert not same_geometry(a, b)
        assert same_geometry(a, b, rmsd_tol=0.01)

    def test_different_elements_are_not_the_same(self, cloud):
        z, coords = cloud
        a = hash_molecule(z, coords).descriptor
        b = hash_molecule(np.where(z == 8, 16, z), coords).descriptor
        assert not same_geometry(a, b, rmsd_tol=1.0)

    def test_unequal_distance_descriptors_are_not_confirmed(self, cloud):
        z, coords = cloud
        a = hash_molecule(z, coords, method="distance").descriptor
        moved = coords.copy()
        moved[0] += [1e-3, 0.0, 0.0]
        b = hash_molecule(z, moved, method="distance").descriptor
        assert same_geometry(a, a)
        assert not same_geometry(a, b, rmsd_tol=1.0)

    def test_precisions_must_match(self, cloud):
        a = hash_molecule(*cloud, precision=1e-3).descriptor
        b = hash_molecule(*cloud, precision=1e-4).descriptor
        with pytest.raises(ValueError, match="precisions"):
            same_geometry(a, b)

    @pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
    def test_invalid_tolerance(self, cloud, tol):
        a = hash_molecule(*cloud).descriptor
        with pytest.raises(ValueError, match="rmsd_tol"):
            same_geometry(a, a, rmsd_tol=tol)


def test_rmsd_bound_finds_permuted_reflected_copies(cloud):
    z, coords = cloud
    rng = np.random.default_rng(5)
    order = np.argsort(z, kind="stable")
    z, coords = z[order], coords[order]
    # Shuffle atoms within each element, flip an axis and rotate slightly,
    # as two canonical frames of one geometry can differ.
    perm = np.concatenate([rng.permutation(np.flatnonzero(z == v)) for v in np.unique(z)])
    t = 0.02
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, -1.0]])
    other = coords[perm] @ rot.T
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert core._rmsd_upper_bound(z, coords, other) < 1e-9
