"""Tests for the rounding-edge margin diagnostic.

``HashMol3DResult.min_margin`` reports how far the quantized values that
entered a descriptor sit from a ``rint`` decision boundary, in grid units. It
is a stability diagnostic and never part of the hashed payload: a geometry
whose margin is near zero can change identifier under nothing worse than
float64 round-off, which is exactly the fragility the spec's §2 warns about
without quantifying.
"""

import warnings

import numpy as np
import pytest

from hashmol3d import hash_molecule
from hashmol3d.core import _edge_margin


def _hash(z, coords, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return hash_molecule(np.asarray(z), np.asarray(coords, dtype=float), **kwargs)


def _rigid(z, coords, rng, reflect):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    moved = coords @ q.T + rng.normal(0.0, 4.0, 3)
    if reflect:
        moved = moved * np.array([1.0, 1.0, -1.0])
    order = rng.permutation(len(z))
    return np.asarray(z)[order], moved[order]


class TestEdgeMargin:
    def test_cell_centre_is_maximally_safe(self):
        assert _edge_margin(np.array([0.0, 3.0, -7.0])) == pytest.approx(0.5)

    def test_exact_edge_is_zero(self):
        assert _edge_margin(np.array([2.5])) == pytest.approx(0.0)
        assert _edge_margin(np.array([-2.5])) == pytest.approx(0.0)

    def test_nothing_to_round_reports_safe(self):
        assert _edge_margin(np.array([])) == pytest.approx(0.5)

    def test_minimum_over_all_values(self):
        assert _edge_margin(np.array([0.0, 0.4, 10.0])) == pytest.approx(0.1)


class TestMarginRange:
    @pytest.mark.parametrize("method", ["frame", "canonical"])
    @pytest.mark.parametrize("precision", [1.0, 1e-2, 1e-4])
    def test_within_bounds(self, method, precision):
        rng = np.random.default_rng(11)
        for _ in range(8):
            z = rng.integers(1, 18, 9)
            coords = rng.normal(0.0, 1.5, (9, 3))
            res = _hash(z, coords, precision=precision, method=method)
            assert 0.0 <= res.min_margin <= 0.5

    @pytest.mark.parametrize("method", ["frame", "canonical"])
    def test_single_atom_has_nothing_at_risk(self, method):
        assert _hash([26], [[1.0, 2.0, 3.0]], method=method).min_margin == pytest.approx(0.5)

    def test_integral_grid_coordinates_are_safe(self, water):
        # Every water coordinate is an exact multiple of 1e-4, so at the default
        # precision the scaled values are integers: cell centres.
        assert _hash(*water).min_margin == pytest.approx(0.5)


class TestMarginFlagsKnownFragileGeometries:
    def test_axial_coordinate_on_an_edge(self):
        # Centred axial coordinates are +/-0.05 A; at a 0.1 A grid that is
        # exactly +/-0.5 grid units, so round-half-to-even decides the cell.
        res = _hash([6, 6], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], precision=0.1)
        assert "|F:" in res.descriptor
        assert res.min_margin < 1e-9

    def test_distance_on_an_edge(self):
        # A 0.15 A separation scales to 1.5 grid units at a 0.1 A grid.
        res = _hash([6, 6], [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], precision=0.1, method="canonical")
        assert res.min_margin == pytest.approx(0.0)

    def test_idealized_ring_radius_on_an_edge(self):
        # 1.39 * cos(60 deg) = 0.695 A is a half-multiple of a 0.01 A grid, so
        # an idealized D6h ring is fragile there while a generic one is not.
        angles = np.arange(6) * np.pi / 3.0
        coords = np.vstack(
            [
                np.column_stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)]),
                np.column_stack([2.47 * np.cos(angles), 2.47 * np.sin(angles), np.zeros(6)]),
            ]
        )
        z = [6] * 6 + [1] * 6
        assert _hash(z, coords, precision=1e-2).min_margin < 1e-9

    def test_low_margin_predicts_an_unstable_identifier(self):
        """The flagged geometry actually flips; a safe one does not."""
        rng = np.random.default_rng(4)
        fragile = _hash([6, 6], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], precision=0.1)
        assert fragile.min_margin < 1e-9
        flips = 0
        for trial in range(60):
            zz, cc = _rigid([6, 6], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], rng, trial % 2 == 1)
            flips += _hash(zz, cc, precision=0.1).hash_str != fragile.hash_str
        assert flips > 0

        safe = _hash([6, 6], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], precision=0.1)
        assert safe.min_margin > 0.1
        for trial in range(60):
            zz, cc = _rigid([6, 6], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], rng, trial % 2 == 1)
            assert _hash(zz, cc, precision=0.1).hash_str == safe.hash_str


class TestMarginIsAGeometryProperty:
    @pytest.mark.parametrize("method", ["frame", "canonical"])
    def test_invariant_under_rigid_motion_and_relabeling(self, method):
        rng = np.random.default_rng(19)
        z = np.array([6, 1, 1, 1, 8])
        coords = rng.normal(0.0, 1.4, (5, 3))
        base = _hash(z, coords, method=method)
        for trial in range(12):
            zz, cc = _rigid(z, coords, rng, trial % 2 == 1)
            other = _hash(zz, cc, method=method)
            assert other.descriptor == base.descriptor
            assert other.min_margin == pytest.approx(base.min_margin, abs=1e-9)

    def test_not_part_of_the_hashed_payload(self, water):
        res = _hash(*water)
        assert "margin" not in res.descriptor
        assert f"{res.min_margin}" not in res.descriptor


class TestDegenerateFrameIsOrderIndependent:
    """Guards the tied-anchor rejection path in the fully degenerate branch.

    Tied first anchors share an invariant key but not necessarily the
    transverse extent measured about them, so rejecting one anchor must not
    reject the branch -- otherwise the F-versus-C decision would depend on the
    caller's atom order.
    """

    @pytest.mark.parametrize("precision", [1.0, 1e-1, 1e-2, 1e-4])
    def test_spherical_top_relabelings_agree(self, precision):
        a = 1.087 / np.sqrt(3.0)
        z = np.array([6, 1, 1, 1, 1])
        coords = np.array(
            [[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], dtype=float
        )
        rng = np.random.default_rng(23)
        base = _hash(z, coords, precision=precision)
        for trial in range(40):
            zz, cc = _rigid(z, coords, rng, trial % 2 == 1)
            assert _hash(zz, cc, precision=precision).descriptor == base.descriptor

    @pytest.mark.parametrize("precision", [1.0, 1e-2])
    def test_octahedral_shell_relabelings_agree(self, precision):
        z = np.array([16] + [9] * 6)
        coords = np.vstack([np.zeros((1, 3)), 1.564 * np.eye(3), -1.564 * np.eye(3)])
        rng = np.random.default_rng(29)
        base = _hash(z, coords, precision=precision)
        for trial in range(40):
            zz, cc = _rigid(z, coords, rng, trial % 2 == 1)
            assert _hash(zz, cc, precision=precision).descriptor == base.descriptor
