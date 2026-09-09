"""Tests for the canonical frame hashing method.

The frame method hashes coordinates in the eigenbasis of the Z-weighted
gyration tensor when its eigenvalues are separated. Degenerate eigenspaces
are resolved by intrinsic point/line descriptors or canonical atom anchors.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import hashmol3d.core as core
from hashmol3d import hash_molecule


def _scramble(z, coords, rng, reflect):
    perm = rng.permutation(len(z))
    rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    c = coords @ rot.T + rng.uniform(-5, 5, size=3)
    if reflect:
        c = c.copy()
        c[:, 0] *= -1
    return z[perm], c[perm]


def _chain(n, seed):
    """Random-walk chain, scaled anisotropically so the principal moments
    are guaranteed well separated (individual random walks can come out
    rod-like and legitimately fail the gap diagnostic)."""
    rng = np.random.default_rng(seed)
    z = rng.choice([1, 6, 7, 8, 16], size=n)
    c = rng.uniform(-1, 1, (n, 3)).cumsum(axis=0) * 0.8
    return z, c * np.array([1.0, 0.6, 0.35])


class TestFrameMethod:
    def test_frame_is_default(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        assert (
            hash_molecule(z, coords).descriptor
            == hash_molecule(z, coords, method="frame").descriptor
        )

    def test_uses_frame_tag_for_asymmetric(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        res = hash_molecule(z, coords, method="frame")
        assert "|F:" in res.descriptor
        assert "|C:" not in res.descriptor

    def test_invariance_including_reflection(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        base = hash_molecule(z, coords, method="frame").geometry_hash
        rng = np.random.default_rng(3)
        for t in range(40):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc, method="frame").geometry_hash == base

    def test_invariance_large_chain(self):
        z, coords = _chain(500, seed=7)
        res = hash_molecule(z, coords, method="frame")
        assert "|F:" in res.descriptor  # the frame path must actually be taken
        rng = np.random.default_rng(8)
        for t in range(5):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc, method="frame").geometry_hash == res.geometry_hash

    def test_sub_precision_noise_stable(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        base = hash_molecule(z, coords, method="frame").geometry_hash
        rng = np.random.default_rng(11)
        noisy = coords + rng.uniform(-1e-9, 1e-9, coords.shape)
        assert hash_molecule(z, noisy, method="frame").geometry_hash == base

    def test_distinct_geometries_distinct_hashes(self):
        za, ca = _chain(60, seed=1)
        zb, cb = _chain(60, seed=2)
        a = hash_molecule(za, ca, method="frame").geometry_hash
        b = hash_molecule(zb, cb, method="frame").geometry_hash
        assert a != b

    def test_not_comparable_with_canonical(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        f = hash_molecule(z, coords, method="frame")
        c = hash_molecule(z, coords, method="canonical")
        assert f.geometry_hash != c.geometry_hash
        assert "|F:" in f.descriptor and "|C:" in c.descriptor

    def test_invalid_method_rejected(self, water):
        z, coords = water
        with pytest.raises(ValueError):
            hash_molecule(z, coords, method="inertia")


class TestDegenerateFrames:
    def _assert_frame_invariant(self, z, coords, seed=17):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = hash_molecule(z, coords, method="frame")
        assert "|F:" in res.descriptor
        rng = np.random.default_rng(seed)
        for t in range(30):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc, method="frame").geometry_hash == res.geometry_hash
        return res

    def test_benzene_symmetric_top(self, benzene):
        self._assert_frame_invariant(*benzene)

    def test_prolate_symmetric_top_anchors_transverse_plane(self):
        coords = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, -3.0],
            ]
        )
        self._assert_frame_invariant(np.full(6, 6), coords)

    def test_linear_co2(self):
        z = np.array([8, 6, 8])
        coords = np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]], float)
        res = self._assert_frame_invariant(z, coords)
        rows = res.descriptor.split("|F:", 1)[1].split(";")
        assert all(row.split(":", 1)[1].split(",")[:2] == ["0", "0"] for row in rows)

    def test_diatomic_uses_intrinsic_line(self):
        z = np.array([1, 35])
        coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
        self._assert_frame_invariant(z, coords)

    @pytest.mark.parametrize("precision", [1.0, 0.1, 0.01, 1e-4, 1e-6])
    @pytest.mark.parametrize("extent", [0.2, 2.0])
    def test_exact_line_precedes_anchor_size_guard(self, precision, extent, monkeypatch):
        z = np.array([6, 6, 6])
        coords = np.array([[-extent / 2, 0, 0], [0, 0, 0], [extent / 2, 0, 0]])

        def unexpected_distance_matrix(*args, **kwargs):
            pytest.fail("an exact line must not allocate a distance matrix")

        monkeypatch.setattr(core, "_scaled_distances", unexpected_distance_matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            base = hash_molecule(z, coords, precision=precision, node_budget=1)
            assert "|F:" in base.descriptor
            rng = np.random.default_rng(71)
            for t in range(10):
                zz, cc = _scramble(z, coords, rng, reflect=t % 2 == 1)
                other = hash_molecule(zz, cc, precision=precision, node_budget=1)
                assert other.descriptor == base.descriptor

    def test_short_line_keeps_resolved_axial_coordinates(self):
        res = hash_molecule([6, 6], [[-1, 0, 0], [1, 0, 0]], precision=1.0)
        assert res.descriptor.endswith("|F:6:0,0,-1;6:0,0,1")

    def test_methane_spherical_top(self):
        a = 1.09 / np.sqrt(3)
        coords = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], float)
        self._assert_frame_invariant(np.array([6, 1, 1, 1, 1]), coords)

    def test_asymmetric_isotropic_tensor_uses_canonical_pair(self):
        rng = np.random.default_rng(29)
        coords = rng.normal(size=(7, 3))
        coords -= coords.mean(axis=0)
        lam, vec = np.linalg.eigh(coords.T @ coords)
        coords = coords @ vec @ np.diag(lam**-0.5) @ vec.T
        self._assert_frame_invariant(np.full(7, 6), coords)

    @pytest.mark.parametrize("geometry", ["benzene", "methane"])
    def test_anchor_branches_are_stable_to_small_noise(self, geometry, benzene):
        if geometry == "benzene":
            z, coords = benzene
        else:
            a = 1.09 / np.sqrt(3)
            z = np.array([6, 1, 1, 1, 1])
            coords = np.array(
                [[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
                float,
            )
        base = hash_molecule(z, coords, method="frame").geometry_hash
        rng = np.random.default_rng(23)
        for _ in range(20):
            noisy = coords + rng.uniform(-1e-7, 1e-7, coords.shape)
            assert hash_molecule(z, noisy, method="frame").geometry_hash == base

    def test_near_degenerate_uses_anchor(self, benzene):
        z, coords = benzene
        coords = coords.copy()
        coords[0, :2] *= 1 + 2e-4 / 1.4
        self._assert_frame_invariant(z, coords)

    def test_single_atom_uses_intrinsic_point(self):
        z = np.array([6])
        coords = np.zeros((1, 3))
        res = self._assert_frame_invariant(z, coords)
        assert res.descriptor.endswith("|F:6:0,0,0")


class TestQM9NearLines:
    @pytest.mark.parametrize(
        "case_id, expected",
        [("gdb_25", "CFF"), ("gdb_14564", "CFF"), ("gdb_5", "FCF")],
    )
    def test_precision_changes_which_molecule_needs_c(self, case_id, expected):
        path = Path(__file__).resolve().parents[1] / "paper" / "qm9_frame_cases.json"
        cases = json.loads(path.read_text())["cases"]
        case = next(case for case in cases if case["qm9_id"] == case_id)
        for precision, tag in zip((1e-4, 1e-5, 1e-6), expected):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = hash_molecule(
                    case["atomic_numbers"], case["coordinates"], precision=precision
                )
            assert result.descriptor.rsplit("|", 1)[1].startswith(tag + ":")


class TestFrameFallback:
    def test_ill_conditioned_anchor_falls_back(self):
        z = np.array([6, 6, 6])
        coords = np.array([[-1.0, 0.0, 0.0], [0.0, 0.01, 0.0], [1.0, 0.0, 0.0]])
        with pytest.warns(UserWarning, match="stable canonical frame"):
            res = hash_molecule(z, coords, precision=1.0, method="frame")
        canonical = hash_molecule(z, coords, precision=1.0, method="canonical")
        assert res.descriptor == canonical.descriptor

    def test_candidate_budget_fallback_is_permutation_invariant(self, benzene, monkeypatch):
        monkeypatch.setattr(core, "_FRAME_CANDIDATE_BUDGET", 3)
        z, coords = benzene
        with pytest.warns(UserWarning, match="3-candidate budget"):
            res = hash_molecule(z, coords, method="frame")
        assert "|C:" in res.descriptor
        rng = np.random.default_rng(19)
        for t in range(10):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            with pytest.warns(UserWarning):
                other = hash_molecule(zz, cc, method="frame")
            assert other.geometry_hash == res.geometry_hash

    def test_canonical_method_never_warns(self, benzene):
        z, coords = benzene
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hash_molecule(z, coords, method="canonical")
