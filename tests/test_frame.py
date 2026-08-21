"""Tests for the O(N) principal-axes ("frame") hashing method.

The frame method hashes coordinates in the eigenbasis of the Z-weighted
gyration tensor. It is reliable only when the eigenvalues are well
separated; degenerate or nearly degenerate cases (symmetric tops, linear
molecules) must emit a UserWarning and fall back to the canonical method.
"""

import warnings

import numpy as np
import pytest

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


class TestFrameFallback:
    def _assert_warns_and_matches_canonical(self, z, coords):
        with pytest.warns(UserWarning, match="unreliable"):
            res = hash_molecule(z, coords, method="frame")
        assert "|F:" not in res.descriptor
        assert res.geometry_hash == hash_molecule(z, coords).geometry_hash

    def test_benzene_symmetric_top(self, benzene):
        self._assert_warns_and_matches_canonical(*benzene)

    def test_linear_co2(self):
        z = np.array([8, 6, 8])
        coords = np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]], float)
        self._assert_warns_and_matches_canonical(z, coords)

    def test_methane_spherical_top(self):
        a = 1.09 / np.sqrt(3)
        coords = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], float)
        self._assert_warns_and_matches_canonical(np.array([6, 1, 1, 1, 1]), coords)

    def test_near_degenerate_rejected(self, benzene):
        # A 2e-4 A symmetry break is far below the gap threshold; the frame
        # must still be refused (this is exactly the numerically explosive
        # regime the diagnostic exists for).
        z, coords = benzene
        coords = coords.copy()
        coords[0, :2] *= 1 + 2e-4 / 1.4
        self._assert_warns_and_matches_canonical(z, coords)

    def test_single_atom_falls_back(self):
        with pytest.warns(UserWarning):
            res = hash_molecule(np.array([6]), np.zeros((1, 3)), method="frame")
        assert res.geometry_hash == hash_molecule(np.array([6]), np.zeros((1, 3))).geometry_hash

    def test_canonical_method_never_warns(self, benzene):
        z, coords = benzene
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hash_molecule(z, coords, method="canonical")
