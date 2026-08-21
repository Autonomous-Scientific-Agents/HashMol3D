"""Collision-resistance tests for the canonical (v5) geometry descriptor.

The v4 descriptor hashed the unlabeled multiset of (Zmin, Zmax, distance)
triples, so *homometric* configurations -- distinct geometries sharing a
distance multiset -- collided. The v5 canonical descriptor hashes the
labeled distance matrix in a canonical atom order and must distinguish:

  1. the classic 6-point homometric pair on a line,
  2. a 24-atom 3D homometric pair (grid product construction),
  3. two 5-point pairs that even share all *per-atom* sorted distance
     signatures (1-WL-indistinguishable) yet are non-congruent; both were
     verified non-congruent by brute force over all 120 atom pairings.

It must also stay permutation-invariant on highly symmetric molecules
(where the canonical search branches over symmetry orbits) and on the
degenerate-rounding fallback path.
"""

import numpy as np
import pytest

import hashmol3d.core as core
from hashmol3d import hash_molecule


def _line(points):
    coords = np.zeros((len(points), 3))
    coords[:, 0] = np.asarray(points, dtype=float)
    return np.full(len(points), 6), coords


def _distance_multiset(coords, decimals=4):
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(coords), k=1)
    return sorted(np.round(d[iu, ju], decimals).tolist())


def _atom_rows(coords, decimals=4):
    """Multiset of per-atom sorted distance rows (the 1-WL signature)."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return sorted(tuple(np.sort(np.round(row, decimals)).tolist()) for row in d)


# Classic homometric pair: same 15 pairwise differences, not congruent.
CLASSIC_A = (0, 1, 4, 10, 12, 17)
CLASSIC_B = (0, 1, 8, 11, 13, 17)

# 5-point integer-lattice pairs sharing the full per-atom distance-row
# multiset (1-WL-indistinguishable) while provably non-congruent (verified
# by exhaustive search over all 5! atom pairings; carbon at every site).
WL1_PAIRS = [
    (
        [[0, 2, 1], [1, 3, 3], [1, 1, 1], [3, 1, 1], [3, 3, 3]],
        [[2, 0, 1], [3, 3, 1], [1, 1, 1], [1, 1, 3], [3, 3, 3]],
    ),
    (
        [[1, 1, 1], [0, 3, 2], [0, 2, 1], [2, 2, 0], [1, 0, 0]],
        [[2, 2, 2], [3, 3, 2], [0, 1, 3], [1, 2, 3], [1, 1, 1]],
    ),
]


class TestHomometricPairs:
    def test_classic_line_pair_distinct(self):
        za, ca = _line(CLASSIC_A)
        zb, cb = _line(CLASSIC_B)
        # Premise: they really are homometric...
        assert _distance_multiset(ca) == _distance_multiset(cb)
        # ...and the canonical hash still separates them.
        assert hash_molecule(za, ca).geometry_hash != hash_molecule(zb, cb).geometry_hash

    def test_3d_grid_pair_distinct(self):
        def grid(base):
            pts = np.array([[x, y, w] for x in base for y in (0.0, 1.1) for w in (0.0, 1.3)])
            return np.full(len(pts), 6), pts

        za, ca = grid(CLASSIC_A)
        zb, cb = grid(CLASSIC_B)
        assert _distance_multiset(ca) == _distance_multiset(cb)
        assert hash_molecule(za, ca).geometry_hash != hash_molecule(zb, cb).geometry_hash

    @pytest.mark.parametrize("pair_idx", [0, 1])
    def test_wl1_indistinguishable_pairs_distinct(self, pair_idx):
        a, b = WL1_PAIRS[pair_idx]
        ca, cb = np.asarray(a, float), np.asarray(b, float)
        z = np.full(5, 6)
        # Premise: homometric AND identical per-atom distance rows.
        assert _distance_multiset(ca) == _distance_multiset(cb)
        assert _atom_rows(ca) == _atom_rows(cb)
        # The canonical descriptor still separates them.
        assert hash_molecule(z, ca).geometry_hash != hash_molecule(z, cb).geometry_hash


def _scramble(z, coords, rng, reflect):
    perm = rng.permutation(len(z))
    h = rng.standard_normal((3, 3))
    rot, _ = np.linalg.qr(h)
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    c = coords @ rot.T + rng.uniform(-5, 5, size=3)
    if reflect:
        c = c.copy()
        c[:, 0] *= -1
    return z[perm], c[perm]


class TestSymmetricInvariance:
    """The canonical search branches over symmetry orbits; every branch
    order must produce the identical hash."""

    def _check(self, z, coords, trials=30, seed=0):
        rng = np.random.default_rng(seed)
        base = hash_molecule(z, coords).geometry_hash
        for t in range(trials):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc).geometry_hash == base

    def test_benzene(self, benzene):
        self._check(*benzene)

    def test_ring12(self):
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        r = 1.4 / (2 * np.sin(np.pi / 12))
        coords = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(12)])
        self._check(np.full(12, 6), coords)

    def test_methane(self):
        a = 1.09 / np.sqrt(3)
        coords = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], float)
        self._check(np.array([6, 1, 1, 1, 1]), coords)

    def test_tetrahedron_identical_atoms(self):
        # All six pair distances equal: the search individualizes three
        # levels deep (41 nodes) and every leaf ties.
        coords = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        self._check(np.full(4, 6), coords)

    def test_wl1_counterexample_geometry(self):
        z = np.full(5, 6)
        self._check(z, np.asarray(WL1_PAIRS[0][0], float))


class TestFallbackPath:
    def test_degenerate_rounding_falls_back_and_stays_invariant(self):
        # 12 atoms inside a 0.01 A box hashed at 1 A precision: every
        # rounded distance is 0, the search tree exceeds the node budget,
        # and the stable-WL fallback (tag "W") must kick in deterministically.
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 0.01, size=(12, 3))
        z = np.full(12, 6)
        res = hash_molecule(z, coords, precision=1.0)
        assert "|W:" in res.descriptor
        assert "|C:" not in res.descriptor
        base = res.geometry_hash
        for t in range(10):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc, precision=1.0).geometry_hash == base

    def test_budget_is_permutation_invariant_when_patched(self, benzene, monkeypatch):
        # Force even benzene onto the fallback path; the trigger and the
        # resulting hash must not depend on the input atom order.
        monkeypatch.setattr(core, "_NODE_BUDGET", 3)
        z, coords = benzene
        res = hash_molecule(z, coords)
        assert "|W:" in res.descriptor
        rng = np.random.default_rng(9)
        for t in range(10):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            assert hash_molecule(zz, cc).geometry_hash == res.geometry_hash

    def test_normal_molecules_use_canonical_path(self, water, benzene):
        for z, coords in (water, benzene):
            assert "|C:" in hash_molecule(z, coords).descriptor
