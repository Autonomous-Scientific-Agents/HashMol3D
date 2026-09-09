"""Collision-resistance tests for the geometry descriptors.

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
budget-exhaustion outcome.
"""

import numpy as np
import pytest

import hashmol3d.core as core
from hashmol3d import SearchBudgetExceeded, hash_molecule


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


class TestCanonicalSearchBudget:
    def test_degenerate_rounding_raises_without_hashing(self, monkeypatch):
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 0.01, size=(12, 3))
        z = np.full(12, 6)

        def unexpected_hash(*args, **kwargs):
            pytest.fail("budget exhaustion must not create a hash")

        monkeypatch.setattr(core.hashlib, "sha256", unexpected_hash)
        with pytest.raises(SearchBudgetExceeded, match="Increase node_budget"):
            hash_molecule(z, coords, precision=1.0, method="canonical")

    def test_budget_failure_is_invariant(self, benzene):
        z, coords = benzene
        rng = np.random.default_rng(9)
        for t in range(10):
            zz, cc = _scramble(z, coords, rng, reflect=(t % 2 == 1))
            with pytest.raises(SearchBudgetExceeded, match=r"node budget \(3\)"):
                hash_molecule(zz, cc, method="canonical", node_budget=3)

    def test_partial_search_is_discarded_and_retry_completes(self, water):
        z, coords = water
        # Root + one leaf fits, but the other tied ordering is still required.
        with pytest.raises(SearchBudgetExceeded):
            hash_molecule(z, coords, method="canonical", node_budget=2)
        res = hash_molecule(z, coords, method="canonical", node_budget=3)
        assert res == hash_molecule(z, coords, method="canonical")
        assert res == hash_molecule(z, coords, method="canonical", node_budget=100_000)
        assert "|C:" in res.descriptor

    def test_frame_fallback_obeys_node_budget(self):
        with pytest.warns(UserWarning, match="falling back"):
            with pytest.raises(SearchBudgetExceeded):
                hash_molecule(
                    [6, 6, 6], [[-1, 0, 0], [0, 0.01, 0], [1, 0, 0]], precision=1.0, node_budget=1
                )
        with pytest.warns(UserWarning, match="falling back"):
            res = hash_molecule(
                [6, 6, 6], [[-1, 0, 0], [0, 0.01, 0], [1, 0, 0]], precision=1.0, node_budget=3
            )
        assert "|C:" in res.descriptor

    @pytest.mark.parametrize("budget", [0, -1, True, np.bool_(True), 1.5, 3.0, "3", None])
    def test_invalid_budget(self, water, budget):
        with pytest.raises(ValueError, match="node_budget must be a positive integer"):
            hash_molecule(*water, node_budget=budget)

    def test_numpy_integer_budget(self, water):
        assert hash_molecule(*water, method="canonical", node_budget=np.int64(3))

    def test_explicit_canonical_method_uses_canonical_path(self, water, benzene):
        for z, coords in (water, benzene):
            assert "|C:" in hash_molecule(z, coords, method="canonical").descriptor


class TestDefaultFramePath:
    def test_point_like_at_grid_uses_intrinsic_frame(self):
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 0.01, size=(12, 3))
        z = np.full(12, 6)
        res = hash_molecule(z, coords, precision=1.0)
        assert "|F:" in res.descriptor
        assert ":0,0,0" in res.descriptor

    def test_normal_molecules_use_frame_path(self, water, benzene):
        for z, coords in (water, benzene):
            assert "|F:" in hash_molecule(z, coords).descriptor
