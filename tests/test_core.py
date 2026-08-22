"""Unit tests for HashMol3D core helpers and the public function."""

import warnings

import numpy as np
import pytest

from hashmol3d import generate_hashmol3d, hash_length_for, hash_molecule
from hashmol3d.core import (
    _canonical_signature,
    _hill_formula,
    _infer_multiplicity,
    _precision_to_decimals,
    _scaled_distances,
    _state_tag,
)


class TestPrecisionToDecimals:
    def test_standard(self):
        # Returns (decimals, effective_precision) -- the canonical grid.
        assert _precision_to_decimals(1e-4) == (4, 1e-4)
        assert _precision_to_decimals(1e-3) == (3, 1e-3)
        assert _precision_to_decimals(1e-2) == (2, 1e-2)
        assert _precision_to_decimals(1e-1) == (1, 1e-1)
        assert _precision_to_decimals(1.0) == (0, 1.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            _precision_to_decimals(0.0)
        with pytest.raises(ValueError):
            _precision_to_decimals(-1.0)
        with pytest.raises(ValueError):
            _precision_to_decimals(float("nan"))

    @pytest.mark.parametrize("precision", [10.0, 0.05, 0.02, 3.16e-4, 3.17e-4])
    def test_rejects_ambiguous_grid_precision(self, precision):
        with pytest.raises(ValueError, match="power of ten"):
            _precision_to_decimals(precision)

    def test_rejects_bool(self):
        # bool is an int subclass; without a guard True would read as 1.0 Å.
        with pytest.raises(ValueError, match="bool"):
            _precision_to_decimals(True)
        with pytest.raises(ValueError):
            hash_molecule([1, 1], [[0, 0, 0], [0, 0, 0.1]], precision=True)

    def test_rejects_precision_too_fine(self):
        # A clean ValueError, not a raw OverflowError from 10**decimals.
        with pytest.raises(ValueError, match="too fine"):
            _precision_to_decimals(1e-301)
        with pytest.raises(ValueError):
            hash_molecule([1, 1], [[0, 0, 0], [0, 0, 0.1]], precision=1e-309)

    def test_dtype_independent(self):
        # numpy floats validate the same as plain floats (no value-based
        # casting shortcut). float32 round-off (~6e-8) is within tolerance,
        # so a valid grid is accepted and canonicalized regardless of dtype.
        assert _precision_to_decimals(np.float64(1e-3)) == (3, 1e-3)
        decimals, effective = _precision_to_decimals(np.float32(1e-4))
        assert decimals == 4
        assert effective == 1e-4

    def test_normalizes_floating_point_roundoff(self):
        precision = np.nextafter(1e-4, np.inf)
        result = hash_molecule([1, 1], [[0, 0, 0], [0, 0, 0.1]], precision=precision)
        assert result.precision == 1e-4
        assert "|P:1.0e-04|" in result.descriptor


class TestInferMultiplicity:
    def test_user_value_wins(self):
        z = np.array([6, 6, 8])
        assert _infer_multiplicity(z, 0, 3) == 3

    def test_singlet_for_even_electrons(self):
        z = np.array([6, 6, 8])  # 20 electrons
        assert _infer_multiplicity(z, 0, None) == 1

    def test_doublet_for_odd_electrons(self):
        z = np.array([1])  # 1 electron
        assert _infer_multiplicity(z, 0, None) == 2

    def test_charge_affects_inference(self):
        z = np.array([1])
        assert _infer_multiplicity(z, 1, None) == 1  # cation, 0 electrons

    def test_invalid_multiplicity(self):
        with pytest.raises(ValueError):
            _infer_multiplicity(np.array([1]), 0, 0)


class TestHillFormula:
    def test_water(self):
        assert _hill_formula(np.array([8, 1, 1])) == "H2O"

    def test_benzene(self):
        assert _hill_formula(np.array([6] * 6 + [1] * 6)) == "C6H6"

    def test_chfclbr(self):
        # Hill: C first, then H, then alphabetical (Br, Cl, F).
        assert _hill_formula(np.array([6, 1, 9, 17, 35])) == "CHBrClF"

    def test_no_carbon(self):
        # Without carbon, all elements alphabetical (H included alphabetically).
        assert _hill_formula(np.array([8, 16])) == "OS"
        assert _hill_formula(np.array([1, 8])) == "HO"

    def test_single_atom(self):
        assert _hill_formula(np.array([6])) == "C"
        assert _hill_formula(np.array([1])) == "H"


class TestStateTag:
    def test_neutral_singlet(self):
        assert _state_tag(0, 1) == "q0m1"

    def test_cation(self):
        assert _state_tag(1, 2) == "q1m2"

    def test_anion(self):
        assert _state_tag(-2, 1) == "q-2m1"


class TestHashLengthFor:
    def test_monotonic_in_corpus_and_stringency(self):
        assert hash_length_for(10) <= hash_length_for(10**9)
        assert hash_length_for(10**9, 1e-6) <= hash_length_for(10**9, 1e-12)

    def test_within_range(self):
        for n in (1, 10, 10**6, 10**12, 10**30):
            assert 1 <= hash_length_for(n) <= 64

    def test_satisfies_birthday_bound(self):
        for n, p in [(10**6, 1e-9), (10**9, 1e-9), (10**6, 1e-12)]:
            L = hash_length_for(n, p)
            assert n**2 / 2 ** (4 * L + 1) <= p

    def test_edge_cases(self):
        assert hash_length_for(1) == 1
        with pytest.raises(ValueError):
            hash_length_for(10, target_prob=1.0)


class TestScaledDistances:
    def test_water_grid_values(self, water):
        _, coords = water
        q = _scaled_distances(coords, decimals=4)
        assert q.dtype == np.int64
        assert np.array_equal(q, q.T)
        assert np.all(np.diag(q) == 0)
        # O-H distance 0.9575 A -> 9575 grid units at 1e-4 precision.
        assert q[0, 1] == 9575
        assert q[0, 2] == 9575
        assert q[1, 2] == 15144

    def test_overflow_guard(self):
        coords = np.array([[0.0, 0.0, 0.0], [1e6, 0.0, 0.0]])
        with pytest.raises(ValueError):
            _scaled_distances(coords, decimals=15)


class TestCanonicalSignature:
    def test_water(self, water):
        z, coords = water
        q = _scaled_distances(coords, decimals=4)
        tag, z_ordered, body = _canonical_signature(z, q)
        assert tag == "C"
        # Canonical order groups atoms by ascending Z.
        assert z_ordered == (1, 1, 8)
        # Three pair distances on the 1e-4 grid.
        assert body == "15144,9575,9575"

    def test_single_atom(self):
        q = _scaled_distances(np.zeros((1, 3)), decimals=4)
        tag, z_ordered, body = _canonical_signature(np.array([6]), q)
        assert tag == "C"
        assert z_ordered == (6,)
        assert body == ""

    def test_permutation_invariant_including_symmetric(self, benzene):
        z, coords = benzene
        q = _scaled_distances(coords, decimals=4)
        base = _canonical_signature(z, q)
        rng = np.random.default_rng(1)
        for _ in range(20):
            perm = rng.permutation(len(z))
            qp = _scaled_distances(coords[perm], decimals=4)
            assert _canonical_signature(z[perm], qp) == base


class TestInputValidation:
    def test_empty(self):
        with pytest.raises(ValueError):
            hash_molecule(np.array([], dtype=int), np.zeros((0, 3)))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            hash_molecule(np.array([1, 1]), np.zeros((3, 3)))

    def test_wrong_coord_shape(self):
        with pytest.raises(ValueError):
            hash_molecule(np.array([1, 1]), np.zeros((2, 4)))

    def test_nonpositive_z(self):
        with pytest.raises(ValueError):
            hash_molecule(np.array([0, 1]), np.zeros((2, 3)))

    def test_nonfinite_coords(self):
        with pytest.raises(ValueError):
            hash_molecule(np.array([1, 1]), np.array([[0, 0, 0], [np.nan, 0, 0]]))

    def test_length_bounds(self, water):
        z, coords = water
        with pytest.raises(ValueError):
            hash_molecule(z, coords, length=0)
        with pytest.raises(ValueError):
            hash_molecule(z, coords, length=65)

    def test_length_accepts_numpy_integer(self, water):
        z, coords = water
        # NumPy integers are not `int` subclasses; they must still be accepted.
        r = hash_molecule(z, coords, length=np.int64(16))
        assert len(r.geometry_hash) == 16

    def test_length_rejects_bool(self, water):
        z, coords = water
        # bool is an int subclass but is not a meaningful length.
        with pytest.raises(ValueError):
            hash_molecule(z, coords, length=True)


class TestDeterminism:
    def test_same_input_same_hash(self, water):
        z, coords = water
        a = hash_molecule(z, coords)
        b = hash_molecule(z, coords)
        assert a.hash_str == b.hash_str
        assert a.descriptor == b.descriptor

    def test_charge_only_changes_prefix(self, water):
        z, coords = water
        a = hash_molecule(z, coords, charge=0)
        b = hash_molecule(z, coords, charge=1)
        # Full identifier differs in the readable prefix...
        assert a.hash_str != b.hash_str
        assert a.hash_str.startswith("H2Oq0")
        assert b.hash_str.startswith("H2Oq1")
        # ...but the geometry hash is identical.
        assert a.geometry_hash == b.geometry_hash

    def test_multiplicity_only_changes_prefix(self, water):
        z, coords = water
        a = hash_molecule(z, coords, multiplicity=1)
        b = hash_molecule(z, coords, multiplicity=3)
        assert a.hash_str != b.hash_str
        assert a.geometry_hash == b.geometry_hash

    def test_different_precision_different_hash(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        a = hash_molecule(z, coords, precision=1e-4)
        b = hash_molecule(z, coords, precision=1e-3)
        assert a.geometry_hash != b.geometry_hash


class TestDeprecatedAlias:
    def test_generate_hashmol3d_warns(self, water):
        z, coords = water
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old = generate_hashmol3d(z, coords)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        new = hash_molecule(z, coords, length=32)
        assert old.hash_str == new.hash_str

    def test_generate_hashmol3d_hash_length_kwarg(self, water):
        z, coords = water
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r = generate_hashmol3d(z, coords, hash_length=16)
        assert len(r.geometry_hash) == 16


class TestKeywordOnly:
    def test_optional_args_are_keyword_only(self, water):
        z, coords = water
        with pytest.raises(TypeError):
            # precision is keyword-only; positional must fail.
            hash_molecule(z, coords, 1e-3)
