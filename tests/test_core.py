"""Unit tests for HashMol3D core helpers and the public function."""

import numpy as np
import pytest

from hashmol3d.core import (
    _infer_multiplicity,
    _pair_signature,
    _precision_to_decimals,
    generate_hashmol3d,
)


class TestPrecisionToDecimals:
    def test_standard(self):
        assert _precision_to_decimals(1e-4) == 4
        assert _precision_to_decimals(1e-3) == 3
        assert _precision_to_decimals(1e-2) == 2
        assert _precision_to_decimals(1e-1) == 1
        assert _precision_to_decimals(1.0) == 0

    def test_invalid(self):
        with pytest.raises(ValueError):
            _precision_to_decimals(0.0)
        with pytest.raises(ValueError):
            _precision_to_decimals(-1.0)
        with pytest.raises(ValueError):
            _precision_to_decimals(float("nan"))


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


class TestPairSignature:
    def test_water_pairs(self, water):
        z, coords = water
        z_sorted, pairs = _pair_signature(z, coords, decimals=4)
        assert z_sorted == (1, 1, 8)
        # 3 atoms -> 3 pairs.  Two O-H bonds + one H-H non-bond.
        assert len(pairs) == 3
        elements = sorted({(a, b) for a, b, _ in pairs})
        assert elements == [(1, 1), (1, 8)]

    def test_single_atom_has_no_pairs(self):
        z, _ = _pair_signature(np.array([6]), np.zeros((1, 3)), decimals=4)
        assert z == (6,)


class TestInputValidation:
    def test_empty(self):
        with pytest.raises(ValueError):
            generate_hashmol3d(np.array([], dtype=int), np.zeros((0, 3)))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            generate_hashmol3d(np.array([1, 1]), np.zeros((3, 3)))

    def test_wrong_coord_shape(self):
        with pytest.raises(ValueError):
            generate_hashmol3d(np.array([1, 1]), np.zeros((2, 4)))

    def test_nonpositive_z(self):
        with pytest.raises(ValueError):
            generate_hashmol3d(np.array([0, 1]), np.zeros((2, 3)))

    def test_nonfinite_coords(self):
        with pytest.raises(ValueError):
            generate_hashmol3d(np.array([1, 1]), np.array([[0, 0, 0], [np.nan, 0, 0]]))

    def test_hash_length_bounds(self, water):
        z, coords = water
        with pytest.raises(ValueError):
            generate_hashmol3d(z, coords, hash_length=0)
        with pytest.raises(ValueError):
            generate_hashmol3d(z, coords, hash_length=65)


class TestDeterminism:
    def test_same_input_same_hash(self, water):
        z, coords = water
        a = generate_hashmol3d(z, coords)
        b = generate_hashmol3d(z, coords)
        assert a.hash_str == b.hash_str
        assert a.descriptor == b.descriptor

    def test_different_charge_different_hash(self, water):
        z, coords = water
        a = generate_hashmol3d(z, coords, charge=0)
        b = generate_hashmol3d(z, coords, charge=1)
        assert a.hash_str != b.hash_str

    def test_different_multiplicity_different_hash(self, water):
        z, coords = water
        a = generate_hashmol3d(z, coords, multiplicity=1)
        b = generate_hashmol3d(z, coords, multiplicity=3)
        assert a.hash_str != b.hash_str

    def test_different_precision_different_hash(self, chiral_chfclbr):
        z, coords = chiral_chfclbr
        a = generate_hashmol3d(z, coords, precision=1e-4)
        b = generate_hashmol3d(z, coords, precision=1e-3)
        assert a.hash_str != b.hash_str

    def test_version_changes_hash(self, water):
        z, coords = water
        a = generate_hashmol3d(z, coords, version="alpha")
        b = generate_hashmol3d(z, coords, version="beta")
        assert a.hash_str != b.hash_str


class TestBackwardsCompatAliases:
    def test_protocol_kwarg_overrides_version(self, water):
        z, coords = water
        r1 = generate_hashmol3d(z, coords, version="X", protocol="Y")
        r2 = generate_hashmol3d(z, coords, version="Y")
        assert r1.hash_str == r2.hash_str
        assert r1.version == "Y"

    def test_protocol_property_mirrors_version(self, water):
        z, coords = water
        r = generate_hashmol3d(z, coords, version="abc")
        assert r.protocol == "abc"
