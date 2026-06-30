"""Smoke tests for the public API."""

import numpy as np

from hashmol3d import HashMol3DResult, hash_molecule


def test_returns_result(water):
    z, coords = water
    res = hash_molecule(z, coords)
    assert isinstance(res, HashMol3DResult)
    assert res.formula == "H2O"
    assert res.charge == 0
    assert res.multiplicity == 1
    assert res.hash_str.startswith("H2Oq0m1-")
    assert res.hash_str.endswith(res.geometry_hash)
    assert str(res) == res.hash_str


def test_descriptor_contains_geometry_fields(water):
    z, coords = water
    res = hash_molecule(z, coords)
    for tag in ("V:", "P:", "Z:", "D:"):
        assert tag in res.descriptor
    # Charge/multiplicity no longer hashed.
    assert "Q:" not in res.descriptor
    assert "M:" not in res.descriptor


def test_length_override(water):
    z, coords = water
    assert len(hash_molecule(z, coords, length=16).geometry_hash) == 16
    assert len(hash_molecule(z, coords, length=32).geometry_hash) == 32
    assert len(hash_molecule(z, coords, length=64).geometry_hash) == 64


def test_auto_length_scales_with_atom_count():
    # Small molecule (N < 16) clamps to 16 hex chars.
    z3 = np.array([8, 1, 1], dtype=int)
    coords3 = np.zeros((3, 3))
    coords3[1, 0] = 1.0
    coords3[2, 1] = 1.0
    assert len(hash_molecule(z3, coords3).geometry_hash) == 16

    # Mid-size molecule: length == N.
    n = 25
    z = np.full(n, 6, dtype=int)
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * 1.5
    assert len(hash_molecule(z, coords).geometry_hash) == 25

    # Big molecule clamps to 64.
    n = 200
    z = np.full(n, 6, dtype=int)
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * 1.5
    assert len(hash_molecule(z, coords).geometry_hash) == 64


def test_single_atom_is_valid():
    z = np.array([6], dtype=int)
    coords = np.array([[0.0, 0.0, 0.0]])
    res = hash_molecule(z, coords)
    assert res.formula == "C"
    assert res.hash_str.startswith("Cq0m1-")
