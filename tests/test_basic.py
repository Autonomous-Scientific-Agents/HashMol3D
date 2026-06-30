"""Smoke tests for the public API."""

import numpy as np

from hashmol3d import HashMol3DResult, hash_molecule


def test_returns_result(water):
    z, coords = water
    res = hash_molecule(z, coords)
    assert isinstance(res, HashMol3DResult)
    assert len(res.hash_str) == 32
    assert str(res) == res.hash_str


def test_descriptor_contains_expected_fields(water):
    z, coords = water
    res = hash_molecule(z, coords)
    for tag in ("V:", "P:", "Z:", "D:", "Q:", "M:"):
        assert tag in res.descriptor


def test_hash_length_options(water):
    z, coords = water
    assert len(hash_molecule(z, coords, length=16).hash_str) == 16
    assert len(hash_molecule(z, coords, length=32).hash_str) == 32
    assert len(hash_molecule(z, coords, length=64).hash_str) == 64


def test_single_atom_is_valid():
    z = np.array([6], dtype=int)
    coords = np.array([[0.0, 0.0, 0.0]])
    res = hash_molecule(z, coords)
    assert len(res.hash_str) > 0
