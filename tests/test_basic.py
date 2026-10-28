"""Smoke tests for the public API."""

import hashlib

import numpy as np
import pytest

from hashmol3d import DEFAULT_LENGTH, HashMol3DResult, hash_length_for, hash_molecule


def test_returns_result(water):
    z, coords = water
    res = hash_molecule(z, coords)
    assert isinstance(res, HashMol3DResult)
    assert res.formula == "H2O"
    assert res.charge == 0
    assert res.multiplicity == 1
    assert res.hash_str == f"h3md-v11.0-p4-H2O-{res.geometry_hash}-q0m1"
    assert str(res) == res.hash_str


def test_documented_water_identifier(water):
    """Pin the identifier quoted throughout the docs for descriptor version 11.

    README.md, docs/specification.md, docs/api_reference.md and
    docs/cli_usage.md all quote this value; a descriptor change that is not
    accompanied by a version bump and a docs sweep should fail here.
    """
    z, coords = water
    res = hash_molecule(z, coords)
    assert res.version == "11.0"
    assert res.descriptor == (
        "V:11.0|P:1.0e-04|Z:1,1,8|F:1:0,-46880000,-75720000;1:0,-46880000,75720000;8:0,11720000,0"
    )
    # The hash covers only the Z field and the hashed F rows.
    payload = "Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0"
    assert res.geometry_hash == hashlib.sha256(payload.encode()).hexdigest()[:32]
    assert res.hash_str == "h3md-v11.0-p4-H2O-a3528714040eea8068be04d9807fadfb-q0m1"


def test_descriptor_contains_geometry_fields(water):
    z, coords = water
    res = hash_molecule(z, coords)
    for tag in ("V:", "P:", "Z:", "F:"):
        assert tag in res.descriptor
    # Charge/multiplicity no longer hashed.
    assert "Q:" not in res.descriptor
    assert "M:" not in res.descriptor


def test_length_override(water):
    z, coords = water
    assert len(hash_molecule(z, coords, length=16).geometry_hash) == 16
    assert len(hash_molecule(z, coords, length=32).geometry_hash) == 32
    assert len(hash_molecule(z, coords, length=64).geometry_hash) == 64


def test_default_length_is_fixed_regardless_of_size():
    # The default length is a fixed 32 hex chars (128 bits): collision
    # resistance is governed by corpus size, not molecule size.
    assert DEFAULT_LENGTH == 32

    z3 = np.array([8, 1, 1], dtype=int)
    coords3 = np.zeros((3, 3))
    coords3[1, 0] = 1.0
    coords3[2, 1] = 1.0
    assert len(hash_molecule(z3, coords3).geometry_hash) == DEFAULT_LENGTH

    for n in (25, 200):
        z = np.full(n, 6, dtype=int)
        coords = np.zeros((n, 3))
        coords[:, 0] = np.arange(n) * 1.5
        assert len(hash_molecule(z, coords).geometry_hash) == DEFAULT_LENGTH


def test_hash_length_for():
    # Monotonic non-decreasing in corpus size and in stringency.
    assert hash_length_for(10) <= hash_length_for(10**9)
    assert hash_length_for(10**9, 1e-6) <= hash_length_for(10**9, 1e-12)
    # Always within the valid API range.
    for n in (1, 10, 10**6, 10**12, 10**30):
        L = hash_length_for(n)
        assert 1 <= L <= 64
    # A hashed length actually satisfies the requested bound: with L hex chars
    # (b = 4L bits), n^2 / 2^(b+1) <= target_prob.
    for n, p in [(10**6, 1e-9), (10**9, 1e-9), (10**6, 1e-12)]:
        L = hash_length_for(n, p)
        assert n**2 / 2 ** (4 * L + 1) <= p
    # Never-collide edge case.
    assert hash_length_for(1) == 1
    with pytest.raises(ValueError):
        hash_length_for(10, target_prob=0.0)


def test_single_atom_is_valid():
    z = np.array([6], dtype=int)
    coords = np.array([[0.0, 0.0, 0.0]])
    res = hash_molecule(z, coords)
    assert res.formula == "C"
    assert res.hash_str.startswith("h3md-v11.0-p4-C-")


def test_identifier_layout(water):
    """h3md-v<version>-p<decimals>-<formula>-<geometry hash>-<state tag>."""
    z, coords = water
    for precision, decimals in ((1.0, 0), (1e-2, 2), (1e-4, 4)):
        res = hash_molecule(z, coords, precision=precision, charge=-1, multiplicity=2)
        prefix, version, grid, formula, digest, state = res.hash_str.split("-", 5)
        assert (prefix, version, grid, formula) == ("h3md", "v11.0", f"p{decimals}", "H2O")
        assert digest == res.geometry_hash
        assert state == "q-1m2"


def test_precision_is_in_the_identifier_not_the_hash(water):
    # The same integers on different grids give the same geometry hash; the
    # identifier still tells them apart through its p<decimals> field.
    z, coords = water
    fine = hash_molecule(z, coords, precision=1e-4)
    coarse = hash_molecule(z, 10 * coords, precision=1e-3)
    assert fine.geometry_hash == coarse.geometry_hash
    assert fine.hash_str != coarse.hash_str


def test_canonical_method_is_a_deprecated_alias(water):
    z, coords = water
    with pytest.warns(DeprecationWarning, match="method='distance'"):
        old = hash_molecule(z, coords, method="canonical")
    assert old == hash_molecule(z, coords, method="distance")
    assert "|D:" in old.descriptor
