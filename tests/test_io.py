"""Tests for the XYZ reader."""

import warnings

import numpy as np
import pytest

from hashmol3d import hash_molecule, hash_xyz, read_xyz


def _write_xyz(tmp_path, text):
    p = tmp_path / "mol.xyz"
    p.write_text(text)
    return str(p)


def test_read_xyz_symbols(tmp_path):
    path = _write_xyz(
        tmp_path,
        "3\nwater\nO  0.0  0.0  0.0\nH  0.7572  0.5860  0.0\nH -0.7572 0.5860 0.0\n",
    )
    z, coords = read_xyz(path)
    assert z.tolist() == [8, 1, 1]
    assert coords.shape == (3, 3)
    assert coords[1, 0] == pytest.approx(0.7572)


def test_read_xyz_atomic_numbers(tmp_path):
    path = _write_xyz(tmp_path, "2\n\n6 0 0 0\n8 0 0 1.2\n")
    z, _ = read_xyz(path)
    assert z.tolist() == [6, 8]


def test_read_xyz_warns_on_extra_lines(tmp_path):
    # Reading stops at the declared count (first frame of a trajectory), but
    # trailing content is reported: an under-declared count would otherwise
    # silently hash a truncated molecule.
    path = _write_xyz(
        tmp_path,
        "1\ncomment\nC 0 0 0\nthis line should be ignored\n",
    )
    with pytest.warns(UserWarning, match="1 non-blank line"):
        z, coords = read_xyz(path)
    assert z.tolist() == [6]
    assert coords.shape == (1, 3)


def test_read_xyz_multiframe_returns_first_frame_with_warning(tmp_path):
    frame = "2\ncomment\nC 0 0 0\nC 1.2 0 0\n"
    path = _write_xyz(tmp_path, frame + frame)
    with pytest.warns(UserWarning, match="only the first frame"):
        z, coords = read_xyz(path)
    assert z.tolist() == [6, 6]
    assert coords.shape == (2, 3)


def test_read_xyz_exact_count_no_warning(tmp_path):
    path = _write_xyz(tmp_path, "1\ncomment\nC 0 0 0\n\n\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z, _ = read_xyz(path)
    assert z.tolist() == [6]


def test_read_xyz_rejects_bad_count(tmp_path):
    path = _write_xyz(tmp_path, "not_a_number\n\nC 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(path)


def test_read_xyz_rejects_zero_count(tmp_path):
    path = _write_xyz(tmp_path, "0\n\n")
    with pytest.raises(ValueError):
        read_xyz(path)


def test_read_xyz_rejects_missing_atoms(tmp_path):
    path = _write_xyz(tmp_path, "3\n\nC 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(path)


def test_read_xyz_rejects_unknown_symbol(tmp_path):
    path = _write_xyz(tmp_path, "1\n\nXx 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(path)


def test_read_xyz_rejects_bad_coords(tmp_path):
    path = _write_xyz(tmp_path, "1\n\nC oops 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(path)


def test_hash_xyz_matches_hash_molecule(tmp_path):
    path = _write_xyz(
        tmp_path,
        "3\n\nO 0 0 0\nH 0.7572 0.5860 0\nH -0.7572 0.5860 0\n",
    )
    z, coords = read_xyz(path)
    direct = hash_molecule(z, coords)
    via_file = hash_xyz(path)
    assert direct.hash_str == via_file.hash_str


def test_hash_xyz_forwards_kwargs(tmp_path):
    path = _write_xyz(tmp_path, "1\n\nH 0 0 0\n")
    r0 = hash_xyz(path)
    r1 = hash_xyz(path, charge=1)
    assert r0.hash_str != r1.hash_str
    assert np.all(read_xyz(path)[0] == np.array([1]))
