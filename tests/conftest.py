"""Shared pytest fixtures for HashMol3D tests.

Kept dependency-free: HashMol3D itself only needs NumPy.
"""

import numpy as np
import pytest


@pytest.fixture
def water():
    """H2O geometry (atomic_nums, coords)."""
    z = np.array([8, 1, 1], dtype=int)
    coords = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.7572, 0.5860, 0.0000],
        [-0.7572, 0.5860, 0.0000],
    ])
    return z, coords


@pytest.fixture
def benzene():
    """Planar benzene (C6H6); 6 C in a hexagon then 6 H radially outside."""
    r_c = 1.40
    r_ch = 1.09
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    c_xy = np.column_stack([np.cos(angles) * r_c, np.sin(angles) * r_c])
    h_xy = np.column_stack([
        np.cos(angles) * (r_c + r_ch),
        np.sin(angles) * (r_c + r_ch),
    ])
    coords = np.zeros((12, 3))
    coords[:6, :2] = c_xy
    coords[6:, :2] = h_xy
    z = np.array([6] * 6 + [1] * 6, dtype=int)
    return z, coords


@pytest.fixture
def chiral_chfclbr():
    """A specific enantiomer of CHFClBr (central C + H, F, Cl, Br ligands)."""
    z = np.array([6, 1, 9, 17, 35], dtype=int)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.03, 0.0, -0.36],
        [-0.5, 0.89, -0.36],
        [-0.5, -0.89, -0.36],
    ])
    return z, coords
