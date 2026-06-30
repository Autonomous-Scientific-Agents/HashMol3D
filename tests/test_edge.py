"""
Corner-case invariance tests for HashMol3D.

The hash is required to be invariant under exactly the operations that
preserve the non-relativistic molecular Hamiltonian's eigenvalues:

  * permutation (relabeling) of atoms
  * rigid translation
  * rigid rotation
  * spatial inversion / reflection (parity)

It MUST change under:
  * changes in atomic number
  * geometry distortions larger than the chosen precision
  * changes in charge or multiplicity
"""

import unittest

import numpy as np

from hashmol3d import hash_molecule


def _random_rotation(rng):
    """Generate a deterministic random 3D proper rotation."""
    H = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class TestHashMol3D(unittest.TestCase):
    def setUp(self):
        # 1) Planar benzene: 6 C in a hexagon, 6 H radially outside.
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        r_c, r_ch = 1.40, 1.09
        c_xy = np.column_stack([np.cos(angles) * r_c, np.sin(angles) * r_c])
        h_xy = np.column_stack([np.cos(angles) * (r_c + r_ch), np.sin(angles) * (r_c + r_ch)])
        coords = np.zeros((12, 3))
        coords[:6, :2] = c_xy
        coords[6:, :2] = h_xy
        self.benzene_z = np.array([6] * 6 + [1] * 6, dtype=int)
        self.benzene_coords = coords

        # 2) A specific chiral tetrahedral carbon: CHFClBr
        self.chiral_z = np.array([6, 1, 9, 17, 35], dtype=int)
        self.chiral_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.09],
                [1.03, 0.0, -0.36],
                [-0.5, 0.89, -0.36],
                [-0.5, -0.89, -0.36],
            ]
        )

        self.rng = np.random.default_rng(0)

    # ----------------------------------------------------------- permutation
    def test_permutation_invariance(self):
        """Relabeling atoms (even of a highly symmetric molecule) must not
        change the hash."""
        base = hash_molecule(self.benzene_z, self.benzene_coords)
        for _ in range(50):
            perm = self.rng.permutation(len(self.benzene_z))
            res = hash_molecule(self.benzene_z[perm], self.benzene_coords[perm])
            self.assertEqual(base.hash_str, res.hash_str)

        # Also for an asymmetric molecule.
        base = hash_molecule(self.chiral_z, self.chiral_coords)
        for _ in range(50):
            perm = self.rng.permutation(len(self.chiral_z))
            res = hash_molecule(self.chiral_z[perm], self.chiral_coords[perm])
            self.assertEqual(base.hash_str, res.hash_str)

    # -------------------------------------------------- rotation/translation
    def test_rigid_body_invariance(self):
        base = hash_molecule(self.chiral_z, self.chiral_coords)
        for _ in range(10):
            R = _random_rotation(self.rng)
            t = self.rng.uniform(-10.0, 10.0, size=3)
            transformed = self.chiral_coords @ R.T + t
            res = hash_molecule(self.chiral_z, transformed)
            self.assertEqual(base.hash_str, res.hash_str)

    # -------------------------------------------------------- parity / mirror
    def test_mirror_invariance_planar(self):
        """A planar molecule reflected through the molecular plane is
        identical; the hash must not change."""
        base = hash_molecule(self.benzene_z, self.benzene_coords)
        mirrored = self.benzene_coords.copy()
        mirrored[:, 0] *= -1
        res = hash_molecule(self.benzene_z, mirrored)
        self.assertEqual(base.hash_str, res.hash_str)

    def test_mirror_invariance_chiral(self):
        """Enantiomers share the eigenvalues of the non-relativistic
        Hamiltonian, so they must share the HashMol3D identifier."""
        base = hash_molecule(self.chiral_z, self.chiral_coords)
        mirrored = self.chiral_coords.copy()
        mirrored[:, 0] *= -1
        res = hash_molecule(self.chiral_z, mirrored)
        self.assertEqual(base.hash_str, res.hash_str)

    # -------------------------------------------------------- noise / geometry
    def test_precision_noise(self):
        precision = 1e-4
        base = hash_molecule(self.chiral_z, self.chiral_coords, precision=precision)

        # (A) tiny sub-precision noise -> same hash
        tiny = self.chiral_coords + self.rng.uniform(-1e-9, 1e-9, size=self.chiral_coords.shape)
        self.assertEqual(
            base.hash_str,
            hash_molecule(self.chiral_z, tiny, precision=precision).hash_str,
        )

        # (B) genuine geometric distortion -> different hash
        distorted = self.chiral_coords.copy()
        distorted[0] += 0.5
        self.assertNotEqual(
            base.hash_str,
            hash_molecule(self.chiral_z, distorted, precision=precision).hash_str,
        )

    def test_atomic_number_matters(self):
        z2 = self.chiral_z.copy()
        z2[1] = 7  # H -> N
        a = hash_molecule(self.chiral_z, self.chiral_coords)
        b = hash_molecule(z2, self.chiral_coords)
        self.assertNotEqual(a.hash_str, b.hash_str)

    # -------------------------------------------------------- achirality
    def test_3d_achiral_mirror_consistency(self):
        """A 3D-but-achiral structure (two orthogonal mirror planes) hashes
        the same as its mirror image."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        z = np.array([6, 6, 9, 9, 9, 9])

        base = hash_molecule(z, coords)
        mirror = coords.copy()
        mirror[:, 0] *= -1
        self.assertEqual(base.hash_str, hash_molecule(z, mirror).hash_str)


if __name__ == "__main__":
    unittest.main()
