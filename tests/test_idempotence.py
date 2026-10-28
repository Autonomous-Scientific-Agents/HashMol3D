"""Idempotent re-hashing (descriptor version 10).

A stored ``F`` descriptor carries canonical-frame coordinates on a grid ten
thousand times finer than the hashing precision; the hash covers only those
values rounded back to the precision grid. Re-hashing a stored descriptor with
``hash_descriptor`` is exact, and the coordinates from ``descriptor_geometry``
hash back to the same identifier through ``hash_molecule``.
"""

import warnings

import numpy as np
import pytest

import hashmol3d.core as core
from hashmol3d import (
    descriptor_geometry,
    hash_descriptor,
    hash_molecule,
)


def _motion(z, coords, rng):
    perm = rng.permutation(len(z))
    rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if rng.random() < 0.5:
        rot[:, 0] *= -1
    return z[perm], (coords @ rot.T + rng.uniform(-5, 5, size=3))[perm]


def _stored_rows(descriptor):
    return np.array(core._parse_frame_rows(descriptor.split("|F:", 1)[1]), dtype=np.int64)


def _stored_coords(descriptor, precision):
    rows = _stored_rows(descriptor)
    decimals = round(-np.log10(precision)) + core._FINE_DIGITS
    return rows[:, 0], rows[:, 1:] / 10.0**decimals


def _round_trip(result, precision):
    z, coords = descriptor_geometry(result.descriptor)
    return hash_molecule(z, coords, precision=precision)


# QM9 molecules (four-decimal SDF coordinates) whose stored rows do not
# re-hash on their own: gdb_10781 stores an exact tie on the precision grid,
# gdb_2216 a value one re-hash shift from a boundary. gdb_62510 at 1e-3 Å
# needs settling only on the second round trip, in the anchored branch.
QM9_SETTLE_CASES = {
    "gdb_10781": (
        [6, 6, 6, 6, 6, 8, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [
            [-0.8169, 1.2278, 0.1722],
            [-0.0291, -0.0687, 0.0457],
            [0.7874, -0.3931, -1.2539],
            [1.2328, 0.8001, -2.0960],
            [1.9041, -0.9863, -0.3272],
            [3.2510, -0.6986, -0.5893],
            [1.2608, -0.2496, 0.8877],
            [1.9003, 1.0053, 1.1113],
            [-0.1805, 2.1031, 0.0274],
            [-1.6256, 1.2642, -0.5654],
            [-1.2697, 1.3110, 1.1660],
            [-0.6901, -0.9145, 0.2658],
            [0.3124, -1.1332, -1.9057],
            [1.6785, 1.5909, -1.4855],
            [1.9769, 0.4859, -2.8344],
            [0.3859, 1.2369, -2.6347],
            [1.8212, -2.0732, -0.2393],
            [3.3497, 0.2495, -0.4340],
            [1.1667, -0.8026, 1.8302],
            [2.6208, 0.8577, 1.7316],
        ],
    ),
    "gdb_2216": (
        [8, 6, 6, 6, 6, 8, 6, 1, 1, 1, 1, 1, 1, 1, 1],
        [
            [0.0619, 1.4340, 0.1970],
            [0.0140, 0.0552, -0.0055],
            [0.3714, -0.5075, -1.4113],
            [-0.9940, -1.2690, -1.4474],
            [-1.3281, -0.7277, -0.0387],
            [-0.8742, -1.6089, 1.0100],
            [0.3642, -0.8782, 1.1738],
            [0.9413, 1.7473, -0.0416],
            [0.4340, 0.2877, -2.1583],
            [1.2761, -1.1187, -1.4659],
            [-1.6673, -0.9111, -2.2301],
            [-0.9393, -2.3581, -1.4903],
            [-2.3137, -0.3010, 0.1603],
            [1.2471, -1.5222, 1.0722],
            [0.3976, -0.3499, 2.1351],
        ],
    ),
    "gdb_62510": (
        [6, 6, 8, 6, 6, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [
            [-0.0858, 1.5458, -0.0504],
            [0.0206, 0.0436, 0.0596],
            [-1.1790, -0.6290, -0.2692],
            [1.3231, -0.6913, -0.2367],
            [1.3344, -1.9728, -1.0382],
            [2.3938, 0.1759, -0.5306],
            [0.7553, -0.6655, 1.1880],
            [1.4630, 0.0847, 2.2894],
            [0.1539, -1.8361, 1.6697],
            [0.8595, 2.0327, 0.1881],
            [-0.3543, 1.8310, -1.0748],
            [-0.8660, 1.9218, 0.6188],
            [-1.2405, -0.7051, -1.2274],
            [1.2752, -1.7606, -2.1116],
            [2.2705, -2.5154, -0.8603],
            [0.5169, -2.6365, -0.7569],
            [3.1983, -0.2237, -0.1807],
            [2.0665, 0.9048, 1.8981],
            [0.7424, 0.4862, 3.0087],
            [2.1218, -0.6035, 2.8299],
            [-0.7021, -1.9013, 1.2241],
        ],
    ),
}


def _qm9(name):
    z, coords = QM9_SETTLE_CASES[name]
    return np.array(z), np.array(coords, dtype=float)


_W_METHANE = "1,0:0-178,0-178,0-178,1-109;1,0:0-178,0-178,0-178,1-109;1,0:0-178,0-178,0-178,1-109;1,0:0-178,0-178,0-178,1-109;6,1:0-109,0-109,0-109,0-109"


class TestHashDescriptor:
    @pytest.mark.parametrize("method", ["frame", "distance"])
    @pytest.mark.parametrize("length", [None, 8, 64])
    def test_rehash_of_stored_descriptor_is_exact(self, water, benzene, method, length):
        for z, coords in (water, benzene):
            res = hash_molecule(z, coords, method=method, length=length)
            assert hash_descriptor(res.descriptor, length=length) == res.geometry_hash
            digest = hash_descriptor(res.descriptor, length=length)
            assert res.hash_str.split("_")[5] == digest

    def test_version_9_descriptor_rehashes_to_its_own_digest(self):
        # The pre-v10 water descriptor and identifier quoted by version 0.13.0.
        v9 = "V:9-FRAME-SHA256|P:1.0e-04|Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0"
        assert hash_descriptor(v9) == "2525b97db42fc1c282844ba9478d561f"

    def test_version_10_descriptor_rehashes_to_its_own_digest(self):
        # The water descriptor and identifier quoted by version 0.14.0.
        v10 = (
            "V:10-FRAME-SHA256|P:1.0e-04|Z:1,1,8|"
            "F:1:0,-46880000,-75720000;1:0,-46880000,75720000;8:0,11720000,0"
        )
        assert hash_descriptor(v10) == "8ec3152acf35859ba58ced3e101c87e1"

    def test_payload_is_the_z_field_and_hashed_rows_only(self, water):
        # Version 11 hashes neither the version nor the precision: they are
        # in the readable identifier. The hashed rows equal version 9's.
        res = hash_molecule(*water)
        assert core._hashed_payload(res.descriptor) == (
            "Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0"
        )
        distance = hash_molecule(*water, method="distance")
        assert core._hashed_payload(distance.descriptor) == "Z:1,1,8|D:15144,9575,9575"

    @pytest.mark.parametrize(
        "descriptor, digest",
        [
            # Emitted by release 0.7.0, which wrote any precision as given.
            ("V:5-CANON-SHA256|P:5.0e-02|Z:1,1,8|C:15,10,10", "435a52219bdb3b356dcfcd79d5599d2c"),
            ("V:5-CANON-SHA256|P:2.0e+00|Z:1,1,8|C:2,1,1", "82afba4a13e26fe68509f42cae80b268"),
            # Release 0.8.0 added F sections under version 5.
            (
                "V:5-CANON-SHA256|P:1.0e-04|Z:1,1,8|F:1:0,-4692,-7572;1:0,-4692,7572;8:0,1173,0",
                "24168e6cd755057a16ff9219bf595524",
            ),
            # The W fallback of versions 5 and 6 (methane, search budget 1).
            (
                "V:5-CANON-SHA256|P:1.0e-02|Z:1,1,1,1,6|W:" + _W_METHANE,
                "b1be682102cbdbf3b36dbec8d777f439",
            ),
            (
                "V:6-FRAME-SHA256|P:1.0e-02|Z:1,1,1,1,6|W:" + _W_METHANE,
                "82e9e70ceaa3f6e675e21713aa9d32db",
            ),
        ],
    )
    def test_historical_descriptors_rehash_to_their_own_digest(self, descriptor, digest):
        assert hash_descriptor(descriptor) == digest

    @pytest.mark.parametrize(
        "descriptor, message",
        [
            ("not a descriptor", "not a HashMol3D descriptor"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|X:6:0,0,0", "not a HashMol3D descriptor"),
            ("V:x-FRAME-SHA256|P:1.0e-04|Z:6|F:6:0,0,0", "unrecognized descriptor version"),
            ("V:12.0|P:1.0e-04|Z:6|F:6:0,0,0", "newer than this HashMol3D"),
            ("V:11.1|P:1.0e-04|Z:6|F:6:0,0,0", "newer than this HashMol3D"),
            ("V:11.00|P:1.0e-04|Z:6|F:6:0,0,0", "unrecognized descriptor version"),
            ("V:11-FRAME-SHA256|P:1.0e-04|Z:6|F:6:0,0,0", "unsupported descriptor version"),
            ("V:11.0|P:1.0e-04|Z:6,6|C:12345", "has no 'C' section"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|D:12345", "has no 'D' section"),
            ("V:11.0|P:1.0e-04|Z:6,6|D:68719476737", "resolution floor"),
            ("V:11.0|P:1.0e-04|Z:6|F:6:5000,0,0", "rounding boundary"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:0,0", "malformed F row"),
            ("V:10-FRAME-SHA256|P:invalid|Z:6|C:invalid", "precision 'invalid'"),
            ("V:10-FRAME-SHA256|P:1e-4|Z:6|F:6:0,0,0", "not written as 1.0e-04"),
            ("V:10-FRAME-SHA256|P:1.0e+01|Z:6|F:6:0,0,0", "power of ten <= 1"),
            ("V:4-GEOM-SHA256|P:1.0e-04|Z:6|C:", "unsupported descriptor version"),
            ("V:10-CANON-SHA256|P:1.0e-04|Z:6|C:", "unsupported descriptor version"),
            ("V:7-FRAME-SHA256|P:1.0e-02|Z:1,1,1,1,6|W:" + _W_METHANE, "has no 'W' section"),
            ("V:6-FRAME-SHA256|P:5.0e-02|Z:6|C:", "power of ten <= 1"),
            ("V:5-CANON-SHA256|P:0.0e+00|Z:6|C:", "positive finite"),
            ("V:5-CANON-SHA256|P:inf|Z:6|C:", "positive finite"),
            ("V:5-CANON-SHA256|P:5.00e-02|Z:6|C:", "not written as 5.0e-02"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:6|W:6,0:", "1 W rows for 1 atoms"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:1,6|W:1,0:1-109;6,1:0", "malformed W row"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:1,6|W:1,0:1-109;6,1:0-+109", "canonical decimal"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:1,8|W:1,0:1-109;6,1:0-109", "do not match the Z"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:1,6|W:6,1:0-109;1,0:1-109", "do not match the Z"),
            ("V:5-CANON-SHA256|P:1.0e-02|Z:1,6|W:1,0:0-109;6,1:0-109", "colors do not match"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:8|F:6:0,0,0", "do not match the Z field"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:|F:6:0,0,0", "atomic number ''"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:119|F:119:0,0,0", r"\[1, 118\]"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:8,1|C:9572", "ascending"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:06|F:06:0,0,0", "canonical decimal"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:+1,0,0", "canonical decimal"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:-0,0,0", "canonical decimal"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|F:6:0,0,0", "1 F rows for 2 atoms"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:5000,0,0", "rounding boundary"),
            (
                "V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|F:6:0,0,10000;6:0,0,-10000",
                "not in canonical order",
            ),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:4611686018427387904,0,0", "format.s range"),
            # 2**35 + 1 grid cells from the centroid: past the resolution floor.
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6|F:6:343597383690000,0,0", "resolution floor"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|C:", "0 distances for 2 atoms"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|C:1,2", "2 distances for 2 atoms"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|C:-1", "canonical decimal"),
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|C:68719476737", "resolution floor"),
            # 1.5 * 2**35 cells: within twice the floor, but the hydrogen is
            # 118/119 of it from the weighted centroid, past the floor.
            ("V:10-FRAME-SHA256|P:1.0e-04|Z:1,118|C:51539607552", "resolution floor"),
        ],
    )
    def test_rejects_malformed_descriptors(self, descriptor, message):
        with pytest.raises(ValueError, match=message):
            hash_descriptor(descriptor)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            hash_descriptor(b"V:10")


class TestNormalization:
    @pytest.mark.parametrize(
        "fine, coarse",
        [
            (0, 0),
            (4999, 0),
            (5000, 0),
            (5001, 1),
            (15000, 2),
            (25000, 2),
            (-4999, 0),
            (-5000, 0),
            (-5001, -1),
            (-15000, -2),
            (-25000, -2),
            (2**70 + 5000, 2**70 // 10**4 + 1),
        ],
    )
    def test_round_half_even_in_exact_integers(self, fine, coarse):
        assert core._coarsen_int(fine) == coarse

    def test_vector_and_scalar_rounding_agree(self):
        rng = np.random.default_rng(5)
        fine = rng.integers(-(10**12), 10**12, size=4000)
        fine[::4] = (fine[::4] // 10**4) * 10**4 + 5000  # exact ties
        expected = [core._coarsen_int(int(v)) for v in fine]
        assert core._coarsen(fine).tolist() == expected

    def test_payload_rounds_only_frame_rows(self):
        stored = "V:10-FRAME-SHA256|P:1.0e-02|Z:1,6|F:1:-15003,4997,-4996;6:0,0,25003"
        assert core._hashed_payload(stored) == (
            "V:10-FRAME-SHA256|P:1.0e-02|Z:1,6|F:1:-2,0,0;6:0,0,3"
        )
        c_desc = "V:10-FRAME-SHA256|P:1.0e-04|Z:6,6|C:12345"
        assert core._hashed_payload(c_desc) == c_desc

    def test_stored_values_keep_a_margin_from_rounding_boundaries(self, benzene):
        z, coords = benzene
        rng = np.random.default_rng(3)
        for _ in range(20):
            noisy = coords + rng.normal(scale=1e-3, size=coords.shape)
            fine = _stored_rows(hash_molecule(z, noisy).descriptor)[:, 1:]
            offset = fine - core._coarsen(fine) * core._FINE_FACTOR
            assert np.abs(offset).max() <= core._FINE_FACTOR // 2 - core._FINE_MARGIN


class TestGeometryRoundTrip:
    @pytest.mark.parametrize("precision", [1e-2, 1e-4, 1e-6])
    def test_fixtures_round_trip(self, water, benzene, chiral_chfclbr, precision):
        for z, coords in (water, benzene, chiral_chfclbr):
            res = hash_molecule(z, coords, precision=precision)
            assert _round_trip(res, precision).hash_str == res.hash_str

    @pytest.mark.parametrize(
        "z, coords",
        [
            ([6], [[0.3, -0.2, 1.0]]),
            ([6, 8], [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
            ([8, 6, 8], [[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]]),
            (
                [6, 1, 1, 1, 1],
                np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
                * 1.09
                / np.sqrt(3),
            ),
        ],
        ids=["atom", "diatomic", "linear", "methane"],
    )
    def test_degenerate_branches_round_trip(self, z, coords):
        z, coords = np.array(z), np.array(coords, dtype=float)
        res = hash_molecule(z, coords)
        assert "|F:" in res.descriptor
        zz, cc = descriptor_geometry(res.descriptor)
        rng = np.random.default_rng(11)
        for _ in range(10):
            assert hash_molecule(*_motion(zz, cc, rng)).hash_str == res.hash_str

    @pytest.mark.parametrize("n", [3, 5, 10, 30, 80])
    def test_random_clouds_round_trip_under_rigid_motion(self, n):
        rng = np.random.default_rng(n)
        for _ in range(25):
            z = rng.choice([1, 6, 7, 8, 16], size=n)
            coords = rng.normal(size=(n, 3)) * n ** (1 / 3)
            res = hash_molecule(z, coords)
            zz, cc = descriptor_geometry(res.descriptor)
            assert hash_molecule(*_motion(zz, cc, rng)).hash_str == res.hash_str

    def test_round_trip_is_stable_under_repetition(self, benzene):
        res = hash_molecule(*benzene)
        current = res
        for _ in range(5):
            current = _round_trip(current, 1e-4)
            assert current.hash_str == res.hash_str

    def test_geometry_is_centered_in_canonical_frame(self, water):
        z, coords = descriptor_geometry(hash_molecule(*water).descriptor)
        assert z.tolist() == [1, 1, 8]
        np.testing.assert_allclose(z.astype(float) @ coords, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            coords, [[0, -0.4688, -0.7572], [0, -0.4688, 0.7572], [0, 0.1172, 0]], atol=1e-12
        )

    def test_canonical_descriptor_has_no_geometry(self, water):
        res = hash_molecule(*water, method="distance")
        with pytest.raises(ValueError, match="distance matrix"):
            descriptor_geometry(res.descriptor)

    def test_other_versions_have_no_rehashable_geometry(self):
        v9 = "V:9-FRAME-SHA256|P:1.0e-04|Z:1,1,8|F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0"
        with pytest.raises(ValueError, match="cannot be re-hashed"):
            descriptor_geometry(v9)


class TestResolutionFloorBoundary:
    """A geometry accepted at the resolution floor reconstructs to one that is
    accepted too: the floor of an F descriptor is decided on its hashed rows,
    which a verified reconstruction shares exactly."""

    def test_reconstruction_at_the_floor_rehashes(self):
        rng = np.random.default_rng(0)
        accepted = 0
        for _ in range(300):
            n = int(rng.integers(2, 12))
            precision = 10.0 ** -int(rng.integers(6, 12))
            z = rng.choice([1, 6, 7, 8], n)
            coords = rng.normal(size=(n, 3))
            w = z.astype(float)
            coords -= (w @ coords) / w.sum()
            # Extent within a few float64 ulps of the floor, either side.
            target = precision * core._RESOLUTION_CELLS * (1 + rng.uniform(-3e-12, 3e-12))
            coords *= target / np.linalg.norm(coords, axis=1).max()
            try:
                res = hash_molecule(z, coords, precision=precision)
            except ValueError:
                continue
            accepted += 1
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                zz, cc = descriptor_geometry(res.descriptor)
                assert hash_molecule(zz, cc, precision=precision).hash_str == res.hash_str
        assert 50 < accepted < 250  # both sides of the floor were exercised

    def test_screen_never_rejects_what_the_exact_rule_accepts(self):
        # A raw extent 0.4 cells beyond the floor rounds to a hashed row at
        # the limit and is accepted; 0.6 cells beyond rounds past it.
        cells = core._RESOLUTION_CELLS
        for excess, accepted in ((0.4, True), (0.6, False)):
            half = (cells + excess) * 1e-4
            coords = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
            if accepted:
                res = hash_molecule([6, 6], coords, precision=1e-4)
                assert core._hashed_payload(res.descriptor).endswith(f"6:0,0,{cells}")
            else:
                with pytest.raises(ValueError, match="float64 can resolve"):
                    hash_molecule([6, 6], coords, precision=1e-4)

    def test_distance_floor_is_decided_on_the_stored_distances(self):
        # hash_molecule and the parser apply the same exact rule to the C
        # distances, so every C descriptor near the floor that hash_molecule
        # emits is accepted, and every one it refuses would be rejected.
        rng = np.random.default_rng(1)
        cells = core._RESOLUTION_CELLS
        outcomes = set()
        for _ in range(300):
            n = int(rng.integers(2, 8))
            z = rng.choice([1, 8, 53, 118], n)
            coords = rng.normal(size=(n, 3))
            coords *= 1e-4 * cells * (1 + rng.uniform(-3, 3) / cells) / core._extent(z, coords)
            qmat = core._scaled_distances(coords, 4)
            _, z_ordered, body = core._canonical_signature(z, qmat)
            descriptor = core._format_descriptor(
                core.DESCRIPTOR_VERSION, 1e-4, z_ordered, "D", body
            )
            try:
                res = hash_molecule(z, coords, precision=1e-4, method="distance")
            except ValueError:
                with pytest.raises(ValueError, match="resolution floor"):
                    hash_descriptor(descriptor)
                outcomes.add(False)
            else:
                assert res.descriptor == descriptor
                assert hash_descriptor(descriptor) == res.geometry_hash
                outcomes.add(True)
        assert outcomes == {True, False}

    @pytest.mark.parametrize(
        "z, distance, accepted",
        [
            ((1, 1), 2 * 2**35, True),  # each atom exactly 2**35 cells out
            ((1, 1), 2 * 2**35 + 1, False),
            ((1, 118), 2**35 * 119 // 118, True),
            ((1, 118), 2**35 * 119 // 118 + 1, False),
        ],
    )
    def test_distance_floor_is_exact(self, z, distance, accepted):
        qmat = np.array([[0, distance], [distance, 0]], dtype=np.int64)
        assert core._distances_within_floor(np.array(z), qmat) is accepted


class TestSettling:
    @pytest.mark.parametrize("name", ["gdb_10781", "gdb_2216"])
    def test_stored_rows_that_miss_are_settled(self, name):
        z, coords = _qm9(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = hash_molecule(z, coords)
            # The stored rows alone land in a neighbouring cell...
            assert hash_molecule(*_stored_coords(res.descriptor, 1e-4)).hash_str != res.hash_str
            # ...the returned geometry does not, in any orientation.
            zz, cc = descriptor_geometry(res.descriptor)
            rng = np.random.default_rng(1)
            for _ in range(10):
                assert hash_molecule(*_motion(zz, cc, rng)).hash_str == res.hash_str
        # Settling moves values by a small fraction of a cell.
        _, stored = _stored_coords(res.descriptor, 1e-4)
        assert np.abs(cc - stored).max() < 0.01 * 1e-4

    def test_anchored_frame_is_settled_without_rotating_it(self):
        z, coords = _qm9("gdb_62510")
        first = hash_molecule(z, coords, precision=1e-3)
        second = _round_trip(first, 1e-3)
        assert second.hash_str == first.hash_str
        assert (
            hash_molecule(*_stored_coords(second.descriptor, 1e-3), precision=1e-3).hash_str
            != first.hash_str
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _round_trip(second, 1e-3).hash_str == first.hash_str

    def test_unsettled_geometry_warns_and_returns_stored_rows(self, monkeypatch):
        monkeypatch.setattr(core, "_REPAIR_STEPS", 0)
        z, coords = _qm9("gdb_10781")
        res = hash_molecule(z, coords)
        with pytest.warns(UserWarning, match="could not be settled"):
            zz, cc = descriptor_geometry(res.descriptor)
        np.testing.assert_array_equal(cc, _stored_coords(res.descriptor, 1e-4)[1])
