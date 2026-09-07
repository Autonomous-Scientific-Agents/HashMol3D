"""Independent verification of the HashMol3D v6 default frame descriptor.

The retained canonical option is checked explicitly where its distinct path
or fallback behavior matters.
"""

import os
import sys
import warnings

# Locate the package whether run from repo root or from paper/.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, os.path.join(_here, "src"))
import numpy as np

from hashmol3d import __version__
from hashmol3d.core import DESCRIPTOR_VERSION, hash_molecule

rng = np.random.default_rng(0)
print(f"HashMol3D package version: {__version__}")
print(f"descriptor version: {DESCRIPTOR_VERSION}")


def h(z, xyz, **kw):
    kw.setdefault("method", "frame")
    return hash_molecule(z, xyz, **kw).geometry_hash


def rand_rot():
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def dist_multiset(xyz, dec=4):
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    return sorted(
        round(float(np.linalg.norm(xyz[i] - xyz[j])), dec)
        for i in range(n)
        for j in range(i + 1, n)
    )


def brute_congruent(a, b, dec=6):
    """Exhaustive congruence check for small same-element point sets:
    congruent iff some atom pairing matches all pairwise distances."""
    from itertools import permutations

    a, b = np.asarray(a, float), np.asarray(b, float)
    da = np.round(np.linalg.norm(a[:, None] - a[None, :], axis=-1), dec)
    db = np.round(np.linalg.norm(b[:, None] - b[None, :], axis=-1), dec)
    n = len(a)
    return any(np.array_equal(da[np.ix_(p, p)], db) for p in permutations(range(n)))


results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# ---- Water reference ----
zw = [8, 1, 1]
xw = np.array([[0, 0, 0], [0.7572, 0.5860, 0], [-0.7572, 0.5860, 0]], float)
base = h(zw, xw)
check("frame method is the default (F tag)", "|F:" in hash_molecule(zw, xw).descriptor)

# 1. Rotation invariance
ok = all(h(zw, xw @ rand_rot().T) == base for _ in range(100))
check("rotation invariance (100x)", ok)

# 2. Translation invariance
ok = all(h(zw, xw + rng.standard_normal(3) * 10) == base for _ in range(100))
check("translation invariance (100x)", ok)

# 3. Permutation invariance
ok = True
for _ in range(100):
    p = rng.permutation(3)
    if h(np.array(zw)[p], xw[p]) != base:
        ok = False
check("permutation invariance (100x)", ok)

# 4. Reflection/parity invariance
check("reflection invariance", h(zw, xw * np.array([-1, 1, 1])) == base)

# 5. Enantiomer collision (documented)
zc = [6, 1, 9, 17, 35]
xc = np.array(
    [[0, 0, 0], [0, 0, 1.09], [1.03, 0, -0.36], [-0.5, 0.89, -0.36], [-0.5, -0.89, -0.36]], float
)
check("enantiomers hash identically (documented)", h(zc, xc) == h(zc, xc * np.array([-1, 1, 1])))

# 6. Linear molecule CO2
zco2 = [8, 6, 8]
xco2 = np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]], float)
b_co2 = h(zco2, xco2)
ok = all(h(zco2, (xco2 @ rand_rot().T) + rng.standard_normal(3)) == b_co2 for _ in range(50))
check("linear molecule (CO2) rigid-motion invariance", ok)

# 7. Overlapping atoms (distance 0)
try:
    hv = h([1, 1], [[0, 0, 0], [0, 0, 0]])
    check("overlapping atoms handled (no crash)", True)
except Exception as e:
    check(f"overlapping atoms handled (no crash) [{e}]", False)

# 8. Isotopes: the API takes atomic numbers only (no masses), so
#    isotopologues are byte-identical inputs by construction. This is a
#    definitional property, not a runtime check, so it is printed as a note.
print("     isotopologues: identical by construction (API takes Z only, no masses)")

# 9. Distinct geometries -> distinct hashes
check(
    "different bond length -> different hash",
    h([1, 1], [[0, 0, 0], [0, 0, 0.74]]) != h([1, 1], [[0, 0, 0], [0, 0, 0.80]]),
)


# 10. TRUE homometric pair (Bloom): {0,1,4,10,12,17} vs {0,1,8,11,13,17}.
#     Same 15 pairwise distances, provably not congruent (B is not a
#     translate or mirror of A). The v4 multiset descriptor merged these;
#     the v5 canonical descriptor must separate them.
def on_line(pts):
    x = np.zeros((len(pts), 3))
    x[:, 0] = pts
    return x


A6 = on_line([0, 1, 4, 10, 12, 17])
B6 = on_line([0, 1, 8, 11, 13, 17])
z6 = [6] * 6
check("homometric premise: distance multisets equal", dist_multiset(A6) == dist_multiset(B6))
check("homometric premise: NOT congruent (all 6! pairings)", not brute_congruent(A6, B6))
check("homometric pair SEPARATED by frame descriptor", h(z6, A6) != h(z6, B6))

# 10b. The 4-point pair {0,1,4,6} vs {0,2,5,6} shares a distance multiset but
#      is CONGRUENT (B = 6 - A, a mirror image, i.e. a proper 3D rotation of A
#      about a perpendicular axis); it must therefore hash identically.
A4 = on_line([0, 1, 4, 6])
B4 = on_line([0, 2, 5, 6])
check("4-point pair premise: congruent (mirror)", brute_congruent(A4, B4))
check("congruent (mirror) 4-point pair correctly merged", h([6] * 4, A4) == h([6] * 4, B4))

# 11. 1-WL counterexample: two 5-point sets sharing the full per-atom sorted
#     distance-row multiset (identical atomic environments to first order)
#     yet non-congruent (verified by exhaustive search over all 120 pairings).
#     The complete frame descriptor must separate these.
P1 = np.array([[0, 2, 1], [1, 3, 3], [1, 1, 1], [3, 1, 1], [3, 3, 3]], float)
P2 = np.array([[2, 0, 1], [3, 3, 1], [1, 1, 1], [1, 1, 3], [3, 3, 3]], float)
rows = lambda X: sorted(
    tuple(np.sort([round(float(np.linalg.norm(p - q2)), 4) for q2 in X]).tolist()) for p in X
)
check("1-WL premise: per-atom distance rows equal", rows(P1) == rows(P2))
check("1-WL premise: NOT congruent (all 5! pairings)", not brute_congruent(P1, P2))
check("1-WL-indistinguishable pair SEPARATED", h([6] * 5, P1) != h([6] * 5, P2))

# 12. Symmetric-molecule permutation invariance (canonical search branches
#     over the D6h orbits of benzene; every branch order must agree).
ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
xb = np.zeros((12, 3))
xb[:6, 0], xb[:6, 1] = 1.40 * np.cos(ang), 1.40 * np.sin(ang)
xb[6:, 0], xb[6:, 1] = 2.49 * np.cos(ang), 2.49 * np.sin(ang)
zb = np.array([6] * 6 + [1] * 6)
bb = h(zb, xb)
ok = True
for _ in range(50):
    p = rng.permutation(12)
    if h(zb[p], (xb @ rand_rot().T)[p]) != bb:
        ok = False
check("benzene (D6h) permutation+rotation invariance (50x)", ok)

# 12b. Spherical-top methane exercises the fully degenerate two-atom anchor.
a = 1.09 / np.sqrt(3)
zm = np.array([6, 1, 1, 1, 1])
xm = np.array(
    [[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
    float,
)
bm = h(zm, xm)
ok = "|F:" in hash_molecule(zm, xm).descriptor
for _ in range(50):
    p = rng.permutation(len(zm))
    if h(zm[p], (xm @ rand_rot().T)[p]) != bm:
        ok = False
check("methane spherical-top anchored frame (50x)", ok)

# 12c. Both public methods remain available and occupy distinct namespaces.
rwf = hash_molecule(zw, xw, method="frame")
rwc = hash_molecule(zw, xw, method="canonical")
check(
    "explicit canonical option retained with distinct C/F descriptors",
    "|F:" in rwf.descriptor and "|C:" in rwc.descriptor
    and rwf.geometry_hash != rwc.geometry_hash,
)

# 12d. An ill-conditioned frame request deterministically uses the complete
# canonical fallback; this is distinct from the weak W budget fallback below.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    rff = hash_molecule([6, 6], [[0, 0, 0], [2, 0, 0]], precision=1.0)
rfc = hash_molecule(
    [6, 6], [[0, 0, 0], [2, 0, 0]], precision=1.0, method="canonical"
)
check("ill-conditioned frame uses deterministic canonical fallback", rff.descriptor == rfc.descriptor)

# 13. Degenerate-rounding fallback: 12 atoms in a 0.01 A box at 1 A precision
#     (every rounded distance is 0) must take the stable-WL fallback path
#     deterministically and stay permutation-invariant.
xd = rng.uniform(0, 0.01, size=(12, 3))
zd = np.full(12, 6)
rd = hash_molecule(zd, xd, precision=1.0, method="canonical")
ok = "|W:" in rd.descriptor
for _ in range(10):
    p = rng.permutation(12)
    if (
        hash_molecule(
            zd[p], (xd @ rand_rot().T)[p], precision=1.0, method="canonical"
        ).geometry_hash
        != rd.geometry_hash
    ):
        ok = False
check("degenerate-rounding fallback deterministic + invariant", ok)

# 14. Sub-precision noise stability
noisy = xw + rng.uniform(-1e-9, 1e-9, xw.shape)
check("sub-precision noise stable", h(zw, noisy, precision=1e-4) == base)

# 15. Rounding-boundary illustration: the two methods quantize different
# quantities, so a perturbation that crosses one method's boundary need not
# cross the other's. A bond length straddling a C distance-bin edge changes C
# but not F; a length straddling an F centered-coordinate edge changes F but
# not C. (F rounds the centered coordinates +/- d/2, so its boundaries fall at
# different bond lengths than C's distance boundaries.)
def _pair(d, method):
    return h([1, 1], [[0, 0, 0], [0, 0, d]], precision=1e-4, method=method)


# 0.12345 straddles a C distance boundary but not an F coordinate boundary.
dc1, dc2 = 0.123450000001, 0.123449999999
check("C boundary crossing changes C descriptor", _pair(dc1, "canonical") != _pair(dc2, "canonical"))
check("same C boundary leaves F descriptor unchanged", _pair(dc1, "frame") == _pair(dc2, "frame"))
# 0.12350 straddles an F coordinate boundary but not a C distance boundary.
df1, df2 = 0.123500000001, 0.123499999999
check("F boundary crossing changes F descriptor", _pair(df1, "frame") != _pair(df2, "frame"))
check("same F boundary leaves C descriptor unchanged", _pair(df1, "canonical") == _pair(df2, "canonical"))

# 16. Charge/mult do not affect geometry hash
check("charge does not affect geom hash", h(zw, xw, charge=1) == h(zw, xw, charge=-2))
check(
    "multiplicity does not affect geom hash", h(zw, xw, multiplicity=1) == h(zw, xw, multiplicity=3)
)

# 17. Determinism
check("determinism (repeat)", h(zw, xw) == h(zw, xw) == base)

print("\nSUMMARY:", sum(1 for _, c in results if c), "/", len(results), "checks passed")
fails = [n for n, c in results if not c]
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
