"""Empirical comparison of candidate geometry-signature algorithms for HashMol3D.

Candidates (each maps (Z, coords) -> canonical descriptor string; equal
strings == equal hashes):

  pairs      -- current v4 scheme: sorted multiset of (Zmin, Zmax, d) triples.
  wl1        -- per-atom signatures: (Z_i, sorted [(Z_j, d_ij)]) multiset;
                one round of Weisfeiler-Leman refinement on the distance matrix.
  canonical  -- iterated WL refinement + individualization-refinement search
                over the rounded distance matrix; hashes the labeled distance
                matrix in a canonical atom order (complete invariant).
  inertia    -- canonical inertia frame: principal-axis coordinates, sign
                conventions, sorted rows.
  spectrum   -- sorted eigenvalues of the Coulomb matrix.

Battery:
  1. invariance under rotation / translation / permutation / reflection
  2. homometric pairs (must NOT collide; the current scheme does)
  3. bounded search for wl1-indistinguishable non-congruent pairs
  4. hash stability under coordinate noise (rounding-cliff behavior)
  5. wall-clock timing vs N, including symmetric (branching) stress cases

Outcome (2026-08): `canonical` shipped as descriptor version 5-CANON-SHA256
(src/hashmol3d/core.py). It distinguished every homometric pair, matched the
distance-multiset methods on rounding stability, ran faster than the v4 sort
for generic N >= 100, and -- decisively -- separated two 5-point lattice
pairs that share identical per-atom signatures (wl1 collides; both verified
non-congruent by brute force over all 120 pairings; kept as fixtures in
tests/test_collisions.py). `inertia` failed invariance on symmetric tops;
`spectrum` globalized rounding noise and is O(N^3). The implementations here
are comparison prototypes; `pairs` replicates v4. Canonical search exhaustion
now raises an error, matching the library, without substituting a weaker summary.

Run:  python experiments/compare_algorithms.py
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict

import numpy as np

DECIMALS = 4  # matches the library default precision of 1e-4 Angstrom


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def qmatrix(coords: np.ndarray, decimals: int = DECIMALS) -> np.ndarray:
    """Integer-scaled rounded distance matrix (int64, zero diagonal)."""
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    q = np.rint(dmat * (10.0**decimals)).astype(np.int64)
    np.fill_diagonal(q, 0)
    return q


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    h = rng.standard_normal((3, 3))
    qmat, _ = np.linalg.qr(h)
    if np.linalg.det(qmat) < 0:
        qmat[:, 0] *= -1
    return qmat


def scramble(z, coords, rng, reflect=False):
    """Random permutation + rotation + translation (+ optional reflection)."""
    perm = rng.permutation(len(z))
    r = random_rotation(rng)
    t = rng.uniform(-10, 10, size=3)
    c = coords @ r.T + t
    if reflect:
        c = c.copy()
        c[:, 0] *= -1
    return z[perm], c[perm]


# --------------------------------------------------------------------------
# candidate 1: current v4 scheme (sorted pair multiset)
# --------------------------------------------------------------------------
def sig_pairs(z: np.ndarray, coords: np.ndarray) -> str:
    q = qmatrix(coords)
    n = len(z)
    iu, ju = np.triu_indices(n, k=1)
    za = np.minimum(z[iu], z[ju])
    zb = np.maximum(z[iu], z[ju])
    trip = sorted(zip(za.tolist(), zb.tolist(), q[iu, ju].tolist()))
    zs = ",".join(map(str, sorted(z.tolist())))
    return f"P|{zs}|" + ",".join(f"{a}-{b}:{d}" for a, b, d in trip)


# --------------------------------------------------------------------------
# candidate 2: per-atom sorted distance signatures (one WL round)
# --------------------------------------------------------------------------
def sig_wl1(z: np.ndarray, coords: np.ndarray) -> str:
    q = qmatrix(coords)
    n = len(z)
    atoms = []
    for i in range(n):
        row = sorted((int(z[j]), int(q[i, j])) for j in range(n) if j != i)
        atoms.append((int(z[i]), tuple(row)))
    atoms.sort()
    parts = [f"{zi}:" + ",".join(f"{zj}-{d}" for zj, d in row) for zi, row in atoms]
    return "W1|" + ";".join(parts)


# --------------------------------------------------------------------------
# candidate 3: iterated WL refinement + canonical distance matrix
# --------------------------------------------------------------------------
class _BudgetExceeded(Exception):
    pass


def _refine(colors: np.ndarray, rank_q: np.ndarray, n_ranks: int) -> np.ndarray:
    """Iterate color refinement to a stable, canonically-ranked partition."""
    n = colors.shape[0]
    n_colors = int(colors.max()) + 1
    while n_colors < n:
        key = colors[None, :] * n_ranks + rank_q
        np.fill_diagonal(key, -1)  # self entry: sorts first, dropped below
        rows = np.sort(key, axis=1)[:, 1:]
        sig = np.concatenate([colors[:, None], rows], axis=1)
        _, inv = np.unique(sig, axis=0, return_inverse=True)
        new_colors = inv.reshape(-1).astype(np.int64)
        new_n = int(new_colors.max()) + 1
        if new_n == n_colors:
            break
        colors, n_colors = new_colors, new_n
    return colors


def _canonical_core(z, q, node_budget=10_000):
    """Return a complete canonical order, or raise on budget exhaustion."""
    n = len(z)
    _, inv = np.unique(z, return_inverse=True)
    colors0 = inv.reshape(-1).astype(np.int64)
    _, invq = np.unique(q, return_inverse=True)
    rank_q = invq.reshape(q.shape).astype(np.int64)
    n_ranks = int(rank_q.max()) + 1

    root = _refine(colors0, rank_q, n_ranks)
    iu, ju = np.triu_indices(n, k=1)

    best = None
    best_order = None
    nodes = 0
    # iterative DFS: each frame is [colors, members, next_member_index]
    stack: list[list] = []

    def visit(colors):
        nonlocal best, best_order, nodes
        nodes += 1
        if nodes > node_budget:
            raise _BudgetExceeded(
                "canonical search exhausted its budget; no descriptor created. "
                "Increase node_budget and retry."
            )
        k = int(colors.max()) + 1
        if k == n:
            order = np.argsort(colors, kind="stable")
            qc = q[np.ix_(order, order)]
            cand = np.ascontiguousarray(qc[iu, ju], dtype=">i8").tobytes()
            if best is None or cand < best:
                best, best_order = cand, order
            return None
        counts = np.bincount(colors, minlength=k)
        nonsingle = np.flatnonzero(counts > 1)
        target = int(nonsingle[np.argmin(counts[nonsingle])])
        members = np.flatnonzero(colors == target)
        return [colors, members, 0]

    frame = visit(root)
    if frame is not None:
        stack.append(frame)
    while stack:
        colors, members, idx = stack[-1]
        if idx >= len(members):
            stack.pop()
            continue
        stack[-1][2] = idx + 1
        child = colors * 2 + 1
        child[members[idx]] -= 1
        _, inv = np.unique(child, return_inverse=True)
        child = _refine(inv.reshape(-1).astype(np.int64), rank_q, n_ranks)
        frame = visit(child)
        if frame is not None:
            stack.append(frame)
    return best_order


def sig_canonical(z: np.ndarray, coords: np.ndarray) -> str:
    q = qmatrix(coords)
    n = len(z)
    if n == 1:
        return f"C|{int(z[0])}|"
    order = _canonical_core(z, q)
    zc = ",".join(str(int(v)) for v in z[order])
    iu, ju = np.triu_indices(n, k=1)
    qc = q[np.ix_(order, order)][iu, ju]
    return f"C|{zc}|" + ",".join(map(str, qc.tolist()))


# --------------------------------------------------------------------------
# candidate 4: canonical inertia frame
# --------------------------------------------------------------------------
def sig_inertia(z: np.ndarray, coords: np.ndarray) -> str:
    w = z.astype(float)
    c = coords - (w[:, None] * coords).sum(0) / w.sum()
    x2 = (c**2).sum(axis=1)
    t = np.zeros((3, 3))
    for k in range(3):
        t[k, k] = np.sum(w * (x2 - c[:, k] ** 2))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        t[a, b] = t[b, a] = -np.sum(w * c[:, a] * c[:, b])
    _, vecs = np.linalg.eigh(t)
    proj = c @ vecs
    # sign convention per axis: third moment positive
    for k in range(3):
        s = np.sum(w * proj[:, k] ** 3)
        if s < 0:
            proj[:, k] *= -1
    rows = np.column_stack([z.astype(float), np.round(proj, DECIMALS) + 0.0])
    order = np.lexsort(rows.T[::-1])
    body = ";".join(
        f"{int(r[0])}:{r[1]:.{DECIMALS}f},{r[2]:.{DECIMALS}f},{r[3]:.{DECIMALS}f}"
        for r in rows[order]
    )
    return "I|" + body


# --------------------------------------------------------------------------
# candidate 5: Coulomb-matrix eigenspectrum
# --------------------------------------------------------------------------
def sig_spectrum(z: np.ndarray, coords: np.ndarray) -> str:
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    zz = np.outer(z, z).astype(float)
    with np.errstate(divide="ignore"):
        m = np.where(dmat > 0, zz / np.where(dmat > 0, dmat, 1.0), 0.0)
    np.fill_diagonal(m, 0.5 * z.astype(float) ** 2.4)
    ev = np.sort(np.linalg.eigvalsh(m))
    zs = ",".join(map(str, sorted(z.tolist())))
    return f"S|{zs}|" + ",".join(f"{v:.{DECIMALS}f}" for v in np.round(ev, DECIMALS) + 0.0)


METHODS = {
    "pairs (current)": sig_pairs,
    "wl1": sig_wl1,
    "canonical": sig_canonical,
    "inertia": sig_inertia,
    "spectrum": sig_spectrum,
}


# --------------------------------------------------------------------------
# geometry battery
# --------------------------------------------------------------------------
def mol_water():
    z = np.array([8, 1, 1])
    c = np.array([[0, 0, 0], [0.7572, 0.586, 0], [-0.7572, 0.586, 0]], float)
    return z, c


def mol_benzene():
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    c = np.zeros((12, 3))
    c[:6, 0], c[:6, 1] = 1.40 * np.cos(ang), 1.40 * np.sin(ang)
    c[6:, 0], c[6:, 1] = 2.49 * np.cos(ang), 2.49 * np.sin(ang)
    return np.array([6] * 6 + [1] * 6), c


def mol_co2():
    return np.array([8, 6, 8]), np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]], float)


def mol_methane():
    a = 1.09 / np.sqrt(3)
    c = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]])
    return np.array([6, 1, 1, 1, 1]), c


def mol_chfclbr():
    z = np.array([6, 1, 9, 17, 35])
    c = np.array(
        [[0, 0, 0], [0, 0, 1.09], [1.03, 0, -0.36], [-0.5, 0.89, -0.36], [-0.5, -0.89, -0.36]]
    )
    return z, c


def mol_cubane():
    s = 1.55 / 2
    corners = np.array(list(itertools.product([-s, s], repeat=3)), float)
    hn = corners + 1.09 * corners / np.linalg.norm(corners, axis=1, keepdims=True)
    return np.array([6] * 8 + [1] * 8), np.vstack([corners, hn])


def mol_ring(n, elem=6, r=None):
    if r is None:
        r = 1.4 / (2 * np.sin(np.pi / n))  # ~1.4 A bond length
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)])
    return np.full(n, elem), c


def mol_cloud(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.choice([1, 6, 7, 8], size=n)
    c = rng.uniform(-5, 5, size=(n, 3)) * (n / 20) ** (1 / 3)
    return z, c


def mol_near_sym_top():
    """Benzene with one C radius stretched by 2e-4: near-degenerate inertia."""
    z, c = mol_benzene()
    c = c.copy()
    c[0, :2] *= 1 + 2e-4 / 1.40
    return z, c


BATTERY = {
    "water": mol_water(),
    "benzene": mol_benzene(),
    "co2": mol_co2(),
    "methane": mol_methane(),
    "chfclbr": mol_chfclbr(),
    "cubane": mol_cubane(),
    "ring24": mol_ring(24),
    "cloud40": mol_cloud(40, 7),
    "near_sym_top": mol_near_sym_top(),
}


# --------------------------------------------------------------------------
# experiment 1: invariance
# --------------------------------------------------------------------------
def run_invariance():
    print("\n=== 1. Invariance (perm + rot + trans, and reflection), 20 trials each ===")
    rng = np.random.default_rng(42)
    header = f"{'molecule':<14}" + "".join(f"{m:>18}" for m in METHODS)
    print(header)
    fails = defaultdict(set)
    for name, (z, c) in BATTERY.items():
        row = f"{name:<14}"
        for mname, fn in METHODS.items():
            base = fn(z, c)
            ok = True
            for t in range(20):
                zz, cc = scramble(z, c, rng, reflect=(t % 2 == 1))
                if fn(zz, cc) != base:
                    ok = False
                    break
            row += f"{'ok' if ok else 'FAIL':>18}"
            if not ok:
                fails[mname].add(name)
        print(row)
    return fails


# --------------------------------------------------------------------------
# experiment 2: homometric pairs
# --------------------------------------------------------------------------
def embed_line(points, elem=6, step=1.0):
    pts = np.asarray(points, float) * step
    c = np.zeros((len(pts), 3))
    c[:, 0] = pts
    return np.full(len(pts), elem), c


def find_1d_homometric(hi=17, k=6, limit=6):
    """Brute-force k-subsets of 0..hi sharing a difference multiset,
    excluding congruent (translate/mirror) pairs."""
    groups = defaultdict(list)
    for comb in itertools.combinations(range(hi + 1), k):
        if comb[0] != 0:
            continue
        diffs = tuple(sorted(b - a for a, b in itertools.combinations(comb, 2)))
        groups[diffs].append(comb)
    out = []
    for _, sets in groups.items():
        canon = set()
        reps = []
        for s in sets:
            m = s[-1]
            mirror = tuple(sorted(m - x for x in s))
            key = min(s, mirror)
            if key not in canon:
                canon.add(key)
                reps.append(s)
        if len(reps) > 1:
            out.append(reps[:2])
        if len(out) >= limit:
            break
    return out


def product_set(a, b, c_):
    pts = np.array([[x, y, w] for x in a for y in b for w in c_], float)
    return np.full(len(pts), 6), pts


def run_homometric():
    print("\n=== 2. Homometric pairs: does each method distinguish them? ===")
    cases = []
    classic_a, classic_b = (0, 1, 4, 10, 12, 17), (0, 1, 8, 11, 13, 17)
    cases.append(("classic-6pt-line", embed_line(classic_a), embed_line(classic_b)))
    for i, (sa, sb) in enumerate(find_1d_homometric()):
        if set(sa) == set(classic_a) and set(sb) == set(classic_b):
            continue
        cases.append((f"search-1d-{i}", embed_line(sa), embed_line(sb)))
    cases.append(
        (
            "3d-grid-24at",
            product_set(classic_a, (0, 1.1), (0, 1.3)),
            product_set(classic_b, (0, 1.1), (0, 1.3)),
        )
    )

    header = f"{'pair':<18}{'non-congruent?':>15}" + "".join(f"{m:>18}" for m in METHODS)
    print(header)
    scores = defaultdict(int)
    total = 0
    for name, (za, ca), (zb, cb) in cases:
        # rigorous one-way non-congruence certificate: full-precision per-atom
        # sorted distance rows differ => definitely not congruent
        ra = sorted(tuple(np.sort(np.linalg.norm(ca - p, axis=1)).round(9)) for p in ca)
        rb = sorted(tuple(np.sort(np.linalg.norm(cb - p, axis=1)).round(9)) for p in cb)
        noncong = ra != rb
        if not noncong:
            continue
        total += 1
        row = f"{name:<18}{'yes':>15}"
        for mname, fn in METHODS.items():
            distinct = fn(za, ca) != fn(zb, cb)
            scores[mname] += distinct
            row += f"{'distinct' if distinct else 'COLLIDE':>18}"
        print(row)
    print(f"\ndistinguished (of {total}): " + ", ".join(f"{m}={scores[m]}" for m in METHODS))
    return scores, total


# --------------------------------------------------------------------------
# experiment 3: bounded search for wl1-breaking configurations
# --------------------------------------------------------------------------
def run_wl1_break_search(n_samples=30_000, seed=3):
    print("\n=== 3. Random lattice search for wl1-equal but non-congruent pairs ===")
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    found = 0
    for _ in range(n_samples):
        npts = rng.integers(5, 8)
        pts = rng.integers(0, 4, size=(npts, 3)).astype(float)
        if len(np.unique(pts, axis=0)) != npts:
            continue
        z = np.full(npts, 6)
        w = sig_wl1(z, pts)
        c = sig_canonical(z, pts)
        for c2 in buckets[w]:
            if c2 != c:
                found += 1
        if c not in buckets[w]:
            buckets[w].append(c)
    print(f"samples={n_samples}  wl1-equal/canonical-different pairs found: {found}")
    return found


# --------------------------------------------------------------------------
# experiment 4: noise stability
# --------------------------------------------------------------------------
def run_noise():
    print("\n=== 4. Hash flip rate under coordinate noise (precision 1e-4 A) ===")
    rng = np.random.default_rng(11)
    mols = {k: BATTERY[k] for k in ("chfclbr", "benzene", "near_sym_top")}
    amps = [1e-9, 1e-7, 1e-6, 1e-5]
    for name, (z, c) in mols.items():
        print(f"\n  {name}: flip fraction over 100 noisy copies")
        print(f"  {'amp':>8}" + "".join(f"{m:>18}" for m in METHODS))
        for amp in amps:
            # identical noise samples for every method: differences in a row
            # then reflect the methods, not sampling variation
            noises = [rng.uniform(-amp, amp, c.shape) for _ in range(100)]
            row = f"  {amp:>8.0e}"
            for _, fn in METHODS.items():
                base = fn(z, c)
                flips = sum(fn(z, c + e) != base for e in noises)
                row += f"{flips / 100:>18.2f}"
            print(row)


# --------------------------------------------------------------------------
# experiment 5: performance
# --------------------------------------------------------------------------
def run_perf():
    print("\n=== 5. Timing (ms, best of 3) ===")
    cases = [
        ("cloud N=10", mol_cloud(10, 1)),
        ("cloud N=30", mol_cloud(30, 2)),
        ("cloud N=100", mol_cloud(100, 3)),
        ("cloud N=300", mol_cloud(300, 4)),
        ("cloud N=1000", mol_cloud(1000, 5)),
        ("ring N=60", mol_ring(60)),
        ("ring N=120", mol_ring(120)),
        ("benzene", mol_benzene()),
        ("cubane", mol_cubane()),
    ]
    header = f"{'case':<14}" + "".join(f"{m:>18}" for m in METHODS)
    print(header)
    for name, (z, c) in cases:
        row = f"{name:<14}"
        for _, fn in METHODS.items():
            best = min(_time_one(fn, z, c) for _ in range(3))
            row += f"{best * 1e3:>18.2f}"
        print(row)


def _time_one(fn, z, c):
    t0 = time.perf_counter()
    fn(z, c)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
def main():
    print("HashMol3D candidate-algorithm comparison")
    print("=" * 100)
    run_invariance()
    run_homometric()
    run_wl1_break_search()
    run_noise()
    run_perf()


if __name__ == "__main__":
    main()
