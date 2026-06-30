import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ============================================================
# 1. Result Container
# ============================================================


@dataclass
class HashMol3DResult:
    hash_str: str
    protocol: str
    precision: float
    charge: int
    multiplicity: int
    chiral_sign: str
    descriptor: str

    def __str__(self) -> str:
        return self.hash_str


# ============================================================
# 2. Pure Geometric Canonicalization
# ============================================================


# Fails for benzene H shuffle
def _get_geometric_canonical_order_v1(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> np.ndarray:
    """
    Sorts atoms based on:
    1. Atomic Number (Descending - heavier atoms first usually helps)
    2. The sorted list of distances to all other atoms (Geometric Environment)
    """
    n_atoms = len(atomic_nums)

    # 1. Compute Distance Matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)

    # 2. Create a "Environment Signature" for each atom
    # We round distances to avoid floating point noise affecting the sort
    # We sort the row so that the order of neighbors doesn't matter, only their distances
    env_signatures = np.sort(np.round(dmat, decimals=decimals), axis=1)

    # 3. Construct Sort Keys
    # Python's default sort is stable and handles tuples lexicographically.
    # We construct a list of tuples: (-AtomicNum, env_signature_tuple, original_index)
    sort_keys = []
    for i in range(n_atoms):
        # Tuple comparison:
        # Primary: Atomic Number (inverted for descending sort logic if desired,
        # but standard ascending Z is fine too. Let's stick to Ascending Z).
        z = int(atomic_nums[i])

        # Secondary: The geometric environment (tuple of float distances)
        env = tuple(env_signatures[i])

        sort_keys.append((z, env, i))

    # 4. Sort
    # If Z and Env are identical (symmetry), the original index breaks the tie.
    # Since the geometry is identical, the order among symmetric atoms doesn't change the hash.
    sort_keys.sort(key=lambda x: (x[0], x[1]))

    return np.array([x[2] for x in sort_keys], dtype=int)


# Might differ for different machines as it relies on eigh
def _get_geometric_canonical_order_v2(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> np.ndarray:
    """
    Sorts atoms to ensure permutation invariance, even for symmetric molecules
    like Benzene.

    Strategy:
    1. Group by Atomic Number (Heavy atoms first).
    2. Within groups, sort by 'Environment Signature' (distances to all others).
    3. BREAK TIES using projection onto Principal Inertia Axes.
       (Inertia tensor is invariant to permutation, so its axes are stable).
    """
    n_atoms = len(atomic_nums)

    # --- Step A: Primary Properties (Z and Geometry Signature) ---

    # 1. Distance Matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)

    # 2. Environment Signature
    #    (Sorted list of distances from atom i to all other atoms)
    #    This handles chemically distinct atoms (e.g. Meta vs Para in Toluene)
    env_signatures = np.sort(np.round(dmat, decimals=decimals), axis=1)

    # --- Step B: Tie-Breaker (Principal Axes Projection) ---

    # 1. Center coordinates (Mass weighted usually, but Z-weighted here is fine)
    #    Using Z as weight helps distinguish atoms chemically.
    weights = atomic_nums[:, None]
    center = np.sum(coords * weights, axis=0) / np.sum(weights)
    centered_coords = coords - center

    # 2. Compute Inertia Tensor (3x3)
    #    I_ab = sum( w_i * r_i_a * r_i_b )
    #    This summation is strictly PERMUTATION INVARIANT.
    inertia = np.dot((centered_coords * weights).T, centered_coords)

    # 3. Eigendecomposition to find Principal Axes
    #    eigh is preferred for symmetric matrices (guaranteed real eigenvalues)
    evals, evecs = np.linalg.eigh(inertia)

    # 4. Sort axes by eigenvalue (Smallest to Largest moment)
    #    This ensures x-axis is always the "long" axis, etc.
    #    Note: For Benzene (planar), 2 axes are degenerate.
    #    However, NumPy's eigh is deterministic for a fixed matrix.
    #    Since the matrix is fixed (invariant), the axes are stable.
    idx = np.argsort(evals)
    sorted_axes = evecs[:, idx]

    # 5. Project atoms onto these canonical axes
    projections = np.dot(centered_coords, sorted_axes)

    # Round projections to avoid float noise in sorting
    # (Using same precision as distances is usually safe)
    projections = np.round(projections, decimals=decimals)

    # --- Step C: Construct Sort Keys ---

    sort_keys = []
    for i in range(n_atoms):
        # 1. Atomic Number (Primary)
        z = int(atomic_nums[i])

        # 2. Environment Signature (Secondary - distinguishes distinct atoms)
        #    Tuple-fy for comparison
        env = tuple(env_signatures[i])

        # 3. Canonical Projection (Tertiary - breaks symmetry ties)
        #    If atoms are symmetric (same env), they must be in different
        #    positions in space. The projections distinguish them.
        proj = tuple(projections[i])

        sort_keys.append((z, env, proj, i))

    # --- Step D: Sort ---
    # Sort hierarchy: Z -> Env -> Projection -> (Original Index as fallback)
    sort_keys.sort(key=lambda x: (x[0], x[1], x[2]))

    return np.array([x[3] for x in sort_keys], dtype=int)


# failed edge case for benzene
def _get_geometric_canonical_order_v3(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> np.ndarray:
    """
    Robust Canonical Ordering using Iterative Geometric Refinement.
    Handles high-symmetry cases (Benzene, C60, Linear) where Inertia fails.
    """
    n_atoms = len(atomic_nums)

    # 1. Initial Invariants: Atomic Number + Environment Signature
    #    Environment Signature = Sorted list of rounded distances to all other atoms.
    #    This distinguishes chemically distinct atoms (e.g. Toluene ipso-C vs meta-C).
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    dmat_rounded = np.round(dmat, decimals=decimals)

    # Create initial sort keys
    # Each atom gets: (Z, [sorted_distances])
    initial_keys = []
    for i in range(n_atoms):
        z = int(atomic_nums[i])
        env = tuple(sorted(dmat_rounded[i].tolist()))
        initial_keys.append((z, env))

    # 2. Assign Initial Ranks based on these keys
    #    Get unique keys, sort them, assign integer ranks.
    unique_keys = sorted(list(set(initial_keys)))
    key_to_rank = {k: i for i, k in enumerate(unique_keys)}

    current_ranks = np.array([key_to_rank[k] for k in initial_keys])

    # 3. Partition atoms into tied groups
    #    We need to return a strict list [0, 1, 2...].
    #    If current_ranks has duplicates, we must break them.

    # If all ranks are unique, we are done.
    if len(set(current_ranks)) == n_atoms:
        # Sort indices by rank
        return np.argsort(current_ranks)

    # 4. Symmetry Breaking (The "Benzene Fix")
    #    If we have ties that geometry (environment) cannot break,
    #    it means the atoms are geometrically symmetric.
    #    We must break symmetry canonically.

    # Strategy:
    # We will compute a "Connectivity Score" for each atom based on
    # distances to *other* atoms, weighted by the other atoms' ranks.
    # Because this can be circular, we might need to brute-force the
    # "Lexicographically Smallest Distance Matrix" for the symmetric subset.

    # Simplified Robust Approach:
    # 1. Identify the highest-priority "Anchor" atom (unique rank).
    # 2. Use distances to that anchor to split ties in other atoms.
    # 3. If no unique anchor exists (e.g. Benzene), promote the
    #    atom that results in the smallest resulting distance string.

    # For performance/simplicity in this script, we use a deterministic
    # "lowest index in the sorted distance matrix" approach.

    # A. Construct the full sort key for every atom:
    #    Key = (Rank, Distance_Row_Sorted_By_Rank)
    #    This allows us to sort the atoms relative to each other.

    final_indices = list(range(n_atoms))

    # Custom comparator for the final sort
    def strict_compare(idx):
        # 1. Primary: The invariant rank we calculated earlier
        r = current_ranks[idx]

        # 2. Secondary: The actual row of distances, but sorted
        #    such that we compare distances to "Rank 0" atoms, then "Rank 1", etc.
        #    This ensures we are comparing geometry, not file index.

        # Get distances from atom 'idx' to all other atoms 'j'
        dists = dmat_rounded[idx]

        # Pair distance with the *rank* of the neighbor
        # (neighbor_rank, distance_to_neighbor)
        weighted_env = sorted([(current_ranks[j], dists[j]) for j in range(n_atoms)])

        # This tuple is rotation/permutation invariant
        return (r, weighted_env)

    # Sort indices based on this strict geometric comparison
    final_indices.sort(key=strict_compare)

    # Check for remaining ties (True Symmetry, e.g. Benzene H's)
    # If strict_compare(H1) == strict_compare(H2), they are symmetric.
    # We need to break the tie deterministically to ensure H1 always comes before H2
    # in the output string if we want strict reproducibility, BUT...
    #
    # CRITICAL: If atoms are truly symmetric (identical view of the universe),
    # swapping them in the output list DOES NOT change the Distance Matrix values
    # because D_ij == D_ji.
    #
    # However, to be safe against numerical noise or "near symmetry",
    # we usually fallback to the *input index* to stabilize the sort
    # for the run, provided the inputs were indistinguishable.
    #
    # But wait! If inputs are shuffled, input index is useless.
    #
    # The Solution for Benzene Permutation:
    # If comparison keys are identical, we accept ANY order among them,
    # because the resulting Distance Matrix string will be identical.
    #
    # Example: H1 and H2 are symmetric.
    # Order [H1, H2] -> D row includes d(H1, H2)
    # Order [H2, H1] -> D row includes d(H2, H1)
    # Since d(H1,H2) == d(H2,H1), the hash output is identical.

    return np.array(final_indices, dtype=int)


# Still fails for edge test 2
def _get_geometric_canonical_order_v4(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> np.ndarray:
    """
    Robust Canonical Ordering with Symmetry Breaking.
    Algorithm:
    1. Calculate 'Invariant Invariants' (Z, Distance Histogram).
    2. Iteratively refine ranks (Morgan Algorithm) using neighbors.
    3. If ties remain (Symmetry), break them by testing all candidates.
    """
    n_atoms = len(atomic_nums)

    # --- Step 1: Initial Invariants ---
    # We round distances to handle numerical noise cleanly
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    dmat_rounded = np.round(dmat, decimals=decimals)

    # Rank 0: Atomic Number
    # Rank 1: Sorted list of distances to all other atoms (Geometry Fingerprint)
    initial_invariants = []
    for i in range(n_atoms):
        z = int(atomic_nums[i])
        # Sort distances to ignore neighbor order
        env = tuple(sorted(dmat_rounded[i].tolist()))
        initial_invariants.append((z, env))

    # Convert invariants to integer ranks [0, 1, 2...]
    unique_invs = sorted(list(set(initial_invariants)))
    inv_map = {inv: idx for idx, inv in enumerate(unique_invs)}
    ranks = np.array([inv_map[inv] for inv in initial_invariants], dtype=int)

    # --- Step 2: Iterative Refinement (Morgan Algorithm) ---
    # If two atoms have the same Z and same distances, they might still be distinct
    # if their neighbors are different. We propagate rank info.

    while True:
        new_invariants = []
        for i in range(n_atoms):
            # Signature: (CurrentRank, SortedList_of_NeighborRanks_weighted_by_dist)
            # We pair (Distance, NeighborRank) and sort.
            # This captures: "I am close to a Rank 5 atom" vs "I am close to a Rank 2 atom"

            # Note: We use rounded distances for stability
            neighbors = []
            for j in range(n_atoms):
                if i == j:
                    continue
                neighbors.append((dmat_rounded[i, j], ranks[j]))

            neighbors.sort()  # Sort by distance, then by neighbor rank
            new_invariants.append((ranks[i], tuple(neighbors)))

        # Re-rank
        unique_new = sorted(list(set(new_invariants)))
        if len(unique_new) == len(unique_invs):
            # No new distinctions found. Convergence.
            break

        unique_invs = unique_new
        inv_map = {inv: idx for idx, inv in enumerate(unique_invs)}
        ranks = np.array([inv_map[inv] for inv in new_invariants], dtype=int)

    # --- Step 3: Symmetry Breaking (Automorphism) ---
    # If all ranks are unique, we are done.
    # If not (e.g. Benzene), we have perfect symmetry. We must break it.

    # We detect groups of tied atoms.
    # We pick the tied group with the *lowest rank* (highest priority).
    # We try *every* atom in that group as the "Pivot".
    # We compare the resulting full distance matrices lexicographically.

    if len(set(ranks)) == n_atoms:
        return np.argsort(ranks)

    # Identify ties
    from collections import defaultdict

    rank_groups = defaultdict(list)
    for idx, r in enumerate(ranks):
        rank_groups[r].append(idx)

    # Find the smallest rank that has ties
    tied_rank = -1
    for r in sorted(rank_groups.keys()):
        if len(rank_groups[r]) > 1:
            tied_rank = r
            break

    if tied_rank == -1:  # Should be caught by set check, but safety first
        return np.argsort(ranks)

    # Get candidates to break the tie
    candidates = rank_groups[tied_rank]

    best_perm = None
    best_d_string = None

    # Try each candidate as the "Winner" of the tie
    for pivot in candidates:
        # Create a temporary rank array where the pivot is promoted
        # We give the pivot a "Super Priority" (e.g. -1)
        temp_ranks = ranks.copy()
        # We assume ranks are positive integers. -1 makes it unique/first.
        # But we need to handle the whole sort.

        # Actually, simpler: Use the Pivot's index to calculate a deterministic
        # "Tie-Broken Sort".
        # We sort by: (Rank, Distance_To_Pivot, Tie_Breaker_Secondary...)

        # We construct a full permutation for this pivot choice
        # 1. Primary Key: The converged Morgan Rank
        # 2. Secondary Key: Distance to Pivot
        # 3. Tertiary Key: Neighbor Ranks relative to Pivot (Propagated)

        # For simplicity/speed in Python:
        # Just sort all atoms by (Rank, Distance_To_Pivot).
        # If that still leaves ties, we need the next pivot.
        # But usually, 1 pivot breaks Benzene.

        # Let's generate the sort key for this pivot
        # Key[i] = (Rank[i], Distance(i, pivot))

        key_list = []
        for i in range(n_atoms):
            key_list.append((temp_ranks[i], dmat_rounded[i, pivot], i))

        # Sort atoms based on this perspective
        # We need a stable output, so final fallback is just 'i' (original index)
        # BUT this fallback is dangerous.
        # However, we compare the RESULTS (d_string), so it's okay.
        key_list.sort()
        current_perm = [x[2] for x in key_list]

        # Construct the Descriptor String (Upper Triangle) for this permutation
        # We assume this permutation is the canonical one.
        # D_sorted = [d(p[0], p[1]), d(p[0], p[2])...]

        flat_d = []
        for r_i in range(n_atoms):
            for c_i in range(r_i + 1, n_atoms):
                idx1 = current_perm[r_i]
                idx2 = current_perm[c_i]
                flat_d.append(dmat_rounded[idx1, idx2])

        # Compare lexicographically
        # Python compares lists element-by-element
        if best_d_string is None or flat_d < best_d_string:
            best_d_string = flat_d
            best_perm = current_perm

    return np.array(best_perm, dtype=int)


def _get_geometric_canonical_order(
    atomic_nums: np.ndarray, coords: np.ndarray, decimals: int
) -> np.ndarray:
    """
    Robust Canonical Ordering using Greedy Geodesic Traversal.

    Strategy:
    1. Resolve "Morgan Ranks" (Z + Environment) to group similar atoms.
    2. Try every atom in the 'best' rank group as a Pivot (Start Atom).
    3. For each Pivot, construct a permutation greedily:
       - Next atom is the one with the lexicographically smallest distance
         vector to all currently placed atoms.
    4. Choose the Pivot/Permutation that yields the lexicographically
       smallest total Distance Matrix string.
    """
    n_atoms = len(atomic_nums)

    # --- Step 1: Initial Invariants (Morgan-style) ---
    # Round distances for stability
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    dmat_rounded = np.round(dmat, decimals=decimals)

    # 1a. Initial Rank: (AtomicNum, SortedEnv)
    initial_invariants = []
    for i in range(n_atoms):
        z = int(atomic_nums[i])
        env = tuple(sorted(dmat_rounded[i].tolist()))
        initial_invariants.append((z, env))

    unique_invs = sorted(list(set(initial_invariants)))
    inv_map = {inv: idx for idx, inv in enumerate(unique_invs)}
    ranks = np.array([inv_map[inv] for inv in initial_invariants], dtype=int)

    # 1b. Iterative Refinement (Propagate neighbor ranks)
    # This distinguishes chemically distinct atoms (e.g. Meta vs Para)
    while True:
        new_invariants = []
        for i in range(n_atoms):
            neighbors = []
            for j in range(n_atoms):
                if i == j:
                    continue
                # (Distance, NeighborRank)
                neighbors.append((dmat_rounded[i, j], ranks[j]))
            neighbors.sort()
            new_invariants.append((ranks[i], tuple(neighbors)))

        unique_new = sorted(list(set(new_invariants)))
        if len(unique_new) == len(unique_invs):
            break
        unique_invs = unique_new
        inv_map = {inv: idx for idx, inv in enumerate(unique_invs)}
        ranks = np.array([inv_map[inv] for inv in new_invariants], dtype=int)

    # --- Step 2: Tie-Breaking via Greedy Traversal ---

    # Identify candidates for the "Pivot" (Start Atom)
    # We only need to check atoms with the lowest Rank (highest priority)
    min_rank = np.min(ranks)
    candidates = np.where(ranks == min_rank)[0]

    best_perm = None
    best_d_string = None

    # Try each symmetric candidate as the starting point
    for start_node in candidates:

        # Initialize traversal
        current_perm = [start_node]
        remaining = set(range(n_atoms))
        remaining.remove(start_node)

        # We build the distance string incrementally to fail fast if suboptimal
        # But for simplicity, we just build the whole perm.

        while remaining:
            # Find the "best" next atom from remaining set
            # Criteria: Minimize vector [Rank, Dist_to_Node0, Dist_to_Node1...]
            best_next = -1
            best_vec = None

            for candidate in remaining:
                # Construct comparison vector
                # 1. Rank (prefer chemically simpler/heavy atoms)
                # 2. Distances to already placed atoms in order
                dists_to_placed = [dmat_rounded[candidate, p] for p in current_perm]

                # We use a tuple for comparison
                vec = (ranks[candidate], tuple(dists_to_placed))

                # Greedy Min
                if best_vec is None or vec < best_vec:
                    best_vec = vec
                    best_next = candidate
                elif vec == best_vec:
                    # True Symmetry at this step (e.g. C2 vs C6 in Benzene)
                    # Break tie by index (deterministic for this specific run)
                    # Since they are symmetric, the branch doesn't matter for the final hash.
                    if candidate < best_next:
                        best_next = candidate

            current_perm.append(best_next)
            remaining.remove(best_next)

        # --- Step 3: Evaluate this Permutation ---
        # Construct the canonical Distance Matrix string (Upper Triangle)
        flat_d = []
        for r_i in range(n_atoms):
            for c_i in range(r_i + 1, n_atoms):
                u = current_perm[r_i]
                v = current_perm[c_i]
                flat_d.append(dmat_rounded[u, v])

        # Is this the lexicographically smallest matrix?
        if best_d_string is None or flat_d < best_d_string:
            best_d_string = flat_d
            best_perm = current_perm

    return np.array(best_perm, dtype=int)


# ============================================================
# 3. Core Logic (NumPy)
# ============================================================


def _calculate_chiral_sign(coords: np.ndarray) -> str:
    """
    Robust geometric chirality check.
    Finds the first set of 4 atoms in the canonical list that are
    not coplanar and calculates their signed simplex volume.

    Returns:
        "+", "-", or "0" (if molecule is totally flat/linear)
    """
    n = coords.shape[0]
    if n < 4:
        return "0"

    # We need 4 points.
    # To keep it deterministic and efficient, we lock the first few
    # atoms and scan for the outliers that define 3D structure.

    # 1. Atom A is always index 0
    a = coords[0]

    # 2. Find Atom B (Index 1 usually, unless 0 and 1 are overlapping?)
    # assuming distinct coordi0tes for distinct atoms:
    b = coords[1]
    v_ab = b - a

    # 3. Find Atom C: The first atom not collinear with A-B
    c_idx = -1
    v_ac = None
    cross_ab_ac = None

    for i in range(2, n):
        c = coords[i]
        v_ac_temp = c - a
        # Cross product to check collinearity
        cp = np.cross(v_ab, v_ac_temp)
        if np.dot(cp, cp) > 1e-6:  # Not collinear
            c_idx = i
            v_ac = v_ac_temp
            cross_ab_ac = cp
            break

    if c_idx == -1:
        # All atoms are collinear (linear molecule)
        return "0"

    # 4. Find Atom D: The first atom not coplanar with A-B-C
    # We continue searching from c_idx + 1
    for i in range(c_idx + 1, n):
        d = coords[i]
        v_ad = d - a

        # Scalar triple product: dot(cross(AB, AC), AD)
        vol = np.dot(cross_ab_ac, v_ad)

        if abs(vol) > 1e-6:
            # Found a non-planar quartet!
            return "+" if vol > 0 else "-"

    # If we get here, the entire molecule is essentially flat (planar).
    # Planar molecules are usually achiral (superimposable on mirror image),
    # so "0" is the correct tag.
    return "0"


def _is_achiral_via_kabsch(coords: np.ndarray, precision: float) -> bool:
    """
    Determines if a molecule is achiral (superimposable on its mirror image).
    Uses the Kabsch algorithm to align the molecule with its reflection.
    """
    # 1. Centering
    center = coords.mean(axis=0)
    p = coords - center

    # 2. Create Mirror Image (Reflect over X)
    q = p.copy()
    q[:, 0] *= -1

    # 3. Kabsch Algorithm (Minimize RMSD between P and Q)
    #    H = P^T @ Q
    #    Since atom order is canonical and fixed, we assume 1-to-1 mapping.
    H = np.dot(p.T, q)
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix R = V^T @ U^T
    R = np.dot(Vt.T, U.T)

    # Ensure R is a proper rotation (det = 1), not reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 4. Calculate RMSD
    #    rotated_Q = Q @ R
    q_rot = np.dot(q, R)
    diff = p - q_rot
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # If the mirror image aligns perfectly with the original, it is Achiral.
    # We use a slightly looser tolerance than the hashing precision to be safe.
    return rmsd < (precision * 2.0)


def _calculate_robust_chiral_sign(coords: np.ndarray) -> str:
    """
    Finds the 'Handedness' of the molecule using the Most Voluminous Quartet.
    This avoids noise issues with 'almost planar' sub-structures.
    """
    n = coords.shape[0]
    if n < 4:
        return "NA"

    # We need to find 4 atoms (i,j,k,l) that maximize the volume of the tetrahedron.
    # Doing this exhaustively is O(N^4). Too slow.
    # Heuristic: Use the Principal Axes (Inertia) to find extremum points.

    # 1. Center and Align to Principal Axes
    center = coords.mean(axis=0)
    centered = coords - center
    inertia = np.dot(centered.T, centered)
    evals, evecs = np.linalg.eigh(inertia)

    # Project atoms onto the 3 axes
    proj = np.dot(centered, evecs)

    # 2. Pick extreme atoms on these axes
    # These are likely to define the 'hull' of the molecule
    indices = set()
    for dim in range(3):
        indices.add(np.argmin(proj[:, dim]))
        indices.add(np.argmax(proj[:, dim]))

    candidates = list(indices)

    # If we have fewer than 4 distinct extremes, just take the first few canonical atoms
    if len(candidates) < 4:
        candidates = list(range(min(n, 10)))  # fallback to checking first 10

    # 3. Check combinations of these candidates for max volume
    max_vol = -1.0
    best_sign = 0

    from itertools import combinations

    for subset in combinations(candidates, 4):
        # Calculate volume
        a, b, c, d = (
            coords[subset[0]],
            coords[subset[1]],
            coords[subset[2]],
            coords[subset[3]],
        )
        vol = np.dot(b - a, np.cross(c - a, d - a))

        abs_vol = abs(vol)
        if abs_vol > max_vol:
            max_vol = abs_vol
            best_sign = np.sign(vol)

    if max_vol < 1e-5:
        return "NA"  # Effectively planar

    return "POS" if best_sign > 0 else "NEG"


# Fails for edge test 2 (meso compound)
def generate_hashmol3d_v4(
    atomic_nums: np.ndarray,
    coords: np.ndarray,
    precision: float = 1e-4,
    charge: int = 0,
    multiplicity: Optional[int] = None,
    hash_length: int = 32,
    protocol: str = "v1-GEO-SHA256",
) -> HashMol3DResult:

    # 1. Setup
    n_atoms = len(atomic_nums)
    decimals = int(max(0, round(-np.log10(precision))))

    # 2. Canonicalization (Pure Geometry)
    # Note: We pass decimals to ensure sorting is robust to noise
    order = _get_geometric_canonical_order(atomic_nums, coords, decimals)

    # Reorder inputs
    z_sorted = atomic_nums[order]
    r_sorted = coords[order]

    # 3. Distance Matrix Calculation
    diff = r_sorted[:, None, :] - r_sorted[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)

    # Extract Upper Triangle
    iu = np.triu_indices(n_atoms, k=1)
    dvals = np.round(dmat[iu], decimals=decimals)

    # 4. Inferences
    # Multiplicity inference (simple electron count)
    if multiplicity is None:
        electrons = np.sum(z_sorted) - charge
        used_mult = 1 if (electrons % 2 == 0) else 2
    else:
        used_mult = int(multiplicity)

    # Chiral Sign
    chiral_sign = _calculate_chiral_sign(r_sorted)

    # 5. String Formatting (Strict)
    prec_str = "{:.1e}".format(precision)
    z_part = ",".join(str(z) for z in z_sorted)
    dist_fmt = f"{{:.{decimals}f}}"
    d_part = ",".join(dist_fmt.format(x) for x in dvals)

    components = [
        f"V:{protocol}",
        f"P:{prec_str}",
        f"Z:{z_part}",
        f"D:{d_part}",
        f"Q:{charge}",
        f"M:{used_mult}",
        f"S:{chiral_sign}",
    ]
    descriptor = "|".join(components)

    # 6. Hashing
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()

    return HashMol3DResult(
        hash_str=digest[:hash_length],
        protocol=protocol,
        precision=precision,
        charge=charge,
        multiplicity=used_mult,
        chiral_sign=chiral_sign,
        descriptor=descriptor,
    )


def generate_hashmol3d(
    atomic_nums: np.ndarray,
    coords: np.ndarray,
    precision: float = 1e-4,
    charge: int = 0,
    multiplicity: Optional[int] = None,
    hash_length: int = 32,
    protocol: str = "v1-GEO-KABSCH-SHA256",  # Updated Protocol Name
) -> HashMol3DResult:

    n_atoms = len(atomic_nums)
    decimals = int(max(0, round(-np.log10(precision))))

    # 1. Canonical Ordering (Greedy Geodesic)
    order = _get_geometric_canonical_order(atomic_nums, coords, decimals)
    z_sorted = atomic_nums[order]
    r_sorted = coords[order]

    # 2. Distance Matrix
    diff = r_sorted[:, None, :] - r_sorted[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n_atoms, k=1)
    dvals = np.round(dmat[iu], decimals=decimals)

    # 3. Chiral Flag Logic
    # A. First, check if physically achiral (Meso or Planar)
    if _is_achiral_via_kabsch(r_sorted, precision):
        chiral_sign = "NA"
    else:
        # B. If chiral, determine sign using the "biggest" feature
        chiral_sign = _calculate_robust_chiral_sign(r_sorted)

    # 4. Inferences
    if multiplicity is None:
        electrons = np.sum(z_sorted) - charge
        used_mult = 1 if (electrons % 2 == 0) else 2
    else:
        used_mult = int(multiplicity)

    # 5. Format & Hash
    prec_str = "{:.1e}".format(precision)
    z_part = ",".join(str(z) for z in z_sorted)
    dist_fmt = f"{{:.{decimals}f}}"
    d_part = ",".join(dist_fmt.format(x) for x in dvals)

    components = [
        f"V:{protocol}",
        f"P:{prec_str}",
        f"Z:{z_part}",
        f"D:{d_part}",
        f"Q:{charge}",
        f"M:{used_mult}",
        f"S:{chiral_sign}",
    ]
    descriptor = "|".join(components)
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()

    return HashMol3DResult(
        hash_str=digest[:hash_length],
        protocol=protocol,
        precision=precision,
        charge=charge,
        multiplicity=used_mult,
        descriptor=descriptor,
        chiral_sign=chiral_sign,
    )
