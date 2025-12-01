# HashMol3D Specification v0.1.0

**Status:** Draft standard  
**Canonical algorithm:** SHA-256  
**Canonical output length:** 32 hex characters (128 bits)  

HashMol3D is a deterministic, rotation-, translation-, and permutation-invariant
identifier for 3D molecular conformers. It is designed for reproducible
identification of geometries in computational chemistry and scientific workflows.

HashMol3D encodes:

- Atomic numbers (Z) in a canonical atom order
- 3D Cartesian geometry via a pairwise distance matrix
- Distance rounding at user-specified precision
- Stereochemistry (R/S/?) from RDKit chiral center perception
- Formal charge
- Spin multiplicity
- A version tag specifying descriptor and hash algorithm

The canonical HashMol3D identifier is a fixed-length hexadecimal string

    <hash>

obtained by truncating the SHA-256 digest of a canonical descriptor string.

---

## 1. Inputs

HashMol3D requires:

1. An RDKit `Mol` object with at least one 3D conformer.
2. A distance precision epsilon in Å (default: 1e-4).
3. Optionally, a user-specified formal charge.
4. Optionally, a user-specified spin multiplicity.

If charge and multiplicity are not given, they are inferred as:

- `charge`: RDKit formal charge (fallback 0 if unavailable)
- `multiplicity`:

      number of electrons = sum(Z) - charge
      multiplicity = 1 if number of electrons is even else 2

No isotope information is included in HashMol3D, but it is possible to extend the specification as described in section 11.

---

## 2. Invariance Guarantees

HashMol3D is invariant under:

- Rigid translations of the coordinates
- Rigid rotations of the molecule
- Permutations of atom indices (via canonical atom ordering)
- Small numerical noise in coordinates, controlled by the rounding precision

It is **not** invariant under:

- Changes in atomic number Z
- Changes in overall charge
- Changes in multiplicity
- Conformer changes larger than the chosen precision threshold
- Inversion of stereochemistry (R ↔ S) for one or more chiral centers

---

## 3. Canonical Atom Ordering

Permutation invariance is achieved by a canonical atom ordering:

1. Use `Chem.CanonicalRankAtoms(mol)` to obtain an integer rank for each atom.
2. Sort atoms by `(rank, atom_index)` ascending.
3. Reorder both coordinates and atomic numbers according to this order.

This guarantees that the descriptor does not depend on input atom numbering.

---

## 4. Distance Matrix Construction

Given canonicalized coordinates \( r_i \in \mathbb{R}^3 \), construct the full
interatomic distance matrix:

\[
D_{ij} = \lVert r_i - r_j \rVert
\]

Only the strict upper-triangular entries (i < j) are used in the descriptor.

---

## 5. Precision and Rounding

Let epsilon be the distance precision, in Å (e.g., 1e-4). The number of
decimal places to keep is:

    decimals = max(0, round(-log10(epsilon)))

Distances in the upper triangle are rounded to `decimals` decimal places:

\[
D_{ij}^{\text{round}} = \text{round}(D_{ij}, \text{decimals})
\]

This controls how tolerant HashMol3D is to small numerical changes in geometry.

Default:

    epsilon = 1e-4 Å  →  decimals = 4

---

## 6. Stereochemistry Encoding

Topological stereochemistry is encoded using RDKit’s chiral center perception:

1. Use `Chem.FindMolChiralCenters(mol, includeUnassigned=True)` to obtain chiral centers.
2. Sort the centers by atom index.
3. For each center:

   - Use `"R"` or `"S"` if RDKit assigns an R/S configuration.
   - Use `"?"` for unassigned or unknown chirality.

4. Join the sequence with commas, e.g.:

   - No chiral centers: empty string `""`
   - One R center: `"R"`
   - Two centers, R then S: `"R,S"`
   - Center 1 unknown, center 4 S: `"?,S"`

This stereochemistry string is included in the descriptor but not printed
in the final hash.

---

## 7. Charge and Multiplicity

### 7.1 Charge

The formal charge is either:

- Provided explicitly by the user, **or**
- Inferred via `rdMolOps.GetFormalCharge(mol)` if possible, otherwise 0.

It is stored as an integer, e.g., `-1`, `0`, `+1`.

### 7.2 Multiplicity

The spin multiplicity is either:

- Provided explicitly by the user, **or**
- Inferred from electron count:

      electrons = sum(Z_i) - charge
      multiplicity = 1 if electrons % 2 == 0 else 2

This simple rule is appropriate for default behavior. Users with more detailed
spin information should pass the multiplicity explicitly.

---

## 8. Descriptor String

The descriptor is a UTF‑8 string that concatenates all components using
pipe (`|`) separators in a fixed order:

1. Version tag
2. Precision
3. Atomic numbers (canonical order)
4. Distance matrix (upper-triangular, rounded)
5. Stereochemistry
6. Charge
7. Multiplicity

The general format is:

    V:<version>|PREC:<epsilon>|Z:z1,z2,...,zN|D:d12,d13,...,d(N-1)N|STEREO:<stereo>|CHARGE:<q>|MULT:<m>

Where:

- `<version>` is a string, e.g., `2-SHA256`.
- `<epsilon>` is printed in scientific notation (e.g., `1.0e-04`).
- `z_i` are atomic numbers in canonical order.
- `d_ij` are rounded distances formatted with fixed decimal places.
- `<stereo>` is the stereochemistry string.
- `<q>` is the formal charge.
- `<m>` is the multiplicity.

Example (truncated):

    V:2-SHA256|PREC:1.0e-04|Z:6,6,8,1,1,1|
    D:1.0900,1.3400,1.8221,...|
    STEREO:R,S|CHARGE:0|MULT:1

This string is what is passed to the SHA-256 algorithm.

---

## 9. Hashing

HashMol3D uses SHA-256 as its canonical hash function.

1. Encode the descriptor as UTF‑8 bytes.
2. Compute the SHA-256 digest.
3. Convert to hexadecimal using `.hexdigest()`.
4. Truncate to a desired length L in hex characters, where

   - L ∈ {16, 32, 64} is permitted.
   - L = 32 (128-bit output) is the recommended canonical form.

Formally:

    hash = SHA256(descriptor_bytes).hexdigest()[:L]

This resulting hex string is the HashMol3D identifier.

---

## 10. Recommended Usage Modes

- **Canonical mode**: SHA-256, L = 32 hex chars, epsilon = 1e-4 Å.
- **Short mode**: SHA-256, L = 16 hex chars (64 bits) – suitable for small datasets.
- **Archival mode**: SHA-256, L = 64 hex chars (full 256 bits).

Implementations should clearly document which mode they are using and should
treat mode changes as semantically different identifiers.

---

## 11. Isotope Considerations (Out of Scope for v0.1.0)

In HashMol3D v0.1.0, isotopes are explicitly **ignored**:

- Only atomic numbers (Z) are considered.
- No isotope mass numbers appear in the descriptor.

This matches the goal of HashMol3D as an identifier for electronic-structure calculations, where isotopes do not change the underlying electronic Hamiltonian.

One can define an optional isotope-aware extension by adding an `ISO:` field:

    ISO:a1,a2,...,aN

where `a_i` are mass numbers, and changing the version tag accordingly.

---

## 12. Determinism and Portability

To guarantee identical hashes across machines and platforms, implementations
must:

- Use the same version tag (e.g., `2-SHA256`).
- Use the same precision epsilon.
- Use the same RDKit canonical ranking and stereochemistry perception.
- Use the exact descriptor formatting rules specified above.
- Encode strings in UTF‑8 before hashing.
- Use SHA-256 as defined in FIPS 180-4.

Any change to the descriptor format or semantics requires a new version tag.

---

## 13. Reference Implementation

The reference implementation is provided as a Python package `hashmol3d`,
which uses:

- RDKit for molecule I/O, canonicalization, and stereochemistry.
- NumPy for distance matrix computation.
- Python's `hashlib` for SHA-256 hashing.

The primary API functions are:

- `generate_hashmol3d(mol, ...)`
- `generate_hashmol3d_from_file(path, ...)`

These functions are considered normative for behavior in descriptor version `2-SHA256`.
