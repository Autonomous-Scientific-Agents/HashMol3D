# HashMol3D

**HashMol3D** is a Python library that implements deterministic 3D molecular
geometry identifiers and hashes for computational chemistry, machine learning,
and HPC workflows. The library provides the implementation used to develop and
define a proposed identifier standard. The accompanying paper describes the
software and methods and proposes that standard; the
[specification](docs/specification.md) is a draft, not an adopted standard.

It produces a file-name-safe identifier of the form

    hm3d_v<major version>_p<decimals>_<Hill formula>_<state tag>_<geometry hash>

e.g. `hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc` for neutral singlet water on the
default 10⁻⁴ Å grid (`p4`). The geometry hash is **rotation-, translation-,
permutation-, and parity-invariant** (matching the invariances of the
eigenvalues of the non-relativistic molecular Hamiltonian), and depends only
on:

- atomic numbers
- the precision of the quantization grid
- coordinates quantized in a canonical frame (or, with the distance method,
  canonical interatomic distances)

The descriptor version is not hashed; its major number is the `v11` field of
the identifier, so identifiers made with different versions never compare
equal. The precision is both hashed and shown as the `p4` field. Fields are joined by `_`, and every character is
a letter, a digit, `_` or `-` (in a negative charge), so identifiers are safe
as file names.

The default method projects the original coordinates into a deterministic
canonical frame, sorts the element-labelled coordinate rows, and hashes that
complete representation. Ordinary geometries use the principal axes of the
Z-weighted gyration tensor. Point-like, linear, symmetric-top, and spherical-
top geometries use intrinsic coordinates or canonical atom anchors, so exact
symmetry does not require a random perturbation. The canonical labelled
distance-matrix method remains available as `method="distance"`.

Charge and spin multiplicity live in the state tag, **not** in the
hash, so two states of the same geometry share the same geometry hash and can
be grouped by it:

```text
hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc     # neutral singlet water
hm3d_v11_p4_H2O_q1m2_8e9eedf5032cf387541d637f56c682dc     # water cation, same geometry → same hash
```

The geometry hash defaults to a fixed length of 32 hex chars (128 bits).
Collision resistance is governed by how many distinct geometries share a
namespace (birthday bound `~ n² / 2^{b+1}` for `b = 4·length` bits), **not**
by molecule size; the 128-bit default keeps the expected collision count below
one for corpora up to ~2.6·10¹⁹ geometries. Pass `length=` to pin any value in
`[1, 64]`, or call `hash_length_for(n_items, target_prob)` to size the hash to
your corpus:

```python
from hashmol3d import hash_length_for, hash_molecule

# z: atomic numbers, coords: (N, 3) Cartesian coordinates in Å
hash_length_for(10**9)  # -> 23 hex chars for 1e9 items at p=1e-9
hash_molecule(z, coords, length=hash_length_for(10**9))
```

| distinct geometries | `p=1e-6` | `p=1e-9` | `p=1e-12` |
|--------------------:|:--------:|:--------:|:---------:|
| 10⁶                 | 15       | 18       | 20        |
| 10⁹                 | 20       | 23       | 25        |
| 10¹²                | 25       | 28       | 30        |
| 10¹⁵                | 30       | 33       | 35        |

(recommended hex length; the default of 32 covers up to ~8·10¹⁴ items at `p=1e-9`.)

The identifier deliberately does **not** distinguish enantiomers, and isotopes
share the same atomic number and therefore the same identifier at fixed
coordinates and electronic state. It carries no connectivity information. The
library depends only on NumPy; the optional RDKit support only reads more file
formats and does not change any identifier.

## Canonical frame method

The default `method="frame"` normally takes O(N log N) time and O(N) memory
(a 100,000-atom asymmetric system hashes in ~0.3 s where a full distance
matrix would need 80 GB):

```python
hash_molecule(z, coords)  # method="frame" is the default
```

When principal moments are degenerate, the method preserves any isolated
axis and canonically anchors only the ambiguous subspace. Linear and point-
like systems are represented in their intrinsic dimension. Exact lines, up to
float64 roundoff, do not need transverse anchors even on coarse grids.
Well-separated principal moments also need no atom anchors: the frame is
accepted even when the centered radius is below ten grid units. Only
unresolved eigenspaces need an atom anchor, and any transverse displacement
of at least one grid unit qualifies (version 9; versions 6-8 required ten),
so nearly linear molecules with a resolved bend anchor their degenerate plane
rather than falling back. A bend below half a grid unit is an intrinsic line.
Canonically tied anchors are all evaluated up to a 10,000-candidate budget.
Clouds below ten grid units of radius with degenerate moments, and over-budget
cases, emit `UserWarning` and use the complete distance method.
The path is visible in `result.descriptor` (`F:` versus `D:`). Canonical
search must finish within `node_budget` (default 10,000 visited states),
including after frame fallback. If it exhausts that budget, Python raises
`SearchBudgetExceeded`; the CLI prints an error to stderr and exits with code
1. No descriptor, hash, or identifier is created. Increase the budget and retry:

```python
hash_molecule(z, coords, node_budget=100_000)
```

```bash
hashmol3d --node-budget 100000 molecule.xyz
```

A larger canonical node budget permits more work but does not change a
successfully completed descriptor. No weaker summary is substituted.

For an explicitly distance-based descriptor, use the distance method (`D:`
section; `method="canonical"` is a deprecated alias):

```python
hash_molecule(z, coords, method="distance")
```

Hashes from different descriptor paths are not comparable; use the same
method and descriptor version throughout a corpus.

With the stated [portability conditions](docs/specification.md#8-determinism-and-portability),
the identifiers support:
- workflow deduplication  
- caching  
- large QC datasets  
- MD conformer tracking  
- ML potential datasets  
- LLM scientific agents  

Version 0.15.0 uses descriptor version `11.0`. It introduces the `hm3d_`
identifier, stops hashing the version (it is in the identifier), and renames
the distance section from `C` to `D`. Version 10 made re-hashing idempotent
(see [Re-hashing a stored descriptor](#re-hashing-a-stored-descriptor)), and
version 9 let nearly linear molecules anchor their frame instead of using the
distance fallback. Every geometry hash changes from earlier versions;
recompute identifiers consistently when migrating a corpus. The paper's corpus
statistics describe version 8 and are labelled accordingly.

## Re-hashing a stored descriptor

Hashing the same input again always gives the same identifier. Version 10 also
makes the stored descriptor self-sufficient, so a corpus can keep descriptors
instead of source geometries and still re-derive every identifier:

```python
from hashmol3d import descriptor_geometry, hash_descriptor, hash_molecule

res = hash_molecule(z, coords)

# Exact, for F and D descriptors: no geometry is recomputed.
assert hash_descriptor(res.descriptor) == res.geometry_hash

# F descriptors also carry a canonical-frame geometry that hashes back to
# the same identifier, in any orientation or atom order.
z2, coords2 = descriptor_geometry(res.descriptor)
assert hash_molecule(z2, coords2).hash_str == res.hash_str
```

`F` rows are stored on a grid 10⁴ times finer than `precision`, and only their
values rounded back to `precision` are hashed, so the extra digits never make
the identifier more sensitive to noise. Before returning coordinates,
`descriptor_geometry` re-hashes them and, in the rare case that recomputing the
frame moves a value across a rounding boundary, settles them by a small
fraction of a grid cell. On all of QM9 at 10⁻², 10⁻³ and 10⁻⁴ Å every round
trip reproduced the identifier (version-9 rows read back as coordinates did so
for 54–56 %). Exactly symmetric inputs with tied anchor frames can occasionally
not be settled; `descriptor_geometry` then warns, and `hash_descriptor` remains
exact. `D` descriptors hold distances, not coordinates; keep their source
geometry if coordinates are needed. Stored descriptors of versions 5–10
(including their `C` and RDKit `S` forms) still re-hash to their original
digests.

## Descriptor tags

The stored descriptor (`result.descriptor`) is a UTF-8 string with fields in
this order:

```text
V:<version>|P:<precision>|Z:<atomic numbers>|F:<coordinate rows>
V:<version>|P:<precision>|Z:<atomic numbers>|D:<distance matrix upper triangle>
```

- `V` identifies the descriptor format and canonicalization rules (currently
  `11.0`).
- `P` records the quantization grid spacing in angstroms, such as `1.0e-04`.
- `Z` lists atomic numbers in the order used by the selected representation.
- `F` stores the sorted element-labelled canonical-frame rows on a grid 10⁴
  times finer than `P`.
- `D` stores the quantized distance matrix upper triangle in canonical atom
  order (the distance method; called `C` before version 11).

`P`, `Z` and the geometry section enter the hash, with `F` values rounded back
to the `P` grid: the geometry hash is the SHA-256 digest of
`P:<precision>|Z:<atomic numbers>|F:<hashed rows>` (or `|D:<distances>`). `V`
is not hashed; it appears in the identifier as `v11` (the major version), next
to `p<decimals>`, and the `q`/`m` state tag records charge and spin
multiplicity.

## Optional RDKit support

RDKit lets HashMol3D read more file formats. It never changes the identifier:
every input is hashed from its atomic numbers and 3D coordinates alone.
Version 11 removed the earlier opt-in `S` extension, which put canonical
SMILES into the hash; stored `S` descriptors still re-hash with
`hash_descriptor`.

```bash
pip install -e '.[rdkit]'  # from a source checkout; see Install

# Read an existing 3D structure.
hashmol3d molecule.sdf --input-format sdf

# A SMILES file has no geometry: explicitly generate a new conformer.
hashmol3d molecule.smi --input-format smi --generate-coordinates
```

```python
from hashmol3d import canonical_smiles, hash_rdkit

# mol can come from any RDKit reader; hashing requires a 3D conformer.
result = hash_rdkit(mol)
smiles = canonical_smiles(mol)  # a separate utility; not part of the identifier
```

MOL, SDF, MOL2, PDB, SMILES, and InChI file adapters are available. PDB input
hashes exactly the atoms in the file, without trusting guessed implicit-H
counts; coordinate generation is refused for PDB because guessed bond orders
can add incorrect hydrogens. Other RDKit inputs with hydrogens without coordinates are rejected
unless explicitly allowed with `allow_implicit_hydrogens=True`
(CLI: `--allow-implicit-hydrogens`); that override hashes only the atoms present.
Arbitrary RDKit readers can feed `hash_rdkit` directly. See the
[RDKit guide](docs/rdkit.md) before mixing sources.

## Install

HashMol3D needs Python ≥ 3.8 and NumPy; RDKit (2023.9.6 or newer) is an
optional extra. Install 0.15.0 from source: the release on PyPI is older
(0.9.x), uses an earlier descriptor version and therefore different
identifiers, and lacks `hash_descriptor`, `descriptor_geometry` and the RDKit
adapters.

### Install from source

```bash
# Clone the repository
git clone https://github.com/Autonomous-Scientific-Agents/HashMol3D.git
cd HashMol3D

# Create and activate a virtual environment (recommended)
# Using uv:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using standard Python:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode
uv pip install -e .  # Or: pip install -e .
# With the optional RDKit adapters:
uv pip install -e '.[rdkit]'  # Or: pip install -e '.[rdkit]'
```

## Usage (CLI)

```bash
$ hashmol3d water.xyz
hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc

# Cation with explicit multiplicity — only the state tag changes.
$ hashmol3d -c 1 -m 2 water.xyz
hm3d_v11_p4_H2O_q1m2_8e9eedf5032cf387541d637f56c682dc

# A coarser precision and a shorter, 20-hex-character hash.
$ hashmol3d -p 1e-3 -l 20 benzene.xyz

# Verbose: a labelled block with the identifier, formula, geometry hash,
# descriptor, and metadata instead of the bare identifier.
$ hashmol3d -v water.xyz

# Request the labelled distance-matrix method explicitly.
$ hashmol3d --method distance water.xyz

# Show the package version.
$ hashmol3d --version
```

Options include `-p/--precision`, `-c/--charge`, `-m/--multiplicity`,
`-l/--length`, `--method`, `--node-budget`, and `-v/--verbose`. Errors on
missing or malformed input go to stderr with exit code 1 (no Python
traceback); invalid command-line usage exits with code 2.

## Usage (Python)

```python
import numpy as np
from hashmol3d import hash_molecule

atomic_nums = np.array([8, 1, 1])
coords = np.array(
    [
        [0.0000, 0.0000, 0.0],
        [0.7572, 0.5860, 0.0],
        [-0.7572, 0.5860, 0.0],
    ]
)
res = hash_molecule(atomic_nums, coords)
print(res.hash_str)  # hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc
print(res.formula)  # H2O
print(res.geometry_hash)  # 8e9eedf5032cf387541d637f56c682dc
print(res.charge, res.multiplicity)  # 0 1
```

All optional arguments are keyword-only: `precision`, `charge`,
`multiplicity`, `length`, `method`, and `node_budget`.

`precision` must be a power of ten no greater than 1 Å (`1.0`, `1e-1`,
`1e-2`, ...). Restricting the grid to powers of ten keeps its descriptor
representation unambiguous and portable. It must also be resolvable in
float64: grids finer than about 3×10⁻¹¹ times the molecule's extent are
rejected with a `ValueError` naming the finest usable precision (e.g. 1e-10 Å
for water, 1e-8 Å for a 100 Å protein), because identifiers on such grids
change under rotation.

Or read straight from a file:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)  # hm3d_v11_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc
print(hash_xyz("water.xyz", charge=1, multiplicity=2).hash_str)
# hm3d_v11_p4_H2O_q1m2_8e9eedf5032cf387541d637f56c682dc
```

See [`docs/`](docs/) for the full
[specification](docs/specification.md),
[API reference](docs/api_reference.md), and
[CLI guide](docs/cli_usage.md).
