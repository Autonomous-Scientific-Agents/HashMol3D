# HashMol3D

**HashMol3D** is a Python library that implements deterministic 3D molecular
geometry identifiers and hashes for computational chemistry, machine learning,
and HPC workflows. The library provides the implementation used to develop and
define a proposed identifier standard. The accompanying paper describes the
software and methods and proposes that standard; the
[specification](docs/specification.md) is a draft, not an adopted standard.

It produces a **readable** identifier of the form

    <Hill formula><state tag>-<geometry hash>

e.g. `H2Oq0m1-b4db5388ff28342bdc809a83891e65ea` for neutral singlet water. The trailing
geometry hash is **rotation-, translation-, permutation-, and
parity-invariant** (matching the invariances of the eigenvalues of the
non-relativistic molecular Hamiltonian), and depends on:

- atomic numbers
- coordinates quantized at a user-specified precision in a canonical frame
- a descriptor version tag

The default method projects the original coordinates into a deterministic
canonical frame, sorts the element-labelled coordinate rows, and hashes that
complete representation. Ordinary geometries use the principal axes of the
Z-weighted gyration tensor. Point-like, linear, symmetric-top, and spherical-
top geometries use intrinsic coordinates or canonical atom anchors, so exact
symmetry does not require a random perturbation. The canonical labelled
distance-matrix method remains available as `method="canonical"`.

Charge and spin multiplicity live in the readable prefix, **not** in
the hash, so two states of the same geometry share the same hex tail
and can be grouped by suffix matching:

```text
H2Oq0m1-b4db5388ff28342bdc809a83891e65ea     # neutral singlet water
H2Oq1m2-b4db5388ff28342bdc809a83891e65ea     # water cation, same geometry → same hex tail
```

The geometry hash defaults to a fixed length of 32 hex chars (128 bits).
Collision resistance is governed by how many distinct geometries share a
namespace (birthday bound `~ n² / 2^{b+1}` for `b = 4·length` bits), **not**
by molecule size; the 128-bit default keeps the expected collision count below
one for corpora up to ~10¹⁶ geometries. Pass `length=` to pin any value in
`[1, 64]`, or call `hash_length_for(n_items, target_prob)` to size the hash to
your corpus:

```python
from hashmol3d import hash_length_for

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

The current format deliberately does **not** distinguish enantiomers, and
isotopes share the same atomic number and therefore the same identifier at
fixed coordinates and electronic state. The proposed standard can be extended
with new tags for enantiomer-sensitive stereochemistry and isotope labels.
Those extensions would require defined canonicalization rules and a new
descriptor version; they are not implemented in this release. The library
depends only on NumPy.

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
float64 roundoff, are recognized before the general ten-grid-unit size guard;
they do not need transverse anchors even on coarse grids. Nearly linear inputs
still undergo the conditioning checks. Canonically tied
anchors are all evaluated up to a 10,000-candidate budget. Ill-conditioned
or over-budget cases emit `UserWarning` and use the complete distance method.
The path is visible in `result.descriptor` (`F:` versus `C:`). Canonical
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

For an explicitly distance-based descriptor, use:

```python
hash_molecule(z, coords, method="canonical")
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

Version 0.10.0 uses descriptor `7-FRAME-SHA256` for this refined line handling.
Because the version field is hashed, all digests change from version 6; recompute
identifiers consistently when migrating a corpus. The paper's archived corpus
statistics describe version 6 and are labelled accordingly.

## Descriptor tags

The geometry hash is computed from a UTF-8 descriptor with fields in this order:

```text
V:<version>|P:<precision>|Z:<atomic numbers>|F:<coordinate rows>
V:<version>|P:<precision>|Z:<atomic numbers>|C:<distance matrix upper triangle>
```

- `V` identifies the descriptor format and canonicalization rules (currently
  `7-FRAME-SHA256`).
- `P` records the quantization grid spacing in angstroms, such as `1.0e-04`.
- `Z` lists atomic numbers in the order used by the selected representation.
- `F` stores the sorted element-labelled, quantized canonical-frame rows.
- `C` stores the quantized distance matrix upper triangle in canonical atom order.

All these fields enter the hash. The readable `q` and `m` tags instead record
charge and spin multiplicity outside the geometry hash.

## Install

### Using pip

```bash
pip install hashmol3d
```

### Using uv

```bash
# Install from PyPI
uv pip install hashmol3d
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/HashMol3D.git
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
```

## Usage (CLI)

```bash
$ hashmol3d water.xyz
H2Oq0m1-b4db5388ff28342bdc809a83891e65ea

# Cation with explicit multiplicity — only the prefix changes.
$ hashmol3d -c 1 -m 2 water.xyz
H2Oq1m2-b4db5388ff28342bdc809a83891e65ea

# Pin a fixed hash length and a coarser precision.
$ hashmol3d -p 1e-3 -l 32 benzene.xyz

# Verbose: also print formula, geometry hash, descriptor, and metadata.
$ hashmol3d -v water.xyz

# Request the labelled distance-matrix method explicitly.
$ hashmol3d --method canonical water.xyz

# Show the package version.
$ hashmol3d --version
```

Options include `-p/--precision`, `-c/--charge`, `-m/--multiplicity`,
`-l/--length`, `--method`, `--node-budget`, and `-v/--verbose`. Errors on missing or malformed input go
to stderr with exit code 1 (no Python traceback).

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
print(res.hash_str)  # H2Oq0m1-b4db5388ff28342bdc809a83891e65ea
print(res.formula)  # H2O
print(res.geometry_hash)  # b4db5388ff28342bdc809a83891e65ea
print(res.charge, res.multiplicity)  # 0 1
```

All optional arguments are keyword-only: `precision`, `charge`,
`multiplicity`, `length`, `method`, and `node_budget`.

`precision` must be a power of ten no greater than 1 Å (`1.0`, `1e-1`,
`1e-2`, ...). Restricting the grid to powers of ten keeps its descriptor
representation unambiguous and portable.

Or read straight from a file:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)  # H2Oq0m1-b4db5388ff28342bdc809a83891e65ea
print(hash_xyz("water.xyz", charge=1, multiplicity=2).hash_str)
# H2Oq1m2-b4db5388ff28342bdc809a83891e65ea
```

See [`docs/`](docs/) for the full
[specification](docs/specification.md),
[API reference](docs/api_reference.md), and
[CLI guide](docs/cli_usage.md).
