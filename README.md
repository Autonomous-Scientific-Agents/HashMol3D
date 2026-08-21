# HashMol3D

**HashMol3D** is a standard, deterministic 3D molecular geometry identifier
for computational chemistry, machine learning, and HPC workflows.

It produces a **readable** identifier of the form

    <Hill formula><state tag>-<geometry hash>

e.g. `H2Oq0m1-a4ba9da41d888939961ef77dae43b297` for neutral singlet water. The trailing
geometry hash is **rotation-, translation-, permutation-, and
parity-invariant** (matching the invariances of the eigenvalues of the
non-relativistic molecular Hamiltonian), and depends on:

- atomic numbers
- pairwise distances rounded to a user-specified precision
- a descriptor version tag

The hash is built from the element-labeled distance matrix written in a
**canonical atom order** (Weisfeiler-Leman refinement plus an
individualization-refinement search), which makes it a **complete**
congruence invariant: two geometries share a geometry hash only if they
are actually congruent at the chosen precision. In particular,
*homometric* structures — distinct geometries that share the same
distance multiset and collided in descriptor versions ≤ 4 — receive
distinct hashes.

Charge and spin multiplicity live in the readable prefix, **not** in
the hash, so two states of the same geometry share the same hex tail
and can be grouped by suffix matching:

```text
H2Oq0m1-a4ba9da41d888939961ef77dae43b297     # neutral singlet water
H2Oq1m2-a4ba9da41d888939961ef77dae43b297     # water cation, same geometry → same hex tail
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
hash_length_for(10**9)            # -> 23 hex chars for 1e9 items at p=1e-9
hash_molecule(z, coords, length=hash_length_for(10**9))
```

| distinct geometries | `p=1e-6` | `p=1e-9` | `p=1e-12` |
|--------------------:|:--------:|:--------:|:---------:|
| 10⁶                 | 15       | 18       | 20        |
| 10⁹                 | 20       | 23       | 25        |
| 10¹²                | 25       | 28       | 30        |
| 10¹⁵                | 30       | 33       | 35        |

(recommended hex length; the default of 32 covers up to ~8·10¹⁴ items at `p=1e-9`.)

It deliberately does **not** distinguish enantiomers (which share their
Hamiltonian eigenvalues). The reference implementation depends only on
NumPy.

## Fast O(N) mode for large systems

For proteins and other large systems where the O(N²) distance matrix is
prohibitive, `method="frame"` hashes coordinates in the principal-axes
frame of the Z-weighted gyration tensor — O(N log N) time, O(N) memory
(a 100,000-atom system hashes in ~0.3 s where the default method would
need an 80 GB matrix):

```python
hash_molecule(z, coords, method="frame")
```

The frame is reliable only when the principal moments are well separated.
If they are degenerate or nearly so (symmetric tops, linear molecules —
detected by a normative relative eigenvalue-gap threshold of 0.05), a
`UserWarning` is emitted and the call falls back to the canonical method;
the path taken is visible in `result.descriptor` (`F:` vs `C:`/`W:`).
Hashes from different methods are **not comparable** — pick one method
per corpus. Typical proteins and other asymmetric structures pass the
check; ideal symmetric molecules do not (and are exactly the cases the
default method handles).

HashMol3D IDs are **stable across machines**, **reproducible**, and ideal for:
- workflow deduplication  
- caching  
- large QC datasets  
- MD conformer tracking  
- ML potential datasets  
- LLM scientific agents  

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
H2Oq0m1-a4ba9da41d888939961ef77dae43b297

# Cation with explicit multiplicity — only the prefix changes.
$ hashmol3d -c 1 -m 2 water.xyz
H2Oq1m2-a4ba9da41d888939961ef77dae43b297

# Pin a fixed hash length and a coarser precision.
$ hashmol3d -p 1e-3 -l 32 benzene.xyz

# Verbose: also print formula, geometry hash, descriptor, and metadata.
$ hashmol3d -v water.xyz

# Show the package version.
$ hashmol3d --version
```

Short flags: `-p/--precision`, `-c/--charge`, `-m/--multiplicity`,
`-l/--length`, `-v/--verbose`. Errors on missing or malformed input go
to stderr with exit code 1 (no Python traceback).

## Usage (Python)

```python
import numpy as np
from hashmol3d import hash_molecule

atomic_nums = np.array([8, 1, 1])
coords = np.array([
    [ 0.0000, 0.0000, 0.0],
    [ 0.7572, 0.5860, 0.0],
    [-0.7572, 0.5860, 0.0],
])
res = hash_molecule(atomic_nums, coords)
print(res.hash_str)        # H2Oq0m1-a4ba9da41d888939961ef77dae43b297
print(res.formula)         # H2O
print(res.geometry_hash)   # a4ba9da41d888939961ef77dae43b297
print(res.charge, res.multiplicity)  # 0 1
```

All optional arguments are keyword-only: `precision`, `charge`,
`multiplicity`, `length`.

Or read straight from a file:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)        # H2Oq0m1-a4ba9da41d888939961ef77dae43b297
print(hash_xyz("water.xyz", charge=1, multiplicity=2).hash_str)
# H2Oq1m2-a4ba9da41d888939961ef77dae43b297
```

See [`docs/`](docs/) for the full
[specification](docs/specification.md),
[API reference](docs/api_reference.md), and
[CLI guide](docs/cli_usage.md).

