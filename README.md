# HashMol3D

**HashMol3D** is a standard, deterministic 3D molecular geometry identifier
for computational chemistry, machine learning, and HPC workflows.

It produces a **readable** identifier of the form

    <Hill formula><state tag>-<geometry hash>

e.g. `H2Oq0m1-68936c504bf5fa3b` for neutral singlet water. The trailing
geometry hash is **rotation-, translation-, permutation-, and
parity-invariant** (matching the invariances of the eigenvalues of the
non-relativistic molecular Hamiltonian), and depends on:

- atomic numbers
- pairwise distances rounded to a user-specified precision
- a descriptor version tag

Charge and spin multiplicity live in the readable prefix, **not** in
the hash, so two states of the same geometry share the same hex tail
and can be grouped by suffix matching:

```text
H2Oq0m1-68936c504bf5fa3b     # neutral singlet water
H2Oq+1m2-68936c504bf5fa3b    # water cation, same geometry → same hex tail
```

The hash length auto-scales with the number of atoms (`clip(N, 16, 64)`
hex chars) so collision risk stays roughly constant as molecules grow;
pass `length=` to pin a fixed value.

It deliberately does **not** distinguish enantiomers (which share their
Hamiltonian eigenvalues). The reference implementation depends only on
NumPy.

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
H2Oq0m1-68936c504bf5fa3b

# Cation with explicit multiplicity — only the prefix changes.
$ hashmol3d -c 1 -m 2 water.xyz
H2Oq+1m2-68936c504bf5fa3b

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
print(res.hash_str)        # H2Oq0m1-68936c504bf5fa3b
print(res.formula)         # H2O
print(res.geometry_hash)   # 68936c504bf5fa3b
print(res.charge, res.multiplicity)  # 0 1
```

All optional arguments are keyword-only: `precision`, `charge`,
`multiplicity`, `length`.

Or read straight from a file:

```python
from hashmol3d import hash_xyz

print(hash_xyz("water.xyz").hash_str)        # H2Oq0m1-68936c504bf5fa3b
print(hash_xyz("water.xyz", charge=1, multiplicity=2).hash_str)
# H2Oq+1m2-68936c504bf5fa3b
```

See [`docs/`](docs/) for the full
[specification](docs/specification.md),
[API reference](docs/api_reference.md), and
[CLI guide](docs/cli_usage.md).

