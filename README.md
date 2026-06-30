# HashMol3D

**HashMol3D** is a standard, deterministic 3D molecular geometry identifier
for computational chemistry, machine learning, and HPC workflows.

It produces a **rotation-, translation-, permutation-, and parity-invariant**
hash string that identifies a **conformer** with the same invariance
properties as the eigenvalues of the non-relativistic molecular
Hamiltonian. The descriptor encodes:

- atomic numbers
- pairwise distances rounded to a user-specified precision
- charge
- spin multiplicity
- a version tag

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
# Compute hash for an XYZ file
hashmol3d path/to/molecule.xyz

# With options (short flags)
hashmol3d -p 1e-3 -l 16 molecule.xyz

# Print the canonical descriptor along with the hash
hashmol3d -v molecule.xyz

# Show version
hashmol3d --version
```

## Usage (Python)

```python
import numpy as np
from hashmol3d import hash_molecule

atomic_nums = np.array([8, 1, 1])
coords = np.array([
    [0.0,    0.0,   0.0],
    [0.7572, 0.586, 0.0],
    [-0.7572,0.586, 0.0],
])
res = hash_molecule(atomic_nums, coords)
print(res.hash_str)
```

Or read straight from a file:

```python
from hashmol3d import hash_xyz

print(hash_xyz("molecule.xyz").hash_str)
```

