# HashMol3D

**HashMol3D** is a standard, deterministic 3D molecular geometry identifier
for computational chemistry, machine learning, and HPC workflows.

It produces a **rotation-invariant, translation-invariant, atom-permutation-invariant**
hash string that uniquely identifies a **conformer**, including:

- exact 3D geometry (distance matrix)
- atomic numbers
- stereochemistry (R/S)
- charge
- spin multiplicity
- numerical precision used

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
# Compute hash for a molecule file
hashmol3d compute path/to/molecule.mol

# With options
hashmol3d compute --precision 1e-3 --hash-length 16 molecule.xyz

# Show version
hashmol3d version
```

