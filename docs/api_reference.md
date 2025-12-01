# HashMol3D API Reference

## `generate_hashmol3d(mol, conf_id=0, precision=1e-4, charge=None, multiplicity=None, hash_length=32, version="2-SHA256")`

Compute the HashMol3D identifier for an RDKit `Mol` with 3D coordinates.

**Parameters:**
- `mol`: RDKit `Mol` object with at least one 3D conformer
- `conf_id`: Conformer ID to use (default: 0)
- `precision`: Distance precision in Å (default: 1e-4)
- `charge`: Formal charge (default: None, inferred from molecule)
- `multiplicity`: Spin multiplicity (default: None, inferred from electron count)
- `hash_length`: Length of hash string in hex characters (default: 32)
- `version`: Descriptor version tag (default: "2-SHA256")

**Returns:** `HashMol3DResult` object with fields:

- `hash_str`: canonical HashMol3D ID (hex string)
- `version`: descriptor+hash version string
- `precision`: distance precision (Å)
- `charge`: charge used
- `multiplicity`: multiplicity used
- `formula`: RDKit molecular formula
- `canonical_smiles`: RDKit canonical SMILES
- `descriptor`: raw descriptor string (for debugging)

## `generate_hashmol3d_from_file(path, file_format=None, precision=1e-4, charge=None, multiplicity=None, hash_length=32, version="2-SHA256")`

High-level convenience function:

1. Loads a molecule from file using RDKit (XYZ, SDF, MOL, MOL2, PDB, SMILES).
2. Ensures a 3D conformer (embedding if needed).
3. Computes the HashMol3D identifier as above.

**Parameters:**
- `path`: Path to molecular file
- `file_format`: File format (default: None, inferred from extension)
- `precision`: Distance precision in Å (default: 1e-4)
- `charge`: Formal charge (default: None, inferred from molecule)
- `multiplicity`: Spin multiplicity (default: None, inferred from electron count)
- `hash_length`: Length of hash string in hex characters (default: 32)
- `version`: Descriptor version tag (default: "2-SHA256")

**Returns:** `HashMol3DResult` object.
