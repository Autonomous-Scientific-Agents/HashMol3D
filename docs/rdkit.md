# Optional RDKit input

Install the `rdkit` extra from a source checkout with
`pip install -e '.[rdkit]'`. The extra requires RDKit 2023.9.6 or newer; pip
selects a release compatible with your Python version. RDKit 2023.9.x needs
NumPy < 2.
Importing HashMol3D does not import RDKit. `hash_molecule`, the default
`hash_xyz`, and the default CLI retain their NumPy-only behavior whether RDKit
is installed or absent.

RDKit is only a structure reader. Every identifier is the geometry-only
identifier of `hash_molecule`, computed from the atomic numbers and 3D
coordinates that were read, whatever the input format. No connectivity,
stereo label, isotope, or SMILES enters the identifier.

## Input adapters

```python
from hashmol3d import hash_file, hash_rdkit, read_rdkit

mol = read_rdkit("molecule.sdf")
result = hash_rdkit(mol)
same = hash_file("molecule.sdf", input_format="sdf")
```

`read_rdkit(path, *, input_format=None)` returns an RDKit Mol. Omitted format
is inferred from the extension. Supported formats are `mol`, `sdf`, `mol2`,
`pdb`, `smi`/`smiles`, and `inchi`. Parsers sanitize molecules and preserve
explicit hydrogen atoms. SMILES/InChI files must contain exactly one nonblank
line; a SMILES name after whitespace is allowed. CXSMILES text is rejected.
SDF must contain exactly one `$$$$`-terminated record, including when later
records are invalid; trailing whitespace after its terminator is accepted.
MOL, MOL2 and PDB likewise reject additional molecule blocks; a PDB file
ends at its first `END` record. An SDF record cannot
hide a second MOL block before its terminator: text after `M  END` must be
ordinary SD property data introduced by a `>` header. Property values may
contain literal `M  END` text. A plain MOL block permits only trailing whitespace.
Multiple conformers in a file are rejected. To process a collection, iterate
an RDKit supplier explicitly:

```python
from rdkit import Chem
from hashmol3d import hash_rdkit

for mol in Chem.SDMolSupplier("collection.sdf", removeHs=False):
    if mol is None:
        raise ValueError("Invalid SDF record")
    print(hash_rdkit(mol))
```

`hash_rdkit(mol, *, conf_id=-1, charge=None, allow_implicit_hydrogens=False,
**kwargs)` accepts a Mol from any RDKit reader. It hashes the selected existing
3D conformer (`-1` selects RDKit's first conformer). It rejects missing/2D
conformers and never changes the caller's molecule. Coordinates, atoms, and
explicit H atoms are preserved exactly. By default, implicit hydrogens and
atom-level H counts without coordinates are rejected, including bracket counts
such as `[NH4+]`. Provide a complete explicit-H geometry, or explicitly pass
`allow_implicit_hydrogens=True` (CLI: `--allow-implicit-hydrogens`) to accept
incomplete geometry: missing H atoms then do not enter the geometry or
formula (an ethanol heavy-atom geometry has formula `C2O`), and no H
coordinates are invented. Unsanitized caller-built molecules have their
property cache initialized on the copy before hydrogen counts are checked;
this does not repair chemistry. Charge defaults to the graph's total formal
charge and, like multiplicity, appears only in the identifier's state tag.
Multiplicity retains the core electron-parity default based on the atoms
actually present, and can be overridden. Other keywords are passed to
`hash_molecule`, including `method`, `precision`, `length`, and `node_budget`.

`hash_file(path, *, input_format="xyz", generate_coordinates=False,
allow_implicit_hydrogens=False, **kwargs)` deliberately defaults to the native
XYZ parser, without extension-based dispatch. As in `hash_rdkit`,
`charge=None` selects the default charge. Choose other formats explicitly.
PDB hashing uses exactly the atoms in the file, without the implicit-H guard:
guessed single bonds can imply extra hydrogens even in complete unsaturated
geometries. Missing atoms cannot be inferred reliably from PDB chemistry, so
the caller must choose the intended atom set. Direct `hash_rdkit` calls retain
the guard because the caller controls the graph. MOL2 support also depends on
atom typing. These adapters do not repair missing chemical information. All
coordinates are interpreted as angstroms. The implicit-H override is rejected
for XYZ, whose atom list carries no implicit-H information.

For inputs without 3D geometry, `generate_coordinates=True` explicitly adds
hydrogens and replaces all conformers with one ETKDGv3 embedding (random seed
0, one thread, no energy optimization). This is unavailable for XYZ and PDB
file inputs; inferred PDB bond orders can produce incorrect hydrogen counts.
A fixed seed does **not** make conformer generation canonical:
equivalent SMILES with different atom orders or different RDKit releases can
produce different geometries. Save the generated geometry for reuse when
reproducibility matters. Embedding failure raises an error.

## Historical S descriptors

Versions 8–10 had an opt-in `S` extension that appended canonical isomeric
SMILES to the hashed descriptor
(`V:<base>-S1-RDKIT-<rdkitVersion>|P|Z|<F or C>|S:<SMILES>`). Version 11
removed it: the geometry hash covers only atomic numbers and geometry, and
`include_smiles` / `--include-smiles` no longer exist. Stored `S` descriptors
of versions 8–10 still re-hash to their original digests with
`hash_descriptor(descriptor, length=len(geometry_hash))`; the SMILES is not
interpreted. To keep chemistry alongside an identifier, store it in a
separate field.

## Canonical SMILES

```python
from rdkit import Chem
from hashmol3d import canonical_smiles

canonical_smiles(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
# 'F[C@](Cl)(Br)I'
```

`canonical_smiles(mol, *, conf_id=None)` is a standalone utility; its result
is not part of any identifier. It copies the graph, rejects query graphs and
enhanced stereo groups, removes atom-map numbers, sanitizes, and serializes
with `MolToSmiles(canonical=True, isomericSmiles=True)` after removing ordinary
explicit H atoms. It uses supplied graph stereochemistry when `conf_id` is
omitted, so it works without coordinates; passing a conformer ID derives
stereo from that 3D conformer with `AssignStereochemistryFrom3D`. It returns a
string and never mutates its input.

The adapter is implemented in `hashmol3d.rdkit_support`; the public functions
remain available from `hashmol3d`. It trusts the conformer's 3D flag, because
planar 3D geometries are legitimate and cannot be distinguished from mislabeled
2D drawings using a zero z coordinate alone. PDB has no 2D form, so PDB
conformers are always treated as 3D, including planar ones. RDKit
diagnostics remain visible; the adapter does not change process-wide logging
settings.

The chemistry operations follow RDKit's
[SMILES and input documentation](https://www.rdkit.org/docs/GettingStartedInPython.html)
and [stereochemistry documentation](https://www.rdkit.org/docs/RDKit_Book.html).
