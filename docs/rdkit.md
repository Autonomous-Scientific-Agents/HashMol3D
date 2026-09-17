# Optional RDKit input and S tags

Install with `pip install 'hashmol3d[rdkit]'`. The extra requires RDKit
2023.9.6 or newer; pip selects a release compatible with your Python version.
Importing HashMol3D does not import RDKit. `hash_molecule`, the default
`hash_xyz`, and the default CLI retain their existing NumPy-only behavior,
descriptor version, and hashes whether RDKit is installed or absent.

## Input adapters

```python
from hashmol3d import hash_file, hash_rdkit, read_rdkit

mol = read_rdkit("molecule.sdf")
geometry = hash_rdkit(mol)
tagged = hash_rdkit(mol, conf_id=0, include_smiles=True)
same = hash_file("molecule.sdf", input_format="sdf", include_smiles=True)
```

`read_rdkit(path, *, input_format=None)` returns an RDKit Mol. Omitted format
is inferred from the extension. Supported formats are `mol`, `sdf`, `mol2`,
`pdb`, `smi`/`smiles`, and `inchi`. Parsers sanitize molecules and preserve
explicit hydrogen atoms. SMILES/InChI files must contain exactly one nonblank
line; a SMILES name after whitespace is allowed. CXSMILES text is rejected.
SDF must contain exactly one record, including when later records are invalid;
trailing whitespace after its `$$$$` terminator is accepted.
Multiple conformers in a file are rejected. To process a collection, iterate
an RDKit supplier explicitly:

```python
from rdkit import Chem
from hashmol3d import hash_rdkit

for mol in Chem.SDMolSupplier("collection.sdf", removeHs=False):
    if mol is None:
        raise ValueError("Invalid SDF record")
    print(hash_rdkit(mol, include_smiles=True))
```

`hash_rdkit(mol, *, conf_id=-1, include_smiles=False, charge=None,
allow_implicit_hydrogens=False, **kwargs)`
accepts a Mol from any RDKit reader. It hashes the selected existing 3D conformer
(`-1` selects RDKit's first conformer). It rejects missing/2D conformers and
never changes the caller's molecule. Coordinates, atoms, and explicit H atoms
are preserved exactly. By default, implicit hydrogens and atom-level H counts
without coordinates are rejected, including bracket counts such as `[NH4+]`.
Provide a complete explicit-H geometry, or explicitly pass
`allow_implicit_hydrogens=True` (CLI: `--allow-implicit-hydrogens`) to accept
incomplete geometry. With that override, missing H atoms do not enter the
geometry/formula even though they remain in S: an ethanol heavy-atom geometry
has formula `C2O` and `S:CCO`. No H coordinates are invented by the override.
Charge defaults to the graph's total formal charge;
multiplicity retains the core electron-parity default based on the atoms
actually present, and can be overridden. Other keywords are passed to
`hash_molecule`, including `method`, `precision`, `length`, and `node_budget`.

`hash_file(path, *, input_format="xyz", include_smiles=False,
generate_coordinates=False, allow_implicit_hydrogens=False, **kwargs)`
deliberately defaults to the native
XYZ parser, without extension-based dispatch. Choose other formats explicitly.
PDB file input is **refused for S tagging**, even with coordinate generation or
the implicit-H override. Proximity bonding cannot reliably recover bond orders
(for example, benzene can be misread as cyclohexane). Use an SDF with verified
bond orders, or `hash_rdkit` with a chemically prepared Mol. PDB geometry-only
hashing uses exactly the atoms in the file, without the implicit-H guard:
guessed single bonds can imply extra hydrogens even in complete unsaturated
geometries. Missing atoms cannot be inferred reliably from PDB chemistry, so
the caller must choose the intended atom set. Direct `hash_rdkit` calls retain
the guard because the caller controls the graph. MOL2 support
also depends on atom typing. These adapters do not repair missing chemical
information; validate the graph before chemistry-sensitive hashing. All
coordinates are interpreted as angstroms. The implicit-H override is rejected
for XYZ, whose atom list carries no implicit-H information.

For inputs without 3D geometry, `generate_coordinates=True` explicitly adds
hydrogens and replaces all conformers with one ETKDGv3 embedding (random seed
0, one thread, no energy optimization). This is available only for non-XYZ
file inputs. A fixed seed does **not** make conformer generation canonical:
equivalent SMILES with different atom orders or different RDKit releases can
produce different geometries. Save the generated geometry for reuse when
reproducibility matters. Embedding failure raises an error.

## S descriptor rules (extension revision 1)

`include_smiles=True` is the only switch that enables S. RDKit input alone
does not enable it. Both `F` and `C` geometry representations are supported:

```text
V:8-FRAME-SHA256-S1-RDKIT-<rdkitVersion>|P:<precision>|Z:<atoms>|F:<rows>|S:<SMILES>
V:8-FRAME-SHA256-S1-RDKIT-<rdkitVersion>|P:<precision>|Z:<atoms>|C:<distances>|S:<SMILES>
```

1. Compute the ordinary geometry descriptor with the unchanged core algorithm.
2. Copy the RDKit graph, reject query graphs and enhanced stereo groups, remove
   atom-map numbers, and sanitize with RDKit's default aromaticity model.
3. Assign stereochemistry from the selected **unrounded 3D conformer** using
   `AssignStereochemistryFrom3D(..., replaceExistingTags=True)`. Geometry is
   authoritative, including when it disagrees with pre-existing stereo labels.
4. Run `AssignStereochemistry(cleanIt=True, force=True)`, remove ordinary
   explicit H atoms with RDKit's default `RemoveHs`, and serialize with
   `MolToSmiles(canonical=True, isomericSmiles=True)`. Isotopic H atoms and other
   H atoms retained by RDKit's default removal policy remain in SMILES. Removal
   applies only to the SMILES copy, never to geometry or the readable formula.
5. Replace only the `V` field with the base version plus
   `-S1-RDKIT-<rdkitVersion>`; append `|S:` and the SMILES verbatim. The RDKit
   version uses URL percent encoding (unreserved characters unchanged), so
   custom version strings cannot introduce delimiters. Plain SMILES payloads
   containing pipes or whitespace are rejected.
6. Hash this entire UTF-8 descriptor with SHA-256 and the requested truncation.
   The existing `HashMol3DResult` fields and readable prefix layout are reused.

`hash_xyz(path, include_smiles=True, **kwargs)` uses the existing XYZ parser
(including its first-frame warning), then RDKit `DetermineBonds` with the
requested total charge (default zero) and `embedChiral=True`. Bond perception
requires complete atom lists, including hydrogens, and chemically plausible
geometry. It can fail or infer the wrong graph for unusual valence, metals,
radicals, or distorted structures. Validate perceived chemistry before using
such identifiers. A chemistry error or exhausted geometry search returns no
identifier; S is never silently omitted. Bond-perception failures include
HashMol3D context and preserve the original RDKit exception as their cause.
Multiplicity affects only the readable state prefix and does not control bond
perception; supplying a radical's spin state does not repair its bond ordering.

For RDKit graphs, a charge override must match the graph when S is requested.
Formal charges are in SMILES and therefore affect the S digest; the default
geometry-only guarantee of equal suffixes across charge states does not apply.
Spin multiplicity remains outside the descriptor. Isotopic labels in the graph
also enter S, while the default atomic-number-only descriptor ignores them.
XYZ has no isotope labels to preserve.

The S payload distinguishes connectivity isomers, supported E/Z double-bond
isomers, and supported tetrahedral enantiomers when stereochemistry is resolved.
It does not guarantee distinction for every stereochemical system: unresolved
or unsupported stereo cannot be recovered by canonical SMILES. Plain SMILES
cannot preserve enhanced AND/OR stereo semantics, so such groups are rejected.
Tautomers, protonation states, and salts are not standardized or merged.
Geometric reflection can change S; rotation, translation, and consistent atom
reordering do not intentionally change it. Stereo perception has its own
numerical boundaries independent of the geometry quantization grid.

RDKit canonicalization and perception may change across releases. S revision
and RDKit version are included in the hashed namespace, so use the same RDKit
version, graph preparation, hydrogen policy, and geometry method for a corpus.
Every RDKit upgrade intentionally changes every S identifier, even when the
SMILES is unchanged: hashing the version prevents silent mixing of identifiers
produced under different canonicalization/perception rules.
The `geometry_hash` field now holds a geometry-plus-chemistry digest for S
results. S results are not interchangeable with default geometry-only results.

## Canonical SMILES without hashing

```python
from rdkit import Chem
from hashmol3d import canonical_smiles

canonical_smiles(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
# 'F[C@](Cl)(Br)I'
```

`canonical_smiles(mol, *, conf_id=None)` uses supplied graph stereochemistry
when `conf_id` is omitted, so it works without coordinates. Passing a conformer
ID derives stereo from that 3D conformer as for S tagging. It returns a string
and never mutates its input.

The adapter is implemented in `hashmol3d.rdkit_support`; the public functions
remain available from `hashmol3d`. It trusts the conformer's 3D flag, because
planar 3D geometries are legitimate and cannot be distinguished from mislabeled
2D drawings using a zero z coordinate alone. RDKit diagnostics remain visible;
the adapter does not change process-wide logging settings.

The chemistry operations follow RDKit's
[SMILES and input documentation](https://www.rdkit.org/docs/GettingStartedInPython.html),
[bond perception API](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html),
and [stereochemistry documentation](https://www.rdkit.org/docs/RDKit_Book.html).
