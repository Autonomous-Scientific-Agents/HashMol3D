"""Opt-in RDKit adapters; importing this module does not import RDKit.

RDKit only reads structures: the descriptor and hash are the geometry-only
ones of :func:`hash_molecule`, whatever the input format. RDKit molecules
are always copied before chemistry or coordinate operations.
"""

from __future__ import annotations

import io
from pathlib import Path

from .core import HashMol3DResult, _resolve_method, hash_molecule
from .io import read_xyz

__all__ = ["canonical_smiles", "hash_file", "hash_rdkit", "read_rdkit"]


def _require_rdkit():
    try:
        from rdkit import Chem, rdBase
    except ImportError as err:
        raise ImportError(
            "RDKit support requires the optional dependency: pip install 'hashmol3d[rdkit]'"
        ) from err
    return Chem, rdBase


def canonical_smiles(mol, *, conf_id: int | None = None) -> str:
    """Return canonical isomeric SMILES from a copy of an RDKit Mol.

    With ``conf_id=None``, use the supplied graph stereochemistry. Otherwise
    derive tetrahedral and double-bond stereo from that 3D conformer, replacing
    existing assignments. Atom maps are discarded; isotope and charge labels
    are retained. Ordinary explicit hydrogens are suppressed in SMILES only.
    Enhanced stereo groups are rejected because plain SMILES cannot encode them.
    """
    Chem, _ = _require_rdkit()
    if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        raise ValueError("expected a non-empty RDKit Mol")
    mol = Chem.Mol(mol)
    if mol.GetStereoGroups():
        raise ValueError("canonical_smiles does not support enhanced stereo groups")
    if any(atom.HasQuery() for atom in mol.GetAtoms()) or any(
        bond.HasQuery() for bond in mol.GetBonds()
    ):
        raise ValueError("canonical_smiles requires a molecule, not a query/SMARTS graph")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    if conf_id is not None:
        conf = mol.GetConformer(conf_id)
        if not conf.Is3D():
            raise ValueError("canonical_smiles requires a 3D conformer")
        # Unknown/wavy bond flags survive replaceExistingTags=True and can
        # suppress otherwise resolved E/Z geometry. The conformer is authoritative.
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol, confId=conf.GetId(), replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles or "|" in smiles or any(c.isspace() for c in smiles):
        raise ValueError("RDKit returned an invalid canonical SMILES")
    return smiles


def hash_rdkit(
    mol,
    *,
    conf_id: int = -1,
    charge: int | None = None,
    allow_implicit_hydrogens: bool = False,
    **kwargs,
) -> HashMol3DResult:
    """Hash one existing 3D conformer of an RDKit Mol without mutating it.

    Geometry includes exactly the atoms present, including explicit hydrogens;
    no hydrogens or coordinates are added. Missing/2D conformers are errors.
    Hydrogens represented only as atom counts are rejected unless
    ``allow_implicit_hydrogens=True`` explicitly permits incomplete geometry.
    Charge defaults to the graph's total formal charge. Other keywords go to
    ``hash_molecule``; the result is its geometry-only identifier.
    """
    if "method" in kwargs:
        kwargs["method"] = _resolve_method(kwargs["method"], stacklevel=3)
    Chem, _ = _require_rdkit()
    if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        raise ValueError("expected a non-empty RDKit Mol")
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "RDKit input has no coordinates; provide or explicitly generate 3D geometry"
        )
    conf = mol.GetConformer(conf_id)
    if not conf.Is3D():
        raise ValueError(
            "RDKit input has a 2D conformer; provide or explicitly generate 3D geometry"
        )
    # Unsanitized caller-built molecules may not yet have implicit valences.
    # Initialize only the copy.
    mol.UpdatePropertyCache(strict=False)
    if not allow_implicit_hydrogens and any(atom.GetTotalNumHs() for atom in mol.GetAtoms()):
        raise ValueError(
            "RDKit input has implicit hydrogens or H counts without coordinates; "
            "provide a complete explicit-H geometry, or set allow_implicit_hydrogens=True "
            "(--allow-implicit-hydrogens) to hash only the atoms present"
        )
    return hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        conf.GetPositions(),
        charge=Chem.GetFormalCharge(mol) if charge is None else charge,
        **kwargs,
    )


def _validate_mol_record(data: bytes, *, sdf: bool = False) -> None:
    """Reject extra MOL blocks while allowing ordinary SDF property data."""
    lines = data.splitlines()
    end = next((i for i in range(4, len(lines)) if lines[i].strip() == b"M  END"), None)
    if end is None:
        raise ValueError("MOL/SDF input must contain a complete M  END-terminated molecule")
    in_data = False
    for line in lines[end + 1 :]:
        if not line.strip():
            in_data = False
        elif sdf and (in_data or line.startswith(b">")):
            in_data = True
        else:
            raise ValueError(
                "MOL/SDF input must contain exactly one molecule; unexpected content "
                "after M  END (SDF values must follow a > data header)"
            )


def _validate_pdb_record(data: str) -> None:
    """Reject content after the first END record, which RDKit ignores."""
    lines = data.splitlines()
    end = next((i for i, line in enumerate(lines) if line[:6].rstrip() == "END"), None)
    if end is not None and any(line.strip() for line in lines[end + 1 :]):
        raise ValueError(
            "PDB input must contain exactly one molecule; unexpected content after END"
        )


def read_rdkit(path, *, input_format: str | None = None):
    """Read one molecule from MOL, SDF, MOL2, PDB, SMILES, or InChI.

    Format defaults to the file extension. SMILES/InChI files contain exactly
    one nonblank line (SMILES may have a whitespace-separated name). SDF files
    must contain exactly one record; iterate an RDKit supplier and call
    ``hash_rdkit`` for multiple molecules. Explicit H atoms are preserved.
    No coordinates are generated. Other RDKit readers can supply a Mol directly.
    """
    Chem, _ = _require_rdkit()
    path = Path(path)
    fmt = (input_format or path.suffix.lstrip(".")).lower()
    with path.open("rb") as stream:
        if fmt == "sdf":
            # Check record boundaries before RDKit buffers the stream: some
            # releases expose trailing blank lines as an extra invalid record.
            record = []
            terminated = False
            for line in stream:
                record.append(line)
                if line.strip() == b"$$$$":
                    terminated = True
                    break
            if not terminated or any(line.strip() for line in stream):
                raise ValueError(
                    "SDF input must contain exactly one terminated molecule; use an RDKit supplier"
                )
            _validate_mol_record(b"".join(record[:-1]), sdf=True)
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(b"".join(record)), removeHs=False)
            mol = next(supplier, None)
        else:
            data = stream.read().decode("utf-8")
            if fmt == "mol":
                _validate_mol_record(data.encode("utf-8"))
                mol = Chem.MolFromMolBlock(data, removeHs=False)
            elif fmt == "mol2":
                if sum(line.strip() == "@<TRIPOS>MOLECULE" for line in data.splitlines()) != 1:
                    raise ValueError("MOL2 input must contain exactly one molecule")
                mol = Chem.MolFromMol2Block(data, removeHs=False)
            elif fmt == "pdb":
                _validate_pdb_record(data)
                mol = Chem.MolFromPDBBlock(data, removeHs=False)
                # PDB coordinates are always 3D; RDKit flags an all-zero z
                # column as 2D, which would reject legitimate planar molecules.
                if mol is not None:
                    for conf in mol.GetConformers():
                        conf.Set3D(True)
            elif fmt in ("smi", "smiles", "inchi"):
                lines = [line.strip() for line in data.splitlines() if line.strip()]
                if len(lines) != 1:
                    raise ValueError("SMILES/InChI input must contain exactly one nonblank line")
                if fmt == "inchi":
                    mol = Chem.MolFromInchi(lines[0], removeHs=False)
                else:
                    if "|" in lines[0]:
                        raise ValueError(
                            "CXSMILES extensions are not supported; supply an RDKit Mol"
                        )
                    params = Chem.SmilesParserParams()
                    params.removeHs = False
                    params.allowCXSMILES = False
                    mol = Chem.MolFromSmiles(lines[0], params)
            else:
                raise ValueError(f"unsupported RDKit input format: {fmt!r}")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse a molecule from {path}")
    if mol.GetNumConformers() > 1:
        raise ValueError("input has multiple conformers; use hash_rdkit with an explicit conf_id")
    return mol


def hash_file(
    path,
    *,
    input_format: str = "xyz",
    generate_coordinates: bool = False,
    allow_implicit_hydrogens: bool = False,
    **kwargs,
) -> HashMol3DResult:
    """Hash a file; only explicitly selected features use RDKit.

    XYZ uses the native parser and defaults, regardless of installed packages.
    ``charge=None`` selects the default charge (0 for XYZ, the graph's formal
    charge otherwise).

    Other formats use ``read_rdkit``; PDB is refused for coordinate
    generation because its inferred bond orders cannot be trusted. PDB
    geometry-only input hashes the atoms present without checking guessed
    implicit-H counts. Other formats
    reject hydrogens without coordinates unless ``allow_implicit_hydrogens=True``.
    ``generate_coordinates=True`` explicitly
    replaces coordinates using ETKDGv3 (seed 0, one thread, explicit H atoms),
    without optimization. It is disallowed for XYZ. Generated geometries depend
    on RDKit version and input atom order; they are not canonical conformers.
    """
    if "method" in kwargs:
        kwargs["method"] = _resolve_method(kwargs["method"], stacklevel=3)
    if kwargs.get("charge", 0) is None:
        # As in hash_rdkit, None selects the default charge.
        del kwargs["charge"]
    fmt = input_format.lower()
    if fmt == "pdb" and generate_coordinates:
        raise ValueError(
            "generate_coordinates is not supported for PDB input: inferred bond orders "
            "can add incorrect hydrogens; use SDF or a chemically prepared RDKit Mol"
        )
    if fmt == "xyz":
        if allow_implicit_hydrogens:
            raise ValueError("allow_implicit_hydrogens is not supported for XYZ input")
        if generate_coordinates:
            raise ValueError("generate_coordinates is not supported for XYZ input")
        atomic_nums, coords = read_xyz(path)
        return hash_molecule(atomic_nums, coords, **kwargs)

    mol = read_rdkit(path, input_format=fmt)
    if generate_coordinates:
        Chem, _ = _require_rdkit()
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(mol)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("RDKit could not generate 3D coordinates")
    return hash_rdkit(
        mol,
        # PDB's guessed single bonds can imply extra H even for complete inputs.
        # Its geometry-only path must not infer completeness from that graph.
        allow_implicit_hydrogens=allow_implicit_hydrogens or fmt == "pdb",
        **kwargs,
    )
