"""Opt-in RDKit adapters; importing this module does not import RDKit.

The NumPy-only descriptor is unchanged unless ``include_smiles=True``.
RDKit molecules are always copied before chemistry or coordinate operations.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from .core import HashMol3DResult, hash_molecule
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
        raise ValueError("S tags do not support enhanced stereo groups")
    if any(atom.HasQuery() for atom in mol.GetAtoms()) or any(
        bond.HasQuery() for bond in mol.GetBonds()
    ):
        raise ValueError("S tags require a molecule, not a query/SMARTS graph")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    if conf_id is not None:
        conf = mol.GetConformer(conf_id)
        if not conf.Is3D():
            raise ValueError("S tagging requires a 3D conformer")
        Chem.AssignStereochemistryFrom3D(mol, confId=conf.GetId(), replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles or "|" in smiles or any(c.isspace() for c in smiles):
        raise ValueError("RDKit returned an invalid S-tag payload")
    return smiles


def hash_rdkit(
    mol,
    *,
    conf_id: int = -1,
    include_smiles: bool = False,
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
    ``hash_molecule``. Without S, this returns the ordinary geometry descriptor.

    ``include_smiles=True`` appends canonical isomeric SMILES, using stereo from
    the selected geometry, and selects a separate versioned descriptor namespace.
    A charge override must then match the graph. Chemistry errors are fatal;
    there is no fallback to a descriptor without S.
    """
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
    if not allow_implicit_hydrogens and any(atom.GetTotalNumHs() for atom in mol.GetAtoms()):
        raise ValueError(
            "RDKit input has implicit hydrogens or H counts without coordinates; "
            "provide a complete explicit-H geometry, or set allow_implicit_hydrogens=True "
            "(--allow-implicit-hydrogens) to hash only the atoms present"
        )
    graph_charge = Chem.GetFormalCharge(mol)
    result = hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        conf.GetPositions(),
        charge=graph_charge if charge is None else charge,
        **kwargs,
    )
    if not include_smiles:
        return result
    if result.charge != graph_charge:
        raise ValueError("charge must match the RDKit graph when include_smiles=True")
    return _with_smiles(result, mol, conf.GetId())


def _with_smiles(result, mol, conf_id):
    _, rdBase = _require_rdkit()
    smiles = canonical_smiles(mol, conf_id=conf_id)
    version = f"{result.version}-S1-RDKIT-{quote(rdBase.rdkitVersion, safe='.-')}"
    descriptor = "V:" + version + "|" + result.descriptor.split("|", 1)[1] + "|S:" + smiles
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[: len(result.geometry_hash)]
    return replace(
        result,
        version=version,
        descriptor=descriptor,
        geometry_hash=digest,
        hash_str=result.hash_str.rsplit("-", 1)[0] + "-" + digest,
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
            for line in stream:
                record.append(line)
                if line.strip() == b"$$$$":
                    break
            if any(line.strip() for line in stream):
                raise ValueError(
                    "SDF input must contain exactly one molecule; use an RDKit supplier"
                )
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(b"".join(record)), removeHs=False)
            mol = next(supplier, None)
        else:
            data = stream.read().decode("utf-8")
            if fmt == "mol":
                mol = Chem.MolFromMolBlock(data, removeHs=False)
            elif fmt == "mol2":
                mol = Chem.MolFromMol2Block(data, removeHs=False)
            elif fmt == "pdb":
                mol = Chem.MolFromPDBBlock(data, removeHs=False)
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
    include_smiles: bool = False,
    generate_coordinates: bool = False,
    allow_implicit_hydrogens: bool = False,
    **kwargs,
) -> HashMol3DResult:
    """Hash a file; only explicitly selected features use RDKit.

    XYZ uses the native parser and defaults, regardless of installed packages.
    With S enabled, XYZ requires all atoms (including H); RDKit determines bonds
    from coordinates and the requested total charge (default zero).

    Other formats use ``read_rdkit``; PDB is refused for S tagging because its
    inferred bond orders cannot be trusted. PDB geometry-only input hashes the
    atoms present without checking guessed implicit-H counts. Other formats
    reject hydrogens without coordinates unless ``allow_implicit_hydrogens=True``.
    ``generate_coordinates=True`` explicitly
    replaces coordinates using ETKDGv3 (seed 0, one thread, explicit H atoms),
    without optimization. It is disallowed for XYZ. Generated geometries depend
    on RDKit version and input atom order; they are not canonical conformers.
    """
    fmt = input_format.lower()
    if fmt == "pdb" and include_smiles:
        raise ValueError(
            "PDB input is not supported for S tagging: bond orders are unreliable; "
            "use SDF or hash_rdkit with a chemically prepared Mol"
        )
    if fmt == "xyz":
        if allow_implicit_hydrogens:
            raise ValueError("allow_implicit_hydrogens is not supported for XYZ input")
        if generate_coordinates:
            raise ValueError("generate_coordinates is not supported for XYZ input")
        atomic_nums, coords = read_xyz(path)
        if not include_smiles:
            return hash_molecule(atomic_nums, coords, **kwargs)
        Chem, _ = _require_rdkit()
        from rdkit.Chem import rdDetermineBonds

        # Validate geometry/options before passing anything to bond perception.
        validated = hash_molecule(atomic_nums, coords, **kwargs)
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(atomic_nums))
        conf.Set3D(True)
        for i, (z, point) in enumerate(zip(atomic_nums, coords)):
            mol.AddAtom(Chem.Atom(int(z)))
            conf.SetAtomPosition(i, tuple(point))
        mol.AddConformer(conf)
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=validated.charge, embedChiral=True)
        except (ValueError, RuntimeError) as err:
            raise ValueError(
                f"bond perception failed for XYZ input at charge {validated.charge}; "
                "check the atom list and charge. Radicals, metals, or unusual valence "
                "may require a chemically prepared RDKit Mol for S tagging; "
                "multiplicity does not control bond perception"
            ) from err
        return _with_smiles(validated, mol, mol.GetConformer().GetId())

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
        include_smiles=include_smiles,
        # PDB's guessed single bonds can imply extra H even for complete inputs.
        # Its geometry-only path must not infer completeness from that graph.
        allow_implicit_hydrogens=allow_implicit_hydrogens or fmt == "pdb",
        **kwargs,
    )
