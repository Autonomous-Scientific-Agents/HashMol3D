from dataclasses import dataclass
from typing import Optional, Sequence

import hashlib
import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolfiles


# ============================================================
# HashMol3D result container
# ============================================================


@dataclass
class HashMol3DResult:
    """
    Container for HashMol3D information.

    Canonical identifier is just `.hash_str` (or str(obj)).
    Other fields are metadata for convenience.
    """

    hash_str: str  # canonical HashMol3D ID
    version: str  # e.g. "2-SHA256"
    precision: float  # distance precision in Å
    charge: int  # formal charge used
    multiplicity: int  # spin multiplicity used
    formula: str  # molecular formula (RDKit-based)
    canonical_smiles: str  # canonical SMILES (RDKit-based)
    descriptor: str  # raw descriptor string (for debugging / verification)

    def __str__(self) -> str:
        return self.hash_str


# ============================================================
# Internal helpers
# ============================================================


def _canonical_atom_order(mol: Chem.Mol) -> Sequence[int]:
    """
    Get a deterministic atom ordering using RDKit's canonical ranks.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol))
    ordered = [idx for idx, _ in sorted(enumerate(ranks), key=lambda x: (x[1], x[0]))]
    return ordered


def _precision_to_decimals(precision: float) -> int:
    """
    Convert a precision in Å (e.g. 1e-4) to number of decimal places.
    """
    if precision <= 0:
        raise ValueError("precision must be positive")
    decimals = int(round(-np.log10(precision)))
    return max(decimals, 0)


def _stereo_string(mol: Chem.Mol) -> str:
    """
    Construct a stereochemistry string based on RDKit's chiral centers.

    Returns:
        ""      if no chiral centers
        "R"     single R
        "S"     single S
        "R,S"   multiple centers, e.g. R at smallest index, S at next, etc.
        "R,?,S" unknown / unassigned are recorded as "?"
    """
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if not centers:
        return ""

    # sort by atom index for determinism
    centers = sorted(centers, key=lambda x: x[0])

    cfgs = []
    for idx, cfg in centers:
        if cfg in ("R", "S"):
            cfgs.append(cfg)
        else:
            cfgs.append("?")
    return ",".join(cfgs)


def _infer_charge(mol: Chem.Mol, user_charge: Optional[int]) -> int:
    """
    Use user-provided charge if given; otherwise use RDKit's formal charge
    (defaulting to 0 if something goes wrong).
    """
    if user_charge is not None:
        return int(user_charge)

    try:
        q = Chem.rdMolOps.GetFormalCharge(mol)
    except Exception:
        q = 0
    return int(q)


def _infer_multiplicity(
    atomic_nums: np.ndarray, charge: int, user_mult: Optional[int]
) -> int:
    """
    Use user-provided multiplicity if given; otherwise infer from electron count:
        electrons = sum(Z) - charge
        even → singlet (1)
        odd  → doublet (2)
    """
    if user_mult is not None:
        return int(user_mult)

    n_electrons = int(atomic_nums.sum()) - charge
    return 1 if (n_electrons % 2 == 0) else 2


# ============================================================
# Core HashMol3D generator
# ============================================================


def generate_hashmol3d(
    mol: Chem.Mol,
    conf_id: int = 0,
    precision: float = 1e-4,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    hash_length: int = 32,
    version: str = "2-SHA256",
) -> HashMol3DResult:
    """
    Generate the HashMol3D identifier for a given RDKit molecule with 3D coordinates.

    HashMol3D v2.0:
      - Uses SHA-256 on a descriptor including:
          * version tag
          * atomic numbers (canonical order)
          * distance matrix (upper triangle, rounded to `precision`)
          * stereochemistry (R/S/? sequence)
          * charge
          * multiplicity
      - Output is the truncated hex digest (hash_length chars, default 32).

    Args:
        mol: RDKit molecule with at least one 3D conformer.
        conf_id: Conformer index to use (default: 0).
        precision: Distance precision in Å (default: 1e-4).
        charge: Optional formal charge. If None, taken from RDKit formal charge.
        multiplicity: Optional spin multiplicity. If None, inferred from electron count.
        hash_length: Number of hex characters to keep from SHA-256 digest (16/32/64 typical).
        version: Version string (embedded in descriptor, not in output hash).

    Returns:
        HashMol3DResult instance. str(result) gives the canonical hash string.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D conformers.")

    if not (0 <= conf_id < mol.GetNumConformers()):
        raise ValueError(
            f"Invalid conf_id={conf_id}; molecule has {mol.GetNumConformers()} conformers."
        )

    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    # 1. Canonical atom ordering
    atom_order = _canonical_atom_order(mol)

    # 2. Coordinates and atomic numbers in canonical order
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in atom_order
        ],
        dtype=float,
    )  # shape (N, 3)

    atomic_nums = np.array(
        [mol.GetAtomWithIdx(i).GetAtomicNum() for i in atom_order],
        dtype=int,
    )

    # 3. Distance matrix (rotation/translation invariant)
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)  # (N, N)

    decimals = _precision_to_decimals(precision)
    iu = np.triu_indices(n_atoms, k=1)
    dvals = np.round(dmat[iu], decimals=decimals)

    # 4. Stereochemistry string
    stereo_str = _stereo_string(mol)

    # 5. Charge and multiplicity
    used_charge = _infer_charge(mol, charge)
    used_mult = _infer_multiplicity(atomic_nums, used_charge, multiplicity)

    # 6. Descriptor string (deterministic)
    z_part = ",".join(str(z) for z in atomic_nums)
    fmt = f"{{:.{decimals}f}}"
    d_part = ",".join(fmt.format(x) for x in dvals.tolist())

    # Put everything into the descriptor
    components = [
        f"V:{version}",
        f"PREC:{precision:.1e}",
        f"Z:{z_part}",
        f"D:{d_part}",
        f"STEREO:{stereo_str}",
        f"CHARGE:{used_charge}",
        f"MULT:{used_mult}",
    ]
    descriptor = "|".join(components)

    # 7. Hash with SHA-256 and truncate
    full_digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    hash_str = full_digest[:hash_length]

    # 8. Metadata (formula and canonical SMILES)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    return HashMol3DResult(
        hash_str=hash_str,
        version=version,
        precision=precision,
        charge=used_charge,
        multiplicity=used_mult,
        formula=formula,
        canonical_smiles=canonical_smiles,
        descriptor=descriptor,
    )


# ============================================================
# RDKit-based file loading and convenience wrapper
# ============================================================


def _load_mol_from_file(
    path: str,
    file_format: Optional[str] = None,
    embed_if_missing: bool = True,
    add_hs_for_embedding: bool = True,
) -> Chem.Mol:
    """
    Load a molecule from file using RDKit, and ensure it has a 3D conformer.

    Supports:
        - xyz
        - sdf/sd
        - mol
        - mol2
        - pdb/pdbqt
        - smi/smiles  (1st line)
    """
    if file_format is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        file_format = ext

    fmt = file_format.lower()

    if fmt == "xyz":
        if not hasattr(rdmolfiles, "MolFromXYZFile"):
            raise RuntimeError("MolFromXYZFile not available in this RDKit build.")
        mol = rdmolfiles.MolFromXYZFile(path)
    elif fmt in ("sdf", "sd"):
        mol = next(
            (m for m in rdmolfiles.SDMolSupplier(path, removeHs=False) if m), None
        )
    elif fmt == "mol":
        mol = rdmolfiles.MolFromMolFile(path, removeHs=False)
    elif fmt == "mol2":
        mol = rdmolfiles.MolFromMol2File(path, removeHs=False)
    elif fmt in ("pdb", "pdbqt"):
        mol = rdmolfiles.MolFromPDBFile(path, removeHs=False)
    elif fmt in ("smi", "smiles"):
        with open(path, "r") as f:
            line = f.readline().strip()
        if not line:
            raise ValueError(f"SMILES file is empty: {path}")
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
    else:
        raise ValueError(f"Unsupported or unknown file format: {file_format}")

    if mol is None:
        raise ValueError(f"Could not parse molecule from file: {path}")

    Chem.SanitizeMol(mol)

    has_3d = mol.GetNumConformers() > 0 and mol.GetConformer(0).Is3D()
    if not has_3d and embed_if_missing:
        m = mol
        if add_hs_for_embedding:
            m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        status = AllChem.EmbedMolecule(m, params)
        if status != 0:
            raise RuntimeError(f"3D embedding failed for molecule from {path}")
        AllChem.UFFOptimizeMolecule(m)
        mol = m

    if mol.GetNumConformers() == 0:
        raise ValueError(
            f"Molecule from {path} has no conformers and embedding disabled or failed."
        )

    return mol


def generate_hashmol3d_from_file(
    path: str,
    file_format: Optional[str] = None,
    precision: float = 1e-4,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    hash_length: int = 32,
    version: str = "2-SHA256",
) -> HashMol3DResult:
    """
    High-level convenience function: load molecule from file and compute HashMol3D.

    Args:
        path: Path to the input file.
        file_format: Optional explicit format; if None, inferred from extension.
        precision: Distance precision in Å.
        charge: Optional formal charge.
        multiplicity: Optional spin multiplicity.
        hash_length: Number of hex chars of SHA-256 output.
        version: Version string embedded in descriptor.

    Returns:
        HashMol3DResult instance.
    """
    mol = _load_mol_from_file(path, file_format=file_format)
    return generate_hashmol3d(
        mol=mol,
        conf_id=0,
        precision=precision,
        charge=charge,
        multiplicity=multiplicity,
        hash_length=hash_length,
        version=version,
    )
