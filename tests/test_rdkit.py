"""Tests for explicitly selected RDKit input and S descriptors."""

import hashlib

import numpy as np
import pytest

from hashmol3d import (
    DESCRIPTOR_VERSION,
    SearchBudgetExceeded,
    canonical_smiles,
    hash_file,
    hash_molecule,
    hash_rdkit,
    hash_xyz,
    read_rdkit,
)
from hashmol3d.cli import cli

pytest.importorskip("rdkit")
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


def molecule(smiles="F[C@](Cl)(Br)I"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def xyz_file(tmp_path, mol):
    path = tmp_path / "molecule.xyz"
    path.write_text(Chem.MolToXYZBlock(mol))
    return path


@pytest.mark.parametrize("method", ["frame", "canonical"])
def test_default_descriptor_is_exactly_core_descriptor(method):
    mol = molecule()
    expected = hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        mol.GetConformer().GetPositions(),
        method=method,
    )
    assert hash_rdkit(mol, method=method) == expected
    assert "|S:" not in expected.descriptor


@pytest.mark.parametrize("method", ["frame", "canonical"])
@pytest.mark.parametrize("length", [None, 1, 64])
def test_s_serialization_namespace_digest_and_no_mutation(method, length):
    mol = molecule()
    mol.GetAtomWithIdx(0).SetAtomMapNum(101)
    before = mol.ToBinary()
    base = hash_rdkit(mol, method=method, length=length)
    result = hash_rdkit(mol, include_smiles=True, method=method, length=length)
    assert result.version == f"{DESCRIPTOR_VERSION}-S1-RDKIT-{rdBase.rdkitVersion}"
    assert result.descriptor == (
        "V:" + result.version + "|" + base.descriptor.split("|", 1)[1] + "|S:F[C@](Cl)(Br)I"
    )
    assert (
        result.geometry_hash
        == hashlib.sha256(result.descriptor.encode()).hexdigest()[: length or 32]
    )
    assert str(result) == f"{base.formula}q0m1-{result.geometry_hash}"
    assert mol.ToBinary() == before


@pytest.mark.parametrize("method", ["frame", "canonical"])
def test_mirror_changes_only_opt_in_namespace(method):
    mol = molecule()
    mirror = Chem.Mol(mol)
    conf = mirror.GetConformer()
    points = conf.GetPositions()
    points[:, 0] *= -1
    for i, point in enumerate(points):
        conf.SetAtomPosition(i, tuple(point))
    assert hash_rdkit(mol, method=method) == hash_rdkit(mirror, method=method)
    left = hash_rdkit(mol, include_smiles=True, method=method)
    right = hash_rdkit(mirror, include_smiles=True, method=method)
    assert left.geometry_hash != right.geometry_hash
    assert left.descriptor.split("|S:")[1] == "F[C@](Cl)(Br)I"
    assert right.descriptor.split("|S:")[1] == "F[C@@](Cl)(Br)I"


def test_s_permutation_rotation_translation_invariance():
    mol = molecule()
    moved = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    conf = moved.GetConformer()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for i, point in enumerate(conf.GetPositions() @ rotation + [3, -2, 7]):
        conf.SetAtomPosition(i, tuple(point))
    assert hash_rdkit(mol, include_smiles=True) == hash_rdkit(moved, include_smiles=True)


@pytest.mark.parametrize(
    "left,right",
    [
        ("CCO", "COC"),
        ("F/C=C/F", "F/C=C\\F"),
        ("C", "[13CH4]"),
        ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"),
        ("O", "[2H]O"),
    ],
)
def test_smiles_distinguishes_isomers_and_isotopes(left, right):
    assert canonical_smiles(Chem.MolFromSmiles(left)) != canonical_smiles(Chem.MolFromSmiles(right))


def test_double_bond_stereo_from_coordinates():
    trans = hash_rdkit(molecule("F/C=C/F"), include_smiles=True)
    cis = hash_rdkit(molecule("F/C=C\\F"), include_smiles=True)
    assert trans.descriptor.split("|S:")[1] != cis.descriptor.split("|S:")[1]


def test_smiles_canonicalization_ignores_maps_order_and_ordinary_explicit_h():
    expected = canonical_smiles(Chem.MolFromSmiles("CCO"))
    assert canonical_smiles(Chem.MolFromSmiles("[OH:3][CH2:2][CH3:1]")) == expected
    assert canonical_smiles(Chem.AddHs(Chem.MolFromSmiles("OCC"))) == expected


def test_isotope_only_affects_s_descriptor():
    mol = molecule("CO")
    isotope = Chem.Mol(mol)
    isotope.GetAtomWithIdx(0).SetIsotope(13)
    assert hash_rdkit(mol) == hash_rdkit(isotope)
    assert hash_rdkit(mol, include_smiles=True) != hash_rdkit(isotope, include_smiles=True)


def test_charge_and_multiplicity_policy():
    mol = molecule("[NH4+]")
    assert hash_rdkit(mol).charge == 1
    result = hash_rdkit(mol, include_smiles=True)
    assert "[NH4+]" in result.descriptor
    assert (
        hash_rdkit(mol, include_smiles=True, multiplicity=3).geometry_hash == result.geometry_hash
    )
    assert hash_rdkit(mol, charge=0).charge == 0
    with pytest.raises(ValueError, match="charge must match"):
        hash_rdkit(mol, include_smiles=True, charge=0)


def test_selected_conformer():
    mol = molecule()
    mirror = Chem.Conformer(mol.GetConformer())
    for i, point in enumerate(mirror.GetPositions() * [-1, 1, 1]):
        mirror.SetAtomPosition(i, tuple(point))
    mirror.SetId(7)
    mol.AddConformer(mirror, assignId=False)
    assert hash_rdkit(mol, conf_id=0, include_smiles=True) != hash_rdkit(
        mol, conf_id=7, include_smiles=True
    )
    with pytest.raises(ValueError):
        hash_rdkit(mol, conf_id=99)


def test_xyz_bond_perception_is_opt_in(tmp_path):
    path = xyz_file(tmp_path, molecule())
    base = hash_xyz(path)
    result = hash_xyz(path, include_smiles=True)
    assert "|S:F[C@](Cl)(Br)I" in result.descriptor
    assert result.descriptor.split("|S:")[0].split("|", 1)[1] == base.descriptor.split("|", 1)[1]
    assert hash_xyz(path) == base
    charged = xyz_file(tmp_path, molecule("[NH4+]"))
    assert "|S:[NH4+]" in hash_xyz(charged, include_smiles=True, charge=1).descriptor


@pytest.mark.parametrize("fmt", ["mol", "sdf", "pdb"])
def test_file_readers_preserve_explicit_atoms_and_geometry(tmp_path, fmt):
    mol = molecule("CO")
    path = tmp_path / ("methanol." + fmt)
    block = Chem.MolToPDBBlock(mol) if fmt == "pdb" else Chem.MolToMolBlock(mol)
    path.write_text(block + ("\n$$$$\n" if fmt == "sdf" else ""))
    read = read_rdkit(path)
    assert read.GetNumAtoms() == mol.GetNumAtoms()
    assert hash_file(path, input_format=fmt) == hash_rdkit(read)
    if fmt == "pdb":
        assert cli([str(path), "--input-format", fmt]) == 0
        return
    assert "|S:CO" in hash_file(path, input_format=fmt, include_smiles=True).descriptor
    assert cli([str(path), "--input-format", fmt, "--include-smiles"]) == 0


def test_sdf_rejects_multiple_or_invalid_records(tmp_path):
    path = tmp_path / "multi.sdf"
    block = Chem.MolToMolBlock(molecule("O")) + "\n$$$$\n"
    for content in (
        block * 2,
        block + "bad\n$$$$\n",
        block + "bad\n",
        block + "$$$$\n",
        "bad\n$$$$\n",
        "",
    ):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_rdkit(path)


@pytest.mark.parametrize("trailing", ["", "\n", "   \n", "\r\n\t \r\n"])
def test_sdf_accepts_trailing_whitespace(tmp_path, trailing):
    path = tmp_path / "single.sdf"
    block = Chem.MolToMolBlock(molecule("O")) + "\n$$$$\n"
    path.write_text(block)
    expected = hash_file(path, input_format="sdf", include_smiles=True)
    path.write_text(block + trailing)
    assert hash_file(path, input_format="sdf", include_smiles=True) == expected


@pytest.mark.parametrize("include_smiles", [False, True])
@pytest.mark.parametrize("smiles", ["CCO", "[NH4+]"])
def test_hydrogens_without_coordinates_require_explicit_opt_in(
    tmp_path, capsys, include_smiles, smiles
):
    mol = Chem.RemoveHs(molecule(smiles))
    before = mol.ToBinary()
    with pytest.raises(ValueError, match="allow_implicit_hydrogens=True"):
        hash_rdkit(mol, include_smiles=include_smiles)
    allowed = hash_rdkit(mol, include_smiles=include_smiles, allow_implicit_hydrogens=True)
    assert allowed.formula == ("C2O" if smiles == "CCO" else "N")
    if include_smiles:
        assert allowed.descriptor.endswith("|S:" + smiles)
    assert mol.ToBinary() == before
    path = tmp_path / "implicit.sdf"
    path.write_text(Chem.MolToMolBlock(mol) + "\n$$$$\n")
    with pytest.raises(ValueError, match="implicit hydrogens"):
        hash_file(path, input_format="sdf", include_smiles=include_smiles)
    assert (
        hash_file(
            path, input_format="sdf", include_smiles=include_smiles, allow_implicit_hydrogens=True
        ).formula
        == allowed.formula
    )
    args = [str(path), "--input-format", "sdf"] + (["--include-smiles"] if include_smiles else [])
    assert cli(args) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "--allow-implicit-hydrogens" in output.err
    assert cli([*args, "--allow-implicit-hydrogens"]) == 0
    # Explicitly generating full coordinates is another supported route.
    assert hash_file(
        path, input_format="sdf", generate_coordinates=True, include_smiles=include_smiles
    ).formula == ("C2H6O" if smiles == "CCO" else "H4N")


@pytest.mark.parametrize("with_conect", [False, True])
def test_pdb_s_is_refused_even_with_hydrogen_override(tmp_path, capsys, with_conect):
    block = Chem.MolToPDBBlock(molecule("c1ccccc1"))
    if not with_conect:
        block = (
            "\n".join(line for line in block.splitlines() if not line.startswith("CONECT")) + "\n"
        )
    path = tmp_path / "benzene.pdb"
    path.write_text(block)
    # Geometry is still usable regardless of guessed bond orders.
    assert hash_file(path, input_format="pdb").formula == "C6H6"
    for generate in (False, True):
        with pytest.raises(ValueError, match="PDB input is not supported for S tagging"):
            hash_file(
                path,
                input_format="pdb",
                include_smiles=True,
                generate_coordinates=generate,
                allow_implicit_hydrogens=True,
            )
    assert (
        cli([str(path), "--input-format", "pdb", "--include-smiles", "--allow-implicit-hydrogens"])
        == 1
    )
    output = capsys.readouterr()
    assert not output.out
    assert "use SDF or hash_rdkit" in output.err


@pytest.mark.parametrize(
    "smiles,formula",
    [("CCO", "C2H6O"), ("C=C", "C2H4"), ("CC(=O)O", "C2H4O2"), ("c1ccccc1", "C6H6")],
)
def test_pdb_geometry_ignores_guessed_implicit_hydrogens(tmp_path, capsys, smiles, formula):
    block = Chem.MolToPDBBlock(molecule(smiles))
    path = tmp_path / "without_conect.pdb"
    path.write_text(
        "\n".join(line for line in block.splitlines() if not line.startswith("CONECT")) + "\n"
    )
    mol = read_rdkit(path)
    expected = hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], mol.GetConformer().GetPositions()
    )
    assert expected.formula == formula
    assert hash_file(path, input_format="pdb") == expected
    assert cli([str(path), "--input-format", "pdb"]) == 0
    assert capsys.readouterr().out.strip() == expected.hash_str
    if smiles != "CCO":
        # Direct Mol input still requires the caller to prepare the chemistry.
        assert any(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        with pytest.raises(ValueError, match="implicit hydrogens"):
            hash_rdkit(mol)


@pytest.mark.parametrize(
    "fmt,text", [("smiles", "CO methanol"), ("inchi", "InChI=1S/CH4O/c1-2/h2H,1H3")]
)
def test_coordinate_generation_requires_explicit_opt_in(tmp_path, fmt, text):
    path = tmp_path / ("input." + fmt)
    path.write_text(text + "\n")
    with pytest.raises(ValueError, match="no coordinates"):
        hash_file(path, input_format=fmt)
    result = hash_file(path, input_format=fmt, generate_coordinates=True, include_smiles=True)
    assert result.formula == "CH4O"
    assert "|S:CO" in result.descriptor
    assert result == hash_file(
        path, input_format=fmt, generate_coordinates=True, include_smiles=True
    )
    assert cli([str(path), "--input-format", fmt, "--generate-coordinates"]) == 0


def test_2d_and_invalid_graphs_are_rejected():
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    with pytest.raises(ValueError, match="2D"):
        hash_rdkit(mol)
    with pytest.raises(ValueError, match="enhanced stereo"):
        canonical_smiles(Chem.MolFromSmiles("F[C@](Cl)(Br)I |o1:1|"))
    with pytest.raises(ValueError, match="query"):
        canonical_smiles(Chem.MolFromSmarts("[#6]"))
    for invalid in (None, "CCO", Chem.Mol()):
        with pytest.raises(ValueError, match="non-empty"):
            hash_rdkit(invalid)


def test_errors_are_clean_and_never_drop_requested_s(tmp_path, capsys):
    path = tmp_path / "invalid.smi"
    for text in ("invalid!", "CCO\nCC", "", "F[C@](Cl)(Br)I |o1:1|"):
        path.write_text(text)
        assert cli([str(path), "--input-format", "smi", "--include-smiles"]) == 1
        output = capsys.readouterr()
        assert not output.out
        assert "hashmol3d:" in output.err
        assert "Traceback" not in output.err
    path = xyz_file(tmp_path, molecule("O"))
    assert cli([str(path), "--generate-coordinates"]) == 1
    path.write_text("3\nwater\nO 0 0 0\nH .7572 .586 0\nH -.7572 .586 0\n")
    with pytest.raises(SearchBudgetExceeded):
        hash_xyz(path, include_smiles=True, method="canonical", node_budget=1)


def test_mol2_reader(tmp_path):
    path = tmp_path / "water.mol2"
    path.write_text(
        "@<TRIPOS>MOLECULE\nwater\n3 2 0 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n1 O 0 0 0 O.3 1 HOH 0\n"
        "2 H1 .7572 .586 0 H 1 HOH 0\n3 H2 -.7572 .586 0 H 1 HOH 0\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n"
    )
    assert read_rdkit(path).GetNumAtoms() == 3
    assert hash_file(path, input_format="mol2").hash_str == (
        "H2Oq0m1-bac9655753f489d6cbfdb299d59adbda"
    )
    assert hash_file(path, input_format="mol2", include_smiles=True).descriptor.endswith("|S:O")


def test_bond_perception_failure_is_fatal_only_for_s(tmp_path, capsys):
    path = tmp_path / "bad_chemistry.xyz"
    path.write_text("2\nseparated carbon and hydrogen\nC 0 0 0\nH 10 0 0\n")
    assert hash_xyz(path).formula == "CH"
    assert cli([str(path), "--include-smiles"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "hashmol3d:" in output.err
    assert "bond perception failed for XYZ input at charge 0" in output.err
    assert "multiplicity does not control bond perception" in output.err
    with pytest.raises(ValueError, match="bond perception failed") as error:
        hash_xyz(path, include_smiles=True, multiplicity=2)
    assert error.value.__cause__ is not None


def test_embedding_failure_is_fatal(tmp_path, monkeypatch):
    path = tmp_path / "ethanol.smi"
    path.write_text("CCO")
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args: -1)
    with pytest.raises(ValueError, match="could not generate"):
        hash_file(path, input_format="smi", generate_coordinates=True)
