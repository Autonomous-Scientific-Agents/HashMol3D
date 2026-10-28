"""Tests for explicitly selected RDKit input and stored S descriptors."""

import hashlib

import numpy as np
import pytest

from hashmol3d import (
    SearchBudgetExceeded,
    canonical_smiles,
    descriptor_geometry,
    hash_descriptor,
    hash_file,
    hash_molecule,
    hash_rdkit,
    hash_xyz,
    read_rdkit,
)
from hashmol3d.cli import cli

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

V10_WATER = (
    "V:10-FRAME-SHA256|P:1.0e-04|Z:1,1,8|"
    "F:1:0,-46880000,-75720000;1:0,-46880000,75720000;8:0,11720000,0"
)
V10_S_WATER = (
    V10_WATER.replace("V:10-FRAME-SHA256", "V:10-FRAME-SHA256-S1-RDKIT-2025.03.3", 1) + "|S:O"
)


def molecule(smiles="F[C@](Cl)(Br)I"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def xyz_file(tmp_path, mol):
    path = tmp_path / "molecule.xyz"
    path.write_text(Chem.MolToXYZBlock(mol))
    return path


@pytest.mark.parametrize("method", ["frame", "distance"])
def test_default_descriptor_is_exactly_core_descriptor(method):
    mol = molecule()
    expected = hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        mol.GetConformer().GetPositions(),
        method=method,
    )
    assert hash_rdkit(mol, method=method) == expected
    assert "|S:" not in expected.descriptor


@pytest.mark.parametrize(
    "suffix, message",
    [
        ("|S:", "S section is empty"),
        ("|X:CO", "not a HashMol3D descriptor"),
    ],
)
def test_malformed_s_descriptors_are_rejected(suffix, message):
    stored = V10_S_WATER.rsplit("|", 1)[0] + suffix
    with pytest.raises(ValueError, match=message):
        hash_descriptor(stored)


def test_stored_v10_s_descriptor_still_rehashes():
    payload = (
        "V:10-FRAME-SHA256-S1-RDKIT-2025.03.3|P:1.0e-04|Z:1,1,8|"
        "F:1:0,-4688,-7572;1:0,-4688,7572;8:0,1172,0|S:O"
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()
    assert hash_descriptor(V10_S_WATER) == expected[:32]
    assert hash_descriptor(V10_S_WATER, length=64) == expected


def test_descriptor_geometry_rejects_historical_s_descriptor():
    with pytest.raises(ValueError, match="cannot be re-hashed by this HashMol3D"):
        descriptor_geometry(V10_S_WATER)


@pytest.mark.parametrize("base", ["7-FRAME-SHA256", "11.0"])
def test_s_extension_requires_a_base_version_from_8_to_10(base):
    stored = V10_S_WATER.replace("V:10-FRAME-SHA256", "V:" + base, 1)
    with pytest.raises(ValueError, match="the S extension exists for versions 8-10 only"):
        hash_descriptor(stored)


def test_include_smiles_keyword_is_gone(tmp_path):
    with pytest.raises(TypeError):
        hash_rdkit(molecule(), include_smiles=True)
    mol = molecule("CO")
    path = tmp_path / "methanol.sdf"
    path.write_text(Chem.MolToMolBlock(mol) + "\n$$$$\n")
    # MOL blocks store coordinates with four decimals.
    assert hash_file(path, input_format="sdf") == hash_molecule(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        np.round(mol.GetConformer().GetPositions(), 4),
    )


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


@pytest.mark.parametrize("smiles", ["F/C=C/F", "F/C=C\\F"])
@pytest.mark.parametrize("flag", ["unknown", "wavy"])
def test_unknown_graph_stereo_does_not_override_3d_geometry(smiles, flag):
    mol = molecule(smiles)
    expected = hash_rdkit(mol)
    double = next(bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    if flag == "unknown":
        double.SetStereo(Chem.BondStereo.STEREOANY)
    else:
        double.SetBondDir(Chem.BondDir.EITHERDOUBLE)
    before = mol.ToBinary()
    # Graph-only calls still honor the unknown assignment; only the 3D path clears it.
    assert canonical_smiles(mol) == "FC=CF"
    assert canonical_smiles(mol, conf_id=0) == smiles
    assert hash_rdkit(mol) == expected
    assert mol.ToBinary() == before


def test_smiles_canonicalization_ignores_maps_order_and_ordinary_explicit_h():
    expected = canonical_smiles(Chem.MolFromSmiles("CCO"))
    assert canonical_smiles(Chem.MolFromSmiles("[OH:3][CH2:2][CH3:1]")) == expected
    assert canonical_smiles(Chem.AddHs(Chem.MolFromSmiles("OCC"))) == expected


def test_isotope_does_not_affect_geometry_result():
    mol = molecule("CO")
    isotope = Chem.Mol(mol)
    isotope.GetAtomWithIdx(0).SetIsotope(13)
    assert hash_rdkit(mol) == hash_rdkit(isotope)


def test_charge_and_multiplicity_policy():
    mol = molecule("[NH4+]")
    result = hash_rdkit(mol)
    assert result.charge == 1
    assert result.hash_str.split("_")[4] == "q1m1"
    triplet = hash_rdkit(mol, multiplicity=3)
    assert triplet.geometry_hash == result.geometry_hash
    assert triplet.descriptor == result.descriptor
    assert triplet.hash_str == result.hash_str.replace("_q1m1_", "_q1m3_")
    # An explicit charge overrides the graph charge; it changes only the state
    # tag (the default multiplicity follows electron parity: neutral NH4 is m2).
    neutral = hash_rdkit(mol, charge=0)
    assert neutral.charge == 0
    assert neutral.geometry_hash == result.geometry_hash
    assert neutral.hash_str == result.hash_str.replace("_q1m1_", "_q0m2_")


def test_selected_conformer():
    mol = molecule()
    # The geometry hash is mirror-invariant, so use a non-congruent conformer.
    stretched = Chem.Conformer(mol.GetConformer())
    for i, point in enumerate(stretched.GetPositions() * 1.1):
        stretched.SetAtomPosition(i, tuple(point))
    stretched.SetId(7)
    mol.AddConformer(stretched, assignId=False)
    first = hash_rdkit(mol, conf_id=0)
    assert first == hash_rdkit(mol)
    assert first.geometry_hash != hash_rdkit(mol, conf_id=7).geometry_hash
    with pytest.raises(ValueError):
        hash_rdkit(mol, conf_id=99)


@pytest.mark.parametrize("fmt", ["mol", "sdf", "pdb"])
def test_file_readers_preserve_explicit_atoms_and_geometry(tmp_path, fmt):
    mol = molecule("CO")
    path = tmp_path / ("methanol." + fmt)
    block = Chem.MolToPDBBlock(mol) if fmt == "pdb" else Chem.MolToMolBlock(mol)
    path.write_text(block + ("\n$$$$\n" if fmt == "sdf" else ""))
    read = read_rdkit(path)
    assert read.GetNumAtoms() == mol.GetNumAtoms()
    assert hash_file(path, input_format=fmt) == hash_rdkit(read)
    assert cli([str(path), "--input-format", fmt]) == 0


def test_sdf_rejects_multiple_or_invalid_records(tmp_path):
    path = tmp_path / "multi.sdf"
    block = Chem.MolToMolBlock(molecule("O")) + "\n$$$$\n"
    unterminated = Chem.MolToMolBlock(molecule("O")) + "\n"
    concatenated = unterminated + Chem.MolToMolBlock(molecule("C")) + "\n"
    for content in (
        block * 2,
        block + "bad\n$$$$\n",
        block + "bad\n",
        block + "$$$$\n",
        unterminated,
        concatenated,
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
    expected = hash_file(path, input_format="sdf")
    path.write_text(block + trailing)
    assert hash_file(path, input_format="sdf") == expected


@pytest.mark.parametrize("force_v3000", [False, True])
def test_mol_accepts_single_block_with_trailing_whitespace(tmp_path, force_v3000):
    mol = molecule("O")
    path = tmp_path / "single.mol"
    path.write_text(Chem.MolToMolBlock(mol, forceV3000=force_v3000) + "\n \t\n")
    assert read_rdkit(path).GetNumAtoms() == 3
    assert hash_file(path, input_format="mol").formula == "H2O"


@pytest.mark.parametrize("fmt", ["mol", "sdf", "mol2", "pdb"])
def test_file_readers_reject_concatenated_molecules(tmp_path, capsys, fmt):
    if fmt == "pdb":
        # RDKit stops at the first END record and would drop the second molecule.
        content = Chem.MolToPDBBlock(molecule("CO")) + Chem.MolToPDBBlock(molecule("CCCl"))
    elif fmt == "mol2":
        block = (
            "@<TRIPOS>MOLECULE\nwater\n3 2 0 0 0\nSMALL\nNO_CHARGES\n\n"
            "@<TRIPOS>ATOM\n1 O 0 0 0 O.3 1 HOH 0\n"
            "2 H1 .7572 .586 0 H 1 HOH 0\n3 H2 -.7572 .586 0 H 1 HOH 0\n"
            "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n"
        )
        content = block + block
    else:
        content = Chem.MolToMolBlock(molecule("O")) + "\n" + Chem.MolToMolBlock(molecule("C"))
        if fmt == "sdf":
            # The first molecule has no $$$$; a delimiter count alone misses it.
            content += "\n$$$$\n"
    path = tmp_path / ("concatenated." + fmt)
    path.write_text(content)
    with pytest.raises(ValueError, match="exactly one molecule"):
        read_rdkit(path)
    assert cli([str(path), "--input-format", fmt]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "exactly one molecule" in output.err
    assert "Traceback" not in output.err


def test_sdf_properties_can_contain_mol_end_text(tmp_path):
    path = tmp_path / "properties.sdf"
    path.write_text(
        Chem.MolToMolBlock(molecule("O"))
        + "\n>  <notes>\nM  END\nnot a second molecule\n\n>  <name>\nwater\n\n$$$$\n"
    )
    mol = read_rdkit(path)
    assert mol.GetNumAtoms() == 3
    assert mol.GetProp("notes") == "M  END\nnot a second molecule"
    assert mol.GetProp("name") == "water"
    assert hash_file(path, input_format="sdf").formula == "H2O"


def test_sdf_rejects_molecule_after_a_complete_data_item(tmp_path):
    path = tmp_path / "hidden.sdf"
    path.write_text(
        Chem.MolToMolBlock(molecule("O"))
        + "\n>  <name>\nwater\n\n"
        + Chem.MolToMolBlock(molecule("C"))
        + "\n$$$$\n"
    )
    with pytest.raises(ValueError, match="exactly one molecule"):
        read_rdkit(path)


@pytest.mark.parametrize("smiles", ["CCO", "[NH4+]"])
def test_hydrogens_without_coordinates_require_explicit_opt_in(tmp_path, capsys, smiles):
    mol = Chem.RemoveHs(molecule(smiles))
    before = mol.ToBinary()
    with pytest.raises(ValueError, match="allow_implicit_hydrogens=True"):
        hash_rdkit(mol)
    allowed = hash_rdkit(mol, allow_implicit_hydrogens=True)
    assert allowed.formula == ("C2O" if smiles == "CCO" else "N")
    assert allowed.charge == (0 if smiles == "CCO" else 1)
    assert mol.ToBinary() == before
    path = tmp_path / "implicit.sdf"
    path.write_text(Chem.MolToMolBlock(mol) + "\n$$$$\n")
    with pytest.raises(ValueError, match="implicit hydrogens"):
        hash_file(path, input_format="sdf")
    assert hash_file(path, input_format="sdf", allow_implicit_hydrogens=True) == hash_rdkit(
        read_rdkit(path), allow_implicit_hydrogens=True
    )
    args = [str(path), "--input-format", "sdf"]
    assert cli(args) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "--allow-implicit-hydrogens" in output.err
    assert cli([*args, "--allow-implicit-hydrogens"]) == 0
    # Explicitly generating full coordinates is another supported route.
    assert hash_file(path, input_format="sdf", generate_coordinates=True).formula == (
        "C2H6O" if smiles == "CCO" else "H4N"
    )


def test_unsanitized_explicit_h_molecule_initializes_only_its_copy():
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.removeHs = False
    unsanitized = Chem.MolFromSmiles("[H]O[H]", params)
    water = Chem.RenumberAtoms(molecule("O"), [1, 0, 2])
    unsanitized.AddConformer(water.GetConformer())
    assert unsanitized.NeedsUpdatePropertyCache()
    before = unsanitized.ToBinary()
    sanitized = Chem.Mol(unsanitized)
    Chem.SanitizeMol(sanitized)
    assert hash_rdkit(unsanitized) == hash_rdkit(sanitized)
    assert unsanitized.NeedsUpdatePropertyCache()
    assert unsanitized.ToBinary() == before


def test_unsanitized_implicit_h_molecule_uses_the_normal_hydrogen_policy():
    mol = Chem.MolFromSmiles("CCO", sanitize=False)
    mol.AddConformer(Chem.RemoveHs(molecule("CCO")).GetConformer())
    before = mol.ToBinary()
    with pytest.raises(ValueError, match="allow_implicit_hydrogens=True"):
        hash_rdkit(mol)
    result = hash_rdkit(mol, allow_implicit_hydrogens=True)
    assert result.formula == "C2O"
    assert mol.ToBinary() == before


@pytest.mark.parametrize("with_conect", [False, True])
def test_pdb_coordinate_generation_is_refused(tmp_path, capsys, with_conect):
    block = Chem.MolToPDBBlock(molecule("c1ccccc1"))
    if not with_conect:
        block = (
            "\n".join(line for line in block.splitlines() if not line.startswith("CONECT")) + "\n"
        )
    path = tmp_path / "benzene.pdb"
    path.write_text(block)
    # Geometry is still usable regardless of guessed bond orders.
    assert hash_file(path, input_format="pdb").formula == "C6H6"
    assert hash_file(path, input_format="pdb", allow_implicit_hydrogens=True).formula == "C6H6"
    with pytest.raises(ValueError, match="generate_coordinates is not supported for PDB"):
        hash_file(path, input_format="pdb", generate_coordinates=True)
    assert cli([str(path), "--input-format", "pdb", "--generate-coordinates"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "generate_coordinates is not supported for PDB" in output.err


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
    result = hash_file(path, input_format=fmt, generate_coordinates=True)
    assert result.formula == "CH4O"
    assert result == hash_file(path, input_format=fmt, generate_coordinates=True)
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


def test_errors_are_clean(tmp_path, capsys):
    path = tmp_path / "invalid.smi"
    for text in ("invalid!", "CCO\nCC", "", "F[C@](Cl)(Br)I |o1:1|"):
        path.write_text(text)
        assert cli([str(path), "--input-format", "smi", "--generate-coordinates"]) == 1
        output = capsys.readouterr()
        assert not output.out
        assert "hashmol3d:" in output.err
        assert "Traceback" not in output.err
    path = xyz_file(tmp_path, molecule("O"))
    assert cli([str(path), "--generate-coordinates"]) == 1
    path.write_text("3\nwater\nO 0 0 0\nH .7572 .586 0\nH -.7572 .586 0\n")
    with pytest.raises(SearchBudgetExceeded):
        hash_xyz(path, method="distance", node_budget=1)


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
        "hm3d_v12_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc"
    )


def test_embedding_failure_is_fatal(tmp_path, monkeypatch):
    path = tmp_path / "ethanol.smi"
    path.write_text("CCO")
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args: -1)
    with pytest.raises(ValueError, match="could not generate"):
        hash_file(path, input_format="smi", generate_coordinates=True)


def test_planar_pdb_is_hashed_as_3d(tmp_path):
    # RDKit flags an all-zero z column as 2D; PDB coordinates are always 3D.
    water = Chem.MolFromXYZBlock("3\n\nO 0 0 0\nH .757 .586 0\nH -.757 .586 0\n")
    path = tmp_path / "planar.pdb"
    path.write_text(Chem.MolToPDBBlock(water))
    xyz = xyz_file(tmp_path, water)
    assert hash_file(path, input_format="pdb") == hash_file(xyz)


@pytest.mark.parametrize("call", ["hash_rdkit", "hash_file"])
def test_canonical_alias_warning_points_at_the_caller(tmp_path, call):
    mol = molecule("O")
    with pytest.warns(DeprecationWarning, match="method='canonical'") as record:
        if call == "hash_rdkit":
            result = hash_rdkit(mol, method="canonical")
        else:
            result = hash_file(xyz_file(tmp_path, mol), method="canonical")
    # Attributed to this file, so Python's default filters show it.
    assert [w.filename for w in record] == [__file__]
    assert result == hash_rdkit(mol, method="distance")


def test_pdb_alternate_locations_are_rejected(tmp_path):
    atoms = [
        ("O", "A", 0.0, 0.0, 0.0),
        ("O", "B", 0.0, 0.0, 0.1),
        ("H1", " ", 0.757, 0.586, 0.0),
        ("H2", " ", -0.757, 0.586, 0.0),
    ]
    lines = [
        f"HETATM{i:5d} {name:<4}{alt}HOH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  0.50  0.00"
        f"          {name[0]:>2}"
        for i, (name, alt, x, y, z) in enumerate(atoms, 1)
    ]
    path = tmp_path / "altloc.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    with pytest.raises(ValueError, match=r"alternate locations \(A, B\)"):
        read_rdkit(path)
    # A single altLoc label A drops nothing.
    path.write_text("\n".join(lines[:1] + lines[2:]) + "\nEND\n")
    assert read_rdkit(path).GetNumAtoms() == 3


@pytest.mark.parametrize("label", ["B", "C"])
def test_pdb_atoms_rdkit_would_drop_are_rejected(tmp_path, label, capsys):
    # RDKit drops an atom labelled B or C even when no other location exists.
    water = Chem.MolFromXYZBlock("3\n\nO 0 0 0.1\nH .757 .586 0\nH -.757 .586 0\n")
    lines = Chem.MolToPDBBlock(water).splitlines()
    lines = [
        line[:16] + label + line[17:]
        if line.startswith("HETATM") and line[76:78].strip() == "O"
        else line
        for line in lines
    ]
    path = tmp_path / "relabelled.pdb"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="RDKit kept 2 of the 3 PDB atom records"):
        read_rdkit(path)
    with pytest.raises(ValueError, match="RDKit kept 2 of the 3"):
        hash_file(path, input_format="pdb")
    assert cli([str(path), "--input-format", "pdb"]) == 1
    assert "RDKit kept 2 of the 3" in capsys.readouterr().err


def test_multi_model_pdb_error_gives_a_working_route(tmp_path):
    water = Chem.MolFromXYZBlock("3\n\nO 0 0 0.1\nH .757 .586 0\nH -.757 .586 0\n")
    block = Chem.MolToPDBBlock(water).replace("END\n", "")
    path = tmp_path / "models.pdb"
    path.write_text(f"MODEL        1\n{block}ENDMDL\nMODEL        2\n{block}ENDMDL\nEND\n")
    with pytest.raises(ValueError, match="split the file, or read it with RDKit"):
        read_rdkit(path)


def test_hash_file_rejects_a_missing_input_format(tmp_path):
    with pytest.raises(ValueError, match="does not infer it from the file extension"):
        hash_file(xyz_file(tmp_path, molecule("O")), input_format=None)


def test_numpy_conf_id_is_accepted():
    mol = molecule("CCO")
    AllChem.EmbedMultipleConfs(mol, 2, randomSeed=1)
    expected = hash_rdkit(mol, conf_id=1)
    assert hash_rdkit(mol, conf_id=np.int64(1)) == expected
    assert canonical_smiles(mol, conf_id=np.int64(1)) == canonical_smiles(mol, conf_id=1)
    with pytest.raises(ValueError, match="conf_id must be an integer"):
        hash_rdkit(mol, conf_id=1.5)
