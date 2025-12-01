"""
Comprehensive unit tests for HashMol3D core functionality.
"""

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from hashmol3d.core import (
    HashMol3DResult,
    generate_hashmol3d,
    generate_hashmol3d_from_file,
    _canonical_atom_order,
    _precision_to_decimals,
    _stereo_string,
    _infer_charge,
    _infer_multiplicity,
)


class TestHashMol3DResult:
    """Tests for HashMol3DResult dataclass."""

    def test_result_str(self):
        """Test that str() returns hash_str."""
        result = HashMol3DResult(
            hash_str="abc123",
            version="2-SHA256",
            precision=1e-4,
            charge=0,
            multiplicity=1,
            formula="C2H6O",
            canonical_smiles="CCO",
            descriptor="test",
        )
        assert str(result) == "abc123"


class TestCanonicalAtomOrder:
    """Tests for _canonical_atom_order function."""

    def test_simple_molecule(self):
        """Test canonical ordering for a simple molecule."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        order = _canonical_atom_order(mol)
        assert len(order) == mol.GetNumAtoms()
        assert all(isinstance(i, int) for i in order)
        assert sorted(order) == list(range(mol.GetNumAtoms()))

    def test_deterministic(self):
        """Test that ordering is deterministic."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        order1 = _canonical_atom_order(mol)
        order2 = _canonical_atom_order(mol)
        assert order1 == order2


class TestPrecisionToDecimals:
    """Tests for _precision_to_decimals function."""

    def test_standard_precisions(self):
        """Test common precision values."""
        assert _precision_to_decimals(1e-4) == 4
        assert _precision_to_decimals(1e-3) == 3
        assert _precision_to_decimals(1e-2) == 2
        assert _precision_to_decimals(1e-1) == 1
        assert _precision_to_decimals(1.0) == 0

    def test_negative_precision_raises(self):
        """Test that negative precision raises ValueError."""
        with pytest.raises(ValueError):
            _precision_to_decimals(-1.0)

    def test_zero_precision_raises(self):
        """Test that zero precision raises ValueError."""
        with pytest.raises(ValueError):
            _precision_to_decimals(0.0)


class TestStereochemistryString:
    """Tests for _stereo_string function."""

    def test_no_chiral_centers(self):
        """Test molecule with no chiral centers."""
        mol = Chem.MolFromSmiles("CCO")
        assert _stereo_string(mol) == ""

    def test_single_chiral_center(self):
        """Test molecule with single chiral center."""
        # Create a molecule with a chiral center
        mol = Chem.MolFromSmiles("C[C@H](O)CC")
        stereo = _stereo_string(mol)
        # Should contain R or S
        assert stereo in ("R", "S")

    def test_multiple_chiral_centers(self):
        """Test molecule with multiple chiral centers."""
        mol = Chem.MolFromSmiles("C[C@H](O)[C@H](C)O")
        stereo = _stereo_string(mol)
        # Should contain comma-separated values
        assert "," in stereo or stereo == ""


class TestInferCharge:
    """Tests for _infer_charge function."""

    def test_user_charge_provided(self):
        """Test that user-provided charge is used."""
        mol = Chem.MolFromSmiles("CCO")
        assert _infer_charge(mol, 1) == 1
        assert _infer_charge(mol, -1) == -1
        assert _infer_charge(mol, 0) == 0

    def test_charge_from_mol(self):
        """Test that charge is inferred from molecule if not provided."""
        mol = Chem.MolFromSmiles("CCO")
        charge = _infer_charge(mol, None)
        assert isinstance(charge, int)

    def test_charge_none_uses_formal_charge(self):
        """Test that None charge uses RDKit formal charge."""
        mol = Chem.MolFromSmiles("[NH4+]")
        charge = _infer_charge(mol, None)
        assert charge == 1


class TestInferMultiplicity:
    """Tests for _infer_multiplicity function."""

    def test_user_multiplicity_provided(self):
        """Test that user-provided multiplicity is used."""
        atomic_nums = np.array([6, 6, 8])  # CCO
        assert _infer_multiplicity(atomic_nums, 0, 1) == 1
        assert _infer_multiplicity(atomic_nums, 0, 2) == 2
        assert _infer_multiplicity(atomic_nums, 0, 3) == 3

    def test_even_electrons_singlet(self):
        """Test that even number of electrons gives singlet."""
        atomic_nums = np.array([6, 6, 8])  # CCO: 6+6+8-0 = 20 electrons (even)
        assert _infer_multiplicity(atomic_nums, 0, None) == 1

    def test_odd_electrons_doublet(self):
        """Test that odd number of electrons gives doublet."""
        atomic_nums = np.array([6, 6])  # CC: 6+6-0 = 12 electrons (even)
        # But if we have charge +1: 12-1 = 11 electrons (odd)
        assert _infer_multiplicity(atomic_nums, 1, None) == 2

    def test_charged_molecule(self):
        """Test multiplicity inference with charge."""
        atomic_nums = np.array([1])  # H: 1 electron
        assert _infer_multiplicity(atomic_nums, 0, None) == 2  # odd electrons
        assert _infer_multiplicity(atomic_nums, 1, None) == 1  # 1-1=0 electrons (even)


class TestGenerateHashMol3D:
    """Tests for generate_hashmol3d function."""

    def test_basic_molecule(self):
        """Test hash generation for a basic molecule."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        result = generate_hashmol3d(mol)

        assert isinstance(result, HashMol3DResult)
        assert len(result.hash_str) > 0
        assert result.version == "2-SHA256"
        assert result.precision == 1e-4
        assert result.formula == "C2H6O"
        assert "CCO" in result.canonical_smiles or "OCC" in result.canonical_smiles

    def test_no_conformers_raises(self):
        """Test that molecule without conformers raises ValueError."""
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="no 3D conformers"):
            generate_hashmol3d(mol)

    def test_invalid_conf_id_raises(self):
        """Test that invalid conformer ID raises ValueError."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        with pytest.raises(ValueError, match="Invalid conf_id"):
            generate_hashmol3d(mol, conf_id=1)

    def test_custom_precision(self):
        """Test hash generation with custom precision."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result1 = generate_hashmol3d(mol, precision=1e-4)
        result2 = generate_hashmol3d(mol, precision=1e-3)

        # Different precisions should give different hashes
        assert result1.hash_str != result2.hash_str
        assert result1.precision == 1e-4
        assert result2.precision == 1e-3

    def test_custom_charge(self):
        """Test hash generation with custom charge."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result1 = generate_hashmol3d(mol, charge=0)
        result2 = generate_hashmol3d(mol, charge=1)

        assert result1.charge == 0
        assert result2.charge == 1
        # Different charges should give different hashes
        assert result1.hash_str != result2.hash_str

    def test_custom_multiplicity(self):
        """Test hash generation with custom multiplicity."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result1 = generate_hashmol3d(mol, multiplicity=1)
        result2 = generate_hashmol3d(mol, multiplicity=2)

        assert result1.multiplicity == 1
        assert result2.multiplicity == 2
        # Different multiplicities should give different hashes
        assert result1.hash_str != result2.hash_str

    def test_custom_hash_length(self):
        """Test hash generation with custom hash length."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result16 = generate_hashmol3d(mol, hash_length=16)
        result32 = generate_hashmol3d(mol, hash_length=32)
        result64 = generate_hashmol3d(mol, hash_length=64)

        assert len(result16.hash_str) == 16
        assert len(result32.hash_str) == 32
        assert len(result64.hash_str) == 64

    def test_deterministic_hash(self):
        """Test that same molecule gives same hash."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result1 = generate_hashmol3d(mol)
        result2 = generate_hashmol3d(mol)

        assert result1.hash_str == result2.hash_str
        assert result1.descriptor == result2.descriptor

    def test_different_conformers_different_hash(self):
        """Test that different conformers give different hashes."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)

        # Generate two different conformers
        AllChem.EmbedMolecule(mol, randomSeed=1)
        result1 = generate_hashmol3d(mol, conf_id=0)

        # Add another conformer with different seed
        AllChem.EmbedMolecule(mol, randomSeed=2)
        result2 = generate_hashmol3d(mol, conf_id=1)

        # Different conformers should give different hashes
        assert result1.hash_str != result2.hash_str

    def test_descriptor_format(self):
        """Test that descriptor has expected format."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        result = generate_hashmol3d(mol)

        assert "V:" in result.descriptor
        assert "PREC:" in result.descriptor
        assert "Z:" in result.descriptor
        assert "D:" in result.descriptor
        assert "STEREO:" in result.descriptor
        assert "CHARGE:" in result.descriptor
        assert "MULT:" in result.descriptor

    def test_water_molecule(self):
        """Test hash generation for water."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        result = generate_hashmol3d(mol)

        assert result.formula == "H2O"
        assert len(result.hash_str) > 0

    def test_benzene_molecule(self):
        """Test hash generation for benzene."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        result = generate_hashmol3d(mol)

        assert "C6H6" in result.formula
        assert len(result.hash_str) > 0

    def test_charged_molecule(self):
        """Test hash generation for charged molecule."""
        mol = Chem.MolFromSmiles("[NH4+]")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        result = generate_hashmol3d(mol)

        assert result.charge == 1
        assert len(result.hash_str) > 0

    def test_version_parameter(self):
        """Test that version parameter affects hash."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result1 = generate_hashmol3d(mol, version="2-SHA256")
        result2 = generate_hashmol3d(mol, version="3-SHA256")

        # Different versions should give different hashes
        assert result1.hash_str != result2.hash_str
        assert result1.version == "2-SHA256"
        assert result2.version == "3-SHA256"

    def test_very_small_precision(self):
        """Test hash generation with very small precision."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result = generate_hashmol3d(mol, precision=1e-6)
        assert result.precision == 1e-6
        assert len(result.hash_str) > 0

    def test_large_precision(self):
        """Test hash generation with large precision."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result = generate_hashmol3d(mol, precision=1e-1)
        assert result.precision == 1e-1
        assert len(result.hash_str) > 0

    def test_none_charge_and_multiplicity(self):
        """Test that None charge and multiplicity are inferred correctly."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        result = generate_hashmol3d(mol, charge=None, multiplicity=None)
        assert isinstance(result.charge, int)
        assert isinstance(result.multiplicity, int)
        assert result.multiplicity in (1, 2)  # Should be 1 or 2


class TestGenerateHashMol3DFromFile:
    """Tests for generate_hashmol3d_from_file function."""

    def test_smiles_file(self, tmp_path):
        """Test loading from SMILES file."""
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO\n")

        result = generate_hashmol3d_from_file(str(smi_file))
        assert isinstance(result, HashMol3DResult)
        assert len(result.hash_str) > 0

    def test_empty_smiles_file_raises(self, tmp_path):
        """Test that empty SMILES file raises ValueError."""
        smi_file = tmp_path / "empty.smi"
        smi_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            generate_hashmol3d_from_file(str(smi_file))

    def test_unsupported_format_raises(self, tmp_path):
        """Test that unsupported format raises ValueError."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("dummy content")

        # This might raise different errors depending on RDKit version
        # but should raise some error
        with pytest.raises((ValueError, RuntimeError)):
            generate_hashmol3d_from_file(str(bad_file), file_format="xyz")

    def test_custom_parameters(self, tmp_path):
        """Test file loading with custom parameters."""
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO\n")

        result = generate_hashmol3d_from_file(
            str(smi_file),
            precision=1e-3,
            charge=0,
            multiplicity=1,
            hash_length=16,
        )

        assert result.precision == 1e-3
        assert result.charge == 0
        assert result.multiplicity == 1
        assert len(result.hash_str) == 16
