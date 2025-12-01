"""
Shared pytest fixtures for HashMol3D tests.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture
def ethanol_mol():
    """Create an ethanol molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


@pytest.fixture
def water_mol():
    """Create a water molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


@pytest.fixture
def benzene_mol():
    """Create a benzene molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol



