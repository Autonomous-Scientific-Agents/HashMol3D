
from hashmol3d.core import generate_hashmol3d
from rdkit import Chem
from rdkit.Chem import AllChem

def test_basic():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol)
    res = generate_hashmol3d(mol)
    assert len(res.hash_str)>0
