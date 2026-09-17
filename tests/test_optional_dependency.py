"""Dependency isolation is tested even when RDKit is installed."""

import subprocess
import sys


def test_default_api_and_cli_work_with_rdkit_imports_blocked(tmp_path):
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH .7572 .586 0\nH -.7572 .586 0\n")
    script = """
import importlib.abc
import sys

class BlockRDKit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rdkit' or fullname.startswith('rdkit.'):
            raise ModuleNotFoundError('RDKit intentionally unavailable')

sys.meta_path.insert(0, BlockRDKit())
import hashmol3d
from hashmol3d.cli import cli
result = hashmol3d.hash_xyz(sys.argv[1])
assert result.hash_str == 'H2Oq0m1-bac9655753f489d6cbfdb299d59adbda'
assert result == hashmol3d.hash_file(sys.argv[1])
assert cli([sys.argv[1]]) == 0
assert 'rdkit' not in sys.modules
try:
    hashmol3d.hash_xyz(sys.argv[1], include_smiles=True)
except ImportError as error:
    assert "hashmol3d[rdkit]" in str(error)
else:
    raise AssertionError('RDKit request must fail')
assert cli([sys.argv[1], '--include-smiles']) == 1
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(xyz)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "hashmol3d[rdkit]" in completed.stderr
    assert "Traceback" not in completed.stderr
