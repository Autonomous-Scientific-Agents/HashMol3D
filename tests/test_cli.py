"""Tests for the hashmol3d command-line interface."""

import subprocess
import sys

import pytest

from hashmol3d import hash_file, hash_xyz
from hashmol3d.cli import cli


def _write_water(tmp_path):
    p = tmp_path / "water.xyz"
    p.write_text(
        "3\nwater\n"
        "O  0.0000  0.0000  0.0000\n"
        "H  0.7572  0.5860  0.0000\n"
        "H -0.7572  0.5860  0.0000\n"
    )
    return str(p)


def test_cli_prints_hash(tmp_path, capsys):
    path = _write_water(tmp_path)
    rc = cli([path])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == hash_xyz(path).hash_str


def test_cli_length_option(tmp_path, capsys):
    path = _write_water(tmp_path)
    rc = cli([path, "-l", "16"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # Identifier is "hm3d_v<major>_p<dec>_<formula>_<state>_<hash>"; only the
    # hash portion is controlled by --length.
    assert out.startswith("hm3d_v11_p4_H2O_q0m1_")
    assert len(out.split("_")[5]) == 16


def test_cli_method_option(tmp_path, capsys):
    path = _write_water(tmp_path)
    rc = cli([path, "--method", "distance", "-v"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "|D:" in out


def test_cli_verbose(tmp_path, capsys):
    path = _write_water(tmp_path)
    rc = cli([path, "-v"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert any(line.startswith("identifier:") for line in lines)
    assert any(line.startswith("geometry_hash:") for line in lines)
    assert any(line.startswith("descriptor:") for line in lines)
    assert any(line.startswith("charge:") for line in lines)


def test_cli_charge_changes_hash(tmp_path, capsys):
    path = _write_water(tmp_path)
    cli([path])
    neutral = capsys.readouterr().out.strip()
    cli([path, "-c", "1"])
    cation = capsys.readouterr().out.strip()
    assert neutral != cation


def test_cli_missing_file_is_clean_error(tmp_path, capsys):
    rc = cli([str(tmp_path / "nope.xyz")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hashmol3d:" in err


def test_cli_bad_xyz_is_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.xyz"
    bad.write_text("not_a_count\n\nC 0 0 0\n")
    rc = cli([str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hashmol3d:" in err


def test_cli_budget_exhaustion_and_retry(tmp_path, capsys):
    path = _write_water(tmp_path)
    for verbose in ([], ["-v"]):
        assert cli([path, "--method", "distance", "--node-budget", "2", *verbose]) == 1
        out = capsys.readouterr()
        assert out.out == ""
        assert "no identifier or hash was created" in out.err
        assert "--node-budget" in out.err
        assert "Traceback" not in out.err
    assert cli([path, "--method", "distance", "--node-budget", "3"]) == 0
    assert capsys.readouterr().out.strip() == hash_xyz(path, method="distance").hash_str


def test_cli_invalid_node_budget(tmp_path, capsys):
    assert cli([_write_water(tmp_path), "--node-budget", "0"]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "node_budget must be a positive integer" in out.err


def test_cli_canonical_alias_warns_and_uses_distance(tmp_path, capsys):
    path = _write_water(tmp_path)
    assert cli([path, "--method", "canonical"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == hash_xyz(path, method="distance").hash_str
    assert "--method canonical is deprecated" in out.err


def test_hash_xyz_canonical_alias_warning_points_at_the_caller(tmp_path):
    with pytest.warns(DeprecationWarning, match="method='canonical'") as record:
        hash_xyz(_write_water(tmp_path), method="canonical")
    assert [w.filename for w in record] == [__file__]


def test_charge_none_selects_the_default_for_xyz(tmp_path):
    path = _write_water(tmp_path)
    assert hash_xyz(path, charge=None) == hash_xyz(path)
    assert hash_file(path, charge=None) == hash_xyz(path)


@pytest.mark.parametrize("module", ["hashmol3d", "hashmol3d.cli"])
def test_python_dash_m_runs_the_cli(tmp_path, module):
    path = _write_water(tmp_path)
    run = subprocess.run(
        [sys.executable, "-m", module, path], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0
    assert run.stdout.strip() == hash_xyz(path).hash_str
    missing = subprocess.run(
        [sys.executable, "-m", module, str(tmp_path / "nope.xyz")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1
