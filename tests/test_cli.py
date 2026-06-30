"""Tests for the hashmol3d command-line interface."""

from hashmol3d import hash_xyz
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
    # Identifier is "<formula><state>-<hash>"; only the hash portion is
    # controlled by --length.
    assert out.startswith("H2Oq0m1-")
    assert len(out.rsplit("-", 1)[-1]) == 16


def test_cli_verbose(tmp_path, capsys):
    path = _write_water(tmp_path)
    rc = cli([path, "-v"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hash:" in out
    assert "descriptor:" in out
    assert "charge:" in out


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
