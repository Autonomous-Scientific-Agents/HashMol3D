# HashMol3D CLI Usage

After installation, a `hashmol3d` command is available.

## Synopsis

```bash
hashmol3d FILE [options]
hashmol3d --version
hashmol3d --help
```

`FILE` is the path to a standard XYZ geometry file. The default
behavior prints the HashMol3D identifier to stdout, one line, with no
extra formatting — suitable for piping.

## Options

| Flag | Long form | Default | Meaning |
| --- | --- | --- | --- |
| `-p` | `--precision`    | `1e-4` | Distance precision in angstroms |
| `-c` | `--charge`       | `0`    | Total formal charge |
| `-m` | `--multiplicity` | infer  | Spin multiplicity (inferred from electron parity if omitted) |
| `-l` | `--length`       | `32`   | Hex characters of SHA-256 digest to retain (1–64) |
| `-v` | `--verbose`      |        | Also print the descriptor, version, and metadata |

## Examples

Hash a geometry with default settings:

```bash
hashmol3d ethanol.xyz
```

Use a coarser precision and a shorter identifier:

```bash
hashmol3d -p 1e-3 -l 16 ethanol.xyz
```

Hash a cation with explicit multiplicity, verbosely:

```bash
hashmol3d -c 1 -m 2 -v ethanol.xyz
```

## Exit codes

- `0` — success
- `1` — input file missing or malformed (a one-line message is written
  to stderr; no Python traceback is shown)

## Supported formats

Only standard XYZ input is supported in this release. The XYZ parser
validates the declared atom count and rejects malformed files.
