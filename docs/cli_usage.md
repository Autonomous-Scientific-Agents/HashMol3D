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

The identifier has the form `<formula><state>-<hash>`, e.g.
`H2Oq0m1-a4ba9da41d888939961ef77dae43b297`. See the
[API reference](api_reference.md#identifier-format) for the format spec.

## Options

| Flag | Long form | Default | Meaning |
| --- | --- | --- | --- |
| `-p` | `--precision`    | `1e-4` | Distance precision in angstroms; power of ten ≤ 1 |
| `-c` | `--charge`       | `0`    | Total formal charge |
| `-m` | `--multiplicity` | infer  | Spin multiplicity (inferred from electron parity if omitted) |
| `-l` | `--length`       | 32     | Hex chars in the geometry hash, 1–64 (default 32 = 128-bit; size by corpus, not molecule) |
| `-v` | `--verbose`      |        | Also print the descriptor, formula, geometry hash, and metadata |

## Examples

Hash a geometry with default settings:

```bash
hashmol3d ethanol.xyz
# C2H6Oq0m1-...
```

Use a coarser precision and a longer fixed identifier:

```bash
hashmol3d -p 1e-3 -l 32 ethanol.xyz
```

Hash a cation with explicit multiplicity, verbosely:

```bash
hashmol3d -c 1 -m 2 -v ethanol.xyz
```

Find all stored states of the same geometry by suffix-matching the
`geometry_hash` portion (after the `-`):

```bash
grep -E -- "-a4ba9da41d888939961ef77dae43b297" identifiers.txt
```

## Exit codes

- `0` — success
- `1` — input file missing or malformed (a one-line message is written
  to stderr; no Python traceback is shown)

## Supported formats

Only standard XYZ input is supported in this release. The XYZ parser
validates the declared atom count and rejects malformed files.
