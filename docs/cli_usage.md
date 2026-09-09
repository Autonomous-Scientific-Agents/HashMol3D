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
`H2Oq0m1-bac9655753f489d6cbfdb299d59adbda`. See the
[API reference](api_reference.md#identifier-format) for the format spec.

## Options

| Flag | Long form | Default | Meaning |
| --- | --- | --- | --- |
| `-p` | `--precision`    | `1e-4` | Geometry-grid precision in angstroms; power of ten ≤ 1 |
| `-c` | `--charge`       | `0`    | Total formal charge |
| `-m` | `--multiplicity` | infer  | Spin multiplicity (inferred from electron parity if omitted) |
| `-l` | `--length`       | 32     | Hex chars in the geometry hash, 1–64 (default 32 = 128-bit; size by corpus, not molecule) |
|      | `--method`       | frame  | Descriptor method: `frame` or `canonical` |
|      | `--node-budget`  | 10000 | Maximum canonical search states, also after frame fallback |
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

Request the O(N²) canonical labelled-distance descriptor explicitly:

```bash
hashmol3d --method canonical ethanol.xyz
```

Hash a cation with explicit multiplicity, verbosely:

```bash
hashmol3d -c 1 -m 2 -v ethanol.xyz
```

Find all stored states of the same geometry by suffix-matching the
`geometry_hash` portion (after the `-`):

```bash
grep -E -- "-bac9655753f489d6cbfdb299d59adbda" identifiers.txt
```

## Exit codes

- `0` — success
- `1` — invalid input, an I/O error, or canonical search budget exhaustion
  (a message is written to stderr; no Python traceback is shown). On exhaustion
  no identifier is printed; increase `--node-budget` and retry, for example
  `hashmol3d --node-budget 100000 molecule.xyz`.

## Supported formats

Only standard XYZ input is supported in this release. The XYZ parser
validates the declared atom count and rejects malformed files.
