# HashMol3D CLI Usage

After installation, a `hashmol3d` command is available; `python -m hashmol3d`
runs the same CLI.

## Synopsis

```bash
hashmol3d FILE [options]
hashmol3d --version
hashmol3d --help
```

`FILE` is the path to a molecular input file (XYZ by default; see
`--input-format`). The default
behavior prints the HashMol3D identifier to stdout, one line, with no
extra formatting — suitable for piping.

The identifier has the form
`hm3d_v<major version>_p<decimals>_<formula>_<state>_<geometry hash>`, e.g.
`hm3d_v12_p4_H2O_q0m1_8e9eedf5032cf387541d637f56c682dc`. See the
[API reference](api_reference.md#identifier-format) for the format spec.

## Options

| Flag | Long form | Default | Meaning |
| --- | --- | --- | --- |
| `-p` | `--precision`    | `1e-4` | Geometry-grid precision in angstroms; power of ten ≤ 1 |
| `-c` | `--charge`       | infer  | Total formal charge: 0 for XYZ; graph charge for RDKit formats |
| `-m` | `--multiplicity` | infer  | Spin multiplicity (inferred from electron parity if omitted) |
| `-l` | `--length`       | 32     | Hex chars in the geometry hash, 1–64 (default 32 = 128-bit; size by corpus, not molecule) |
|      | `--method`       | frame  | Descriptor method: `frame` or `distance` (`canonical` is a deprecated alias of `distance` and prints a warning to stderr) |
|      | `--node-budget`  | 10000 | Maximum canonical search states, also after frame fallback |
|      | `--input-format` | xyz | `xyz`, `mol`, `sdf`, `mol2`, `pdb`, `smi`/`smiles`, or `inchi` |
|      | `--generate-coordinates` | off | Generate a new 3D conformer with explicit H atoms (not XYZ or PDB) |
|      | `--allow-implicit-hydrogens` | off | Accept hydrogens without coordinates and hash only atoms present (non-XYZ) |
| `-v` | `--verbose`      |        | Print a labelled block (identifier, formula, geometry hash, descriptor, version, precision, charge, multiplicity) instead of the bare identifier |

## Examples

The example files are in [examples](examples/); run these commands from
that directory.

Hash a geometry with default settings:

```bash
hashmol3d ethanol.xyz
# hm3d_v12_p4_C2H6O_q0m1_...
```

Use a coarser precision and a shorter, 20-hex-character hash:

```bash
hashmol3d -p 1e-3 -l 20 ethanol.xyz
```

Request the O(N²) labelled-distance descriptor (distance method) explicitly:

```bash
hashmol3d --method distance ethanol.xyz
```

Hash a cation with explicit multiplicity, verbosely:

```bash
hashmol3d -c 1 -m 2 -v ethanol.xyz
```

Find all stored states of the same geometry by suffix-matching the
`geometry_hash`, the last field:

```bash
grep -E -- "_8e9eedf5032cf387541d637f56c682dc$" identifiers.txt
```

## Exit codes

- `0` — success
- `1` — invalid input, an I/O error, or canonical search budget exhaustion
  (a message is written to stderr; no Python traceback is shown). On exhaustion
  no identifier is printed; increase `--node-budget` and retry, for example
  `hashmol3d --node-budget 100000 molecule.xyz`.
- `2` — command-line usage error, such as a missing `FILE` or an unknown
  option value (reported by argparse).

## Supported formats

Standard XYZ remains the default input. The XYZ parser
validates the declared atom count and rejects malformed files. If non-blank
lines follow the declared atoms (a multi-frame trajectory, or a header that
under-counts the atoms), only the first frame is hashed and a warning is
printed to stderr. Warnings, such as this one or a frame-method fallback,
are printed as `hashmol3d: warning: <message>`.

Install the `rdkit` extra (`pip install -e '.[rdkit]'` from a source checkout)
for the other formats. Select the format
explicitly; file extensions never switch the CLI backend automatically. The
identifier depends only on the atoms and coordinates read, whatever the
format. Inputs must contain a single molecule; SMILES/InChI and 2D structures
require explicit coordinate generation:

```bash
hashmol3d molecule.sdf --input-format sdf
hashmol3d molecule.smi --input-format smi --generate-coordinates
```

RDKit input other than PDB with implicit H atoms or atom-level H counts without coordinates is
rejected by default. Provide complete coordinates, explicitly generate them,
or use `--allow-implicit-hydrogens` to accept an incomplete geometry/formula.
PDB input hashes exactly the atoms in the file, without checking implicit H
counts derived from guessed bond orders, and cannot be combined with
`--generate-coordinates`. A PDB file is rejected if RDKit would drop any of
its atom records, such as alternate locations (altLoc) other than the first. Missing RDKit, invalid input, and embedding failure
exit with code 1. See the [RDKit guide](rdkit.md) for input policies and
limitations.
