# HashMol3D CLI Usage

After installation, a `hashmol3d` command is available.

## Compute a hash for an XYZ file

```bash
hashmol3d compute molecule.xyz
```

Options:

- `--precision FLOAT`   Distance precision in Å (default: `1e-4`)
- `--charge INT`        Total formal charge (default: `0`)
- `--multiplicity INT`  Spin multiplicity (default: inferred from electron parity)
- `--hash-length INT`   Number of hex characters (default: `32`, max `64`)
- `--verbose`, `-v`     Also print the canonical descriptor and metadata

Example:

```bash
hashmol3d compute --precision 1e-3 --hash-length 16 ethanol.xyz
```

## Show package version

```bash
hashmol3d version
```

## Supported formats

Only standard XYZ input is supported in this release. The XYZ parser
validates the declared atom count and rejects malformed files.
