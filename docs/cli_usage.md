# HashMol3D CLI Usage

After installation, a `hashmol3d` command is available.

## Compute a hash for a file

```bash
hashmol3d compute molecule.xyz
```

Options:

- `--precision FLOAT`  Distance precision in Å (default: 1e-4)
- `--charge INT`       Override formal charge
- `--multiplicity INT` Override multiplicity
- `--hash-length INT`  Number of hex characters (default: 32)

Example:

```bash
hashmol3d compute --precision 1e-3 --hash-length 16 ethanol.sdf
```

## Show package version

```bash
hashmol3d version
```
