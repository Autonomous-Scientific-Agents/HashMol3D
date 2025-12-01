# HashMol3D Design Notes

This document collects background rationale for key design choices.

## Geometry vs. Topology

HashMol3D is explicitly a **conformer-level** identifier that depends on 3D
geometry, charge, and multiplicity. It is not a replacement for InChI or
topology-based identifiers (SMILES, RDKit MolHash, etc.), but rather a
complement for cases where conformer identity matters.

## Precision and Rounding

Small changes in floating-point coordinates should not produce different
identifiers. Therefore, distances are rounded to a user-defined precision
before hashing. This explicitly encodes the "resolution" at which conformers
are considered identical.

## Stereochemistry

Stereochemistry is encoded via RDKit's R/S perception. This ensures that
enantiomers (mirror images) produce different hashes, even if their distance
matrices are otherwise identical up to rotation.

## Charge and Multiplicity

Different charge or spin states generally correspond to different electronic
structures, even for the same geometry. HashMol3D includes both in the
descriptor to avoid conflating distinct physical systems.

## Isotopes

Isotopes are deliberately out-of-scope for descriptor version `2-SHA256`. They may be added in a
future version by extending the descriptor and version tag.
