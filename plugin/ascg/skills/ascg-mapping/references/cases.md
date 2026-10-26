# Mapping decisions (general)

- When a source gives only resolutions (beads per unit), reconstruct the atom partition and confirm it with the
  user. Mark proposed or reconstructed partitions as unapproved until then.
- Functional groups that dominate the interactions (amide, ester, aromatic ring) normally get their own bead types.
- Small beads of two heavy atoms are acceptable for simple backbones; do not merge them just to reduce the bead
  count.
- End beads keep the interior type unless the end chemistry differs.
- Random copolymers: an independent sequence per chain from the build seed. Block copolymers: the same block
  sequence in every chain.
- System size: polymers ~1000 CG beads (chains = 1000 / beads per chain, >= 20 chains); molecular liquids and
  mixtures ~1000 molecules.
- CG dihedrals: combined bending-torsion on every path of three bonds, by default.
- Rings mapped to several beads form closed CG cycles: generate angles and dihedrals from the cyclic CG graph and
  check exclusions.
