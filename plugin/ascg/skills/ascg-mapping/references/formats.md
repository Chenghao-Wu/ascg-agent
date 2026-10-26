# Writing a mapping down

A mapping is a JSON object. `ascg map check --spec '<json>'` validates it, and the same JSON goes to `ascg map save`.
Write it in the **annotated-smiles** form unless the user already has atom indices.

## annotated-smiles (preferred)

Every heavy atom carries its bead number as the SMILES atom-map number. `bead_types` gives each bead number its
type.

```json
{"format": "annotated-smiles",
 "smiles": "[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]",
 "bead_types": {"1": "END", "2": "MID", "3": "END"}}
```

This is n-pentane as CH3-CH2 | CH2 | CH2-CH3: two end beads of one type and a middle bead of another.

- **Bracket atoms with explicit hydrogens:** a map number needs brackets, and brackets switch off implicit hydrogens.
  Write `[CH3:1]`, `[CH2:2]`, `[CH:1]`, `[cH:2]`, `[c:2]` (aromatic carbon without H), `[OH:3]`, `[NH:4]`, `[O:2]`,
  `[N:4]`. Writing `[C:1]` for a CH2 makes a radical: `map check` rejects it.
- **Hydrogens follow their heavy atom.** Never annotate hydrogens.
- **Bead numbers** are arbitrary integers, one per bead. Beads of the same type still get their own numbers.

## Polymers: one repeat unit with two attachment points

Write the repeat unit with `*` at the two ends, head first. The `*` atoms carry no bead number.

```json
{"format": "annotated-smiles",
 "smiles": "*[CH2:1][CH:1](*)[C:2](=[O:2])[O:2][CH3:2]",
 "bead_types": {"1": "BB", "2": "ES"}}
```

This is poly(methyl acrylate): a backbone bead and an ester-side-group bead. `map check` links the tail bead of one
unit to the head bead of the next and lists the bond, angle and dihedral types of the chain.

- **End groups** are not part of the mapping. Their atoms join the nearest bead when the reference is built, and
  the end beads keep the interior types.
- **Tacticity and sequence** belong to the build, not the mapping.

## Mixtures

Mixtures use `{"species": [<molecule>, <molecule>, ...]}` with one entry per component. Bead types can be shared
across species when the chemistry is the same.

## residue-atom-names (the user's own AA topology)

For an AA topology with residue and atom names (GROMACS `.tpr` / `.top`, PSF), beads are given by atom names per
residue:

```json
{"format": "residue-atom-names",
 "residues": {"PNT": [{"type": "END", "atoms": ["C1", "C2"]},
                      {"type": "MID", "atoms": ["C3"]},
                      {"type": "END", "atoms": ["C4", "C5"]}]}}
```

- Use the residue and atom names `ascg ref inspect` lists.
- Hydrogens that are not listed follow their bonded heavy atom.
- A residue name may be a glob (`M*`).
- Polymers need one residue per monomer, with the end residues (other names) listed too.
- `map check` checks only the structure. `ascg ref inspect --mapping` checks the assignment on the real topology:
  beads and masses per molecule type, and bonded types.
- Topologies without names (LAMMPS data files) use the annotated-smiles form, which works for molecules only.

## rdkit-heavy-atom-indices (equivalent)

```json
{"format": "rdkit-heavy-atom-indices", "smiles": "CCCCC",
 "bead_sites": [{"site_id": "E1", "bead_type": "END", "heavy_atom_indices_1based": [1, 2]},
                {"site_id": "M", "bead_type": "MID", "heavy_atom_indices_1based": [3]},
                {"site_id": "E2", "bead_type": "END", "heavy_atom_indices_1based": [4, 5]}]}
```

Indices are 1-based, in the order the atoms appear in the SMILES. In a repeat unit the `*` atoms are counted too.
Counting errors are easy to make, which is why the annotated form is preferred.

## What `map check` reports

- **Errors (block the mapping):**
  - a heavy atom in no bead or in two beads;
  - a radical from a bracket atom without its hydrogens;
  - a bead number with no type;
  - a polymer species without exactly two `*`, or a liquid species with them;
  - more than 16 bead types (the fastMD limit).
- **Per bead:** heavy atoms, mass in amu (hydrogens included), and whether the bead's atoms are bonded to each other.
- **Bonded types:** bonds, angles and dihedrals (combined bending-torsion on every path of three bonds). For
  polymers these are taken over four linked units.
- **Warnings (show them to the user):**
  - beads outside 2-5 heavy atoms;
  - a bead whose atoms are not bonded to each other (its centre of mass can sit between them);
  - one bead type with masses more than 15 % apart (fastMD uses one mass per type);
  - single-bead molecules.
