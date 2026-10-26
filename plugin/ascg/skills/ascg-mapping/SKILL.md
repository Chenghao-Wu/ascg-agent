---
name: ascg-mapping
description: Propose, check, review and approve atom-to-bead mappings from the user's description ("two beads per repeat unit", "three beads for this solvent"): resolution, bead types, end groups, bonded types; written as a SMILES mapping, checked with ascg map check, saved and approved. The mapping is always the user's decision.
---
# Mapping

Every number comes from `ascg` commands (one per shell call); read the reference files below with your file-reading
tool. If another shell command is refused, the `ascg` commands still work: use them.

1. **Find an existing mapping first.**
   - System library: `ascg map show --task K` shows the mapping, its format and its approval status. Unapproved or
     reconstructed partitions need the user's confirmation.
   - A study may already hold one: `ascg map show --study S --task K`.
2. **New systems: propose from the chemistry.** Follow the user's resolution ("N beads per repeat unit / per
   molecule"):
   - backbone vs side groups, and functional groups whole;
   - 2-5 heavy atoms per bead;
   - chemically distinct beads get distinct types; end beads normally keep the interior type;
   - fastMD allows <= 16 types.

   When several partitions fit, give each with its trade-offs: beads per molecule, number of types, broken groups.
   General decisions: `references/cases.md`.
3. **Write each candidate down.** For the user's own AA topology with residue names, use `residue-atom-names` and
   check it with `ascg ref inspect --mapping` (ascg-bring-reference). Otherwise write an annotated SMILES. Each heavy atom carries its bead number:
   `[CH3:1][CH2:1][CH2:2]...` with `"bead_types": {"1": "END", ...}`; polymers are one repeat unit with two `*`
   attachment points. The format, with polymer and mixture examples, is in `references/formats.md`.
4. **Check before showing:** `ascg map check --spec '<json>' --system-class polymer|liquid|liquid_mixture`.
   - Fix every error.
   - Show the user the result for each candidate: partition, bead masses, bonded types and every warning.
     Never compute masses or bonded types by hand.
5. **Save the chosen one:** `ascg map save --study S --key K --spec '<json>' --system-class ... --rationale "..."`.
   It stays "proposed". When ascg also builds the AA system, the same annotated SMILES goes into the system instead
   (`ascg system save`, ascg-specify-system): it saves this mapping too.
6. **Approval is the user's.** Only after they explicitly approve this partition:
   `ascg map approve --study S --key K --note "<their words>"`. Without approval, the mapping is not used.
7. **Once a mapped reference is registered,** `ascg map review --study S --task K` audits bead masses, types,
   molecules, bonded topology, 1-2 / 1-3 exclusions and dihedrals. Any review issue blocks fitting until fixed.
