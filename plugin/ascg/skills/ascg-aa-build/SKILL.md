---
name: ascg-aa-build
description: Build and audit the all-atom topology and force field of a polymer, molecular liquid or mixture before any simulation (AutoPoly v2.0 OPLS-AA, a published force field where OPLS-AA lacks terms, or user files): typing errors, net charges, clashes, zero-LJ hydrogens.
---
# Atomistic topology and force field

Plan with `ascg aa plan --task K --study S --states <approved>`: it reads the recipe derived from the study's
system (`ascg system save`, from SMILES; ascg-specify-system), or, without `--study`, the user's system library.
A system needs only the repeat unit or the molecules as SMILES:
- **polymer:** the repeat unit with two `*` (head first); the start and end units are derived by capping it (H by
  default; CH3, or end units given as SMILES for condensation polymers: `references/system.md`);
- **liquid or mixture:** each molecule's SMILES with a count or a mole fraction.

Building and simulating are expensive. Agree chemistry, size, end groups, tacticity and composition with the user
first. Typical sizes are about 1000 CG beads (>= 20 chains), or 1000 molecules for a liquid.

**Audit every build before neutralizing or simulating.** `ascg aa build` runs the build, neutralization and audit as one
job and `ascg aa status` reports the verdict and findings; for a build made elsewhere,
`ascg aa audit --target T --system-dir /abs/build/dir` reports the net charge per molecule and a
type / charge / environment table. OPLS-AA charges sum to zero only over whole neutral groups, so a non-zero net
charge almost always means a mistyped atom. Read the table and the hints:
- `HINT_ESTER_ALKOXY_AS_ETHER`: ester O-CH2 typed as ether C (+0.14 instead of +0.19).
- `HINT_ARYL_ESTER_CARBONYL_AS_ALIPHATIC`: Ar-C(=O)-O carbonyl typed as an aliphatic ester (+0.49 instead of +0.605).
- `HINT_NITRILE_ALPHA_AS_ALKANE`: CH3-CN methyl typed as an alkane (-0.18 instead of -0.08).
- `HINT_UNTYPED`, or the build failing ("No bond types defined"): the force field has no type for that chemistry.

What to do with each finding, in order (references/cases.md):
1. **A standard OPLS type exists:** add a higher-priority typing rule that selects it, in a recorded patch, and
   rebuild. This is a typing correction; report it.
2. **OPLS-AA lacks the chemistry** (e.g. siloxanes):
   - Propose a published, system-specific force field.
   - With the user's approval, apply it by chemical environment after conversion: `prepare-gmx-protocol
     --ff-override <file>.json`; terms it does not define are refused.
3. **A small residual with no type available** (e.g. a diaryl carbonate C):
   - Move it onto listed OPLS types with `neutralize-aa` (`charge_fix_opls_types` in the recipe).
   - This is a force-field adjustment; report it.

Never neutralize away an unexplained residual.

Other audits before simulating:
- packing fails at the target density -> build dilute, compress in the equilibration;
- builder clashes (hundreds of pairs < 1 Angstrom) -> soft-core relaxation before minimization;
- OPLS hydrogens on O / N / S with zero LJ collapse onto acceptors at high temperature -> small LJ on them (user
  decision);
- unseeded tacticity / sequence randomness -> seed per recipe; builder-default cut-offs -> set the protocol's.

Report every fix to the user; none is silent.
