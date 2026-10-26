---
name: ascg-specify-system
description: Turn a coarse-graining request into a specification: identify the system (the user's system library or a new system), composition, state points, resolution and budget, and collect the approvals the user must give.
---
# Specify the system

1. **Identify the system.** If the user has a system library configured, `ascg catalogue search --query "<the user's
   words>"` finds its entries by name, key or alias (with their mapping and recipe status).
   - Check that a match is really the same chemistry; trivial names can mislead.
   - Otherwise it is a new system. Ask for:
     - the structure: SMILES, repeat unit or molecule files;
     - composition and architecture: chain length, tacticity or sequence, end groups;
     - for mixtures, the components and mole fractions.
   - The user's own AA trajectories need no build: ascg-bring-reference.
   - A new system that ascg should build is written as a **system**: one SMILES per species, plus size and chain
     ends (`ascg system check`, `references/system.md`). With the bead numbers of the mapping in the same SMILES
     (annotated SMILES), one description gives the AA build and the CG mapping. Defaults: 10 repeat units, atactic,
     H end caps, chains for about 1000 CG beads (at least 20); liquids 1000 molecules. Tell the user every default
     you used; chain length and end groups change the physics.
2. **State points** are the user's decision, never a silent default.
   - Take them from the request, or propose them with a reason (melt above Tm / Tg, the application's pressure) and
     get approval. Liquids usually need one state; transferable polymer models two or more.
   - A state point is a temperature and a pressure: `400` (1 atm), `400@1000` (atm) or `400@1013.25bar`.
   - The engine fits any approved set. The CG potential depends on temperature only, U(x, T) = A(x) + B(x) T;
     pressure sets each state's NPT runs and pressure target.
     - One state is ordinary IBI, valid at that state only.
     - Two states at two temperatures are exact.
     - Anything else is a weighted least-squares compromise, so a state may keep a residual; every gate still applies
       at every state. This covers three or more states, and states that differ only in pressure (which share one
       table set).
   - Tell the user this before they approve states that differ only in pressure.
   - Every approved state needs its own AA reference at that (T, P). Missing references mean new AA simulations; ask
     the user.
3. **Resolution and mapping:** the library's mapping (`ascg map show --task K`, with its approval status), the
   user's own, or a new one proposed from their description ("two beads per repeat unit"). Every mapping is
   written as a SMILES mapping, checked with `ascg map check`, saved with `ascg map save` and approved by the user
   (ascg-mapping). For a system ascg builds, put the bead numbers into the system's SMILES instead:
   `ascg system save` saves the mapping with it and `ascg system approve` approves both.
4. **Budget:** about 0.3-1 GPU-hour per task for the default CG fit of a ~1000-bead polymer on a recent GPU. Small
   molecular liquids, more beads or recoveries can need 1.5 GPU-hours or more. Propose a budget with margin and say
   that the measured projection follows after the first iterations. AA references cost far more.
5. **Proposal.** Write a short one for the user, covering:
   - system and references (reuse or build);
   - mapping;
   - model: tabulated pair, bond, angle and CBT dihedral; U = A + B T over the state points;
   - fitting;
   - acceptance gates: density within 5 % and every pair RDF NRMSE <= 0.15 at every state;
   - compute and time.

   Proceed when the user agrees. For a system ascg builds (in a study: `ascg study init` first):
   `ascg system save --study S --key K --spec '<json>' --rationale "<their request and your defaults>"`, show the result, and after their explicit approval
   `ascg system approve --study S --key K --note "<their words>"`; then `ascg aa plan --task K --study S --states ...`.
