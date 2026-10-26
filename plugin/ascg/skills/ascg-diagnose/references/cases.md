# CG case library

General patterns: signature, tools, what it was, what fixed it, and what NOT to do. Systems and runs are deliberately
not named; the numbers are typical magnitudes.

## C1 Weak bond wall
- Signature: bonds escape within picoseconds of the Boltzmann-inverted start, usually at the higher temperature;
  density far low (-10 to -20 %) and Rg far high.
  - The bond table ends at a small energy (a few tens of kJ/mol) with almost no force. This is the quadratic
    continuation of a broad or bimodal AA distribution.
  - Beyond the table end fastMD clamps, so there is no restoring force.
- Tools: `diag bonds` (longest far beyond, many bonds, persistent), `audit tables` (table_end_energy_kj small).
- Fix: harmonic walls 1/2 k (b - b0)^2 with k = kT / var outside the AA-sampled range (always on in the current
  protocol). Do not try smaller time steps: the wall, not the integration, is the problem.

## C2 Single-frame excursion of a stiff bond
- Signature: one or two bonds just beyond the break length in one frame, back in the next; other seeds well inside.
- Tools: `diag bonds`: single_frame_excursions > 0, consecutive_frame_pairs_beyond = 0.
- Conclusion: not a broken bond. The current rule (two consecutive frames, or twice the break length) ignores it.
  Stops of this kind under an older rule: amendment + re-score / rewind; report both outcomes.

## C3 Barostat expansion in an in-loop NPT check
- Signature: a model far from 1 atm (thousands of bar) in the in-loop NPT density check. The box grows many-fold and
  bonds are stretched for many frames by the barostat's position rescaling.
- Tools: `diag bonds --run npt_T<label> --iteration N` (volume_first_last).
- Conclusion: diagnostic artifact; the current rule records it as a warning. Not a model failure.

## C4 Genuine instability
- Signature: bonds beyond the break length in consecutive frames, with intact walls (the stretched length is
  hundreds of kJ/mol up the wall).
  - Often a very stiff bond (k ~ 1e5 kJ/mol/nm^2) at the higher temperature.
  - Often early fitting, with high pressure and poor bonded structure (dihedral NRMSE > 0.2).
- Recommended: `rewind_smaller_timestep` to an iteration before the spike, dt_factor 0.6-0.75. If it recurs,
  escalate: it needs a closer look at angle / dihedral forces.
- At iteration 0 (no completed iteration; typical of very stiff bonds on light beads, e.g. methyl groups of small
  molecules, with the automatically chosen timestep): `restart_smaller_timestep`, dt_factor 0.6-0.75.

## C5 Slow density convergence
- Signature: structure converged (worst RDF a few 0.01) but the NPT density stays several % low. Pressure is
  +100-150 bar and falling slowly.
- Cause: a Reith-type correction, whose step shrinks with the pressure error (~0.003 kT per step at 100 bar).
- Recommended: `extend` (2-3 cycles). The virial correction of the current protocol reaches |P| < 100 bar in 3-6
  iterations.

## C6 Chain size off with gates passing
- Signature: Rg off by > 10 % with good local structure and passing density / RDF gates.
  - Often at a state where the AA chains themselves change size strongly between states (`aa qa` flag).
  - Often a state below the melting point, or near the glass transition, of the chains simulated.
  - Often strongly associating groups (hydrogen bonding).
  - Also seen with coarse mappings (one bead per unit).
- CG dihedrals can match P(phi) without fixing Rg.
- Conclusion: reference / model limitation; IBI actions do not fix it. Escalate with the evidence (`aa qa` flag,
  `diag chains`). Options: check the AA reference, intrachain targets, or accept at the other states.

## C7 Collinear dihedral singularity
- Signature: blow-up with a bond angle near 180 deg along a dihedral path; torsion forces scale as 1/sin(theta).
- Fix: combined bending-torsion (sin^3 theta1 sin^3 theta2 U(phi)), the default.

## C8 Pressure saw-tooth after pair steps
- Signature: pressure rises by hundreds of bar after each pair update and falls back after the bond / angle /
  dihedral steps.
- Fix: validate at the end of a full cycle (the protocol ends on the dihedral step).

## C9 Infrastructure
- fastMD crash / CUDA error: retried automatically (twice per stage, resuming after the last complete iteration).
- Job ended without a final status: node failure or kill; check hosts, resubmit.
