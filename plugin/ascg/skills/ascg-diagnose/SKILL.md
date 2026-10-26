---
name: ascg-diagnose
description: Diagnose a coarse-graining task that stopped on a gate, missed convergence or shows a suspicious result, using the ascg diagnostic tools and the case library; recommend one bounded action, an escalation or a rule-change proposal.
---
# Diagnose a task

Start with `ascg diag digest --study S --task K` (iterations, worst RDF, pressure, NPT density, failure, settings).
Match the signature to the case library (references/cases.md) and confirm it with the listed tool before concluding.

| Signature | Confirm with | Conclusion | Recommend |
|---|---|---|---|
| bond far beyond break length, table end energy low (< ~50 kJ/mol) and end force ~0 | `diag bonds`, `audit tables` | weak bond wall outside the AA range | model fix (harmonic walls - current protocol has them): re-run with the current protocol; escalate if walls are present |
| one bond in one frame just beyond the break length, others far below | `diag bonds` (single_frame_excursions, consecutive_frame_pairs_beyond = 0) | transient excursion, not broken (the current rule already ignores it) | if a stop came from an older rule: propose an amendment / re-score |
| bonds stretched only in an in-loop NPT check at very high pressure, box grows many-fold | `diag bonds --run npt_T<T>` (volume_first_last) | barostat expansion artifact of a diagnostic | not a model failure (current rule records it); rewind if a run was stopped |
| bonds beyond break in consecutive frames, wall intact (hundreds of kJ/mol up the wall) | `diag bonds`, `audit tables` | genuine instability (force spike; stiff bonds, poor bonded structure) | `rewind_smaller_timestep` (dt_factor 0.6-0.75) to an iteration before the spike; escalate if it recurs |
| convergence missed: NPT density off 4-10 %, pressure still falling | `diag digest`, `diag pressure` | density still converging | `extend` 1-3 cycles |
| pressure oscillates or the RDF error grows after pair steps | `diag digest`, `diag structure` | over-correction | `extend_lower_damping` (lam 0.1) |
| validated gates but Rg off by > 10 % | `diag chains`, `aa qa` | reference or model limitation (e.g. state below Tm, intrachain correlations beyond 1-4) | escalate with the evidence; not fixable by IBI actions |
| run blows up near a straight bond angle with plain dihedral tables | `diag structure` (angle near 180 deg) | 1/sin(theta) singularity | use combined bending-torsion (default) |
| fastMD crash, CUDA error | job log, `compute hosts` | infrastructure | already retried; resubmit / tell the user |

Write the diagnosis as: signature (numbers from the tools), confirming evidence, conclusion, one recommendation with
parameters. Never change gates, targets or protocols; if a rule seems wrong, `ascg amend propose` with the evidence.
