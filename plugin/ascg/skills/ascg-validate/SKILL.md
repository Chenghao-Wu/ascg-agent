---
name: ascg-validate
description: Interpret held-out validation of a CG model: acceptance gates, outcome classes, chain size, bonded terms, seed spread; check claims independently; handle amendment proposals.
---
# Validation

Validation runs automatically after convergence (3 held-out NPT seeds per state from the mapped-AA start, never used
in fitting). Gates (fixed): |density error| <= 5 % and every pair RDF NRMSE <= 0.15, at every state, for every seed.
Classes: validated (gates and Rg within 10 %), needs expert input (gates pass, Rg outside), failed under budget
(gates missed, or a deterministic stop), unsupported (preflight).

Independent check (reviewer role): fetch with `ascg report task`, and verify every number you repeat against
`report/outcome.json` and `cg/ibi/final/validation.json`; look at seed spread (> 1 % in density deserves a remark),
bonded NRMSE (reported, not gated; > 0.15 worth mentioning), Rg / Ree. A claim without a matching number is not made.

Rule changes: only `ascg amend propose --title --rule-change --evidence --tasks`; the user approves at a terminal;
after approval `ascg amend rescore` re-analyses held-out runs without new sampling and keeps the old outcome. Report
both outcomes.
