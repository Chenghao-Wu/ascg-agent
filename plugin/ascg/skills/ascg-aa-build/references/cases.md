# Topology / force-field cases (AutoPoly v2.0 OPLS-AA, moltemplate, GROMACS conversion)

General patterns: signature -> fix. Systems are deliberately not named; the cases are chemistry-level.

| Case | Signature | Fix |
|---|---|---|
| Packing fails at melt density | the builder cannot place many chains at ~0.9 g/cm3 | dilute cube on a grid (~0.01-0.06 g/cm3), compression in the equilibration |
| Builder-default cut-offs | hard-coded pair style / cut-offs in the generated input | the protocol's cut-off and mixing |
| Unseeded randomness | tacticity / sequence differs between builds | seed per recipe |
| Chemistry outside the typing rules (aliphatic C-F, amide N-H) | untyped atoms | standard OPLS types via a recorded rule patch |
| Carbinol H typed as ether alpha-H | alcohol groups not neutral | alkane H type (OPLS alcohol group: O + HO + C + H sums to 0) |
| CF2 next to CH2 typed with a perfluoro class | no bonded parameters between the classes | CT-class carbon; the small residual moved onto that carbon |
| Groups without a neutral OPLS set (e.g. acetal C, nitrile alpha-CH, chlorinated alkene) | per-chain residual of a few 0.01 e | residual spread over listed carbons (`neutralize-aa`), recorded |
| Ester O-CH2 typed as ether C | -0.05 e per ester; HINT_ESTER_ALKOXY_AS_ETHER | OPLS ethyl-ester C (+0.19) by a higher-priority rule (generic ether rules can outrank it) |
| Aromatic ester carbonyl typed aliphatic | -0.115 e per Ar-C(=O)-O; HINT_ARYL_ESTER_CARBONYL_AS_ALIPHATIC | OPLS methyl-benzoate carbonyl (+0.605); the ring ipso C stays benzene C |
| Nitrile methyl typed as alkane | -0.10 e per CH3-CN; HINT_NITRILE_ALPHA_AS_ALKANE | OPLS CH3CN methyl (-0.08) |
| Diaryl carbonate C: no OPLS type | +0.035 e per carbonate, no hint | aryl-ester carbonyl type with the residual moved onto it (recorded force-field adjustment) |
| Siloxane backbone: OPLS-AA has no D-unit Si nor Si-O torsions | untyped Si / "No bond types defined" | published siloxane force field via `--ff-override` (user approval); OPLS silyl-ether types only to build the structure |
| Small-molecule liquid | 1000 molecules placed on a dilute grid | build as one-unit "polymers"; direct compression in the AA protocol |
| Builder clashes | hundreds to thousands of intra-chain pairs < 1 A | soft-core relaxation (decoupled B state, lambda ramp) before minimization |
| Zero-LJ polar H collapse at high temperature | OPLS H on O / N / S approaches acceptors in hot stages | small LJ (sigma 0.10 nm, epsilon 0.19 kJ/mol; `lmp2gmx.POLAR_H_LJ`) on those H |
