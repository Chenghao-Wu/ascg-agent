# Writing a system down

A system is one JSON object: the species as SMILES, plus size and chain ends. `ascg system check --spec '<json>'`
checks it and reports what will be built; `ascg system save` records it in a study.

## Polymer (homopolymer)

```json
{"system_class": "polymer",
 "species": [{"name": "MA", "smiles": "*[CH2:1][CH:1](*)[C:2](=[O:2])[O:2][CH3:2]",
              "bead_types": {"1": "BB", "2": "EST"}}],
 "repeat_units": 10, "tacticity": "atactic", "end_groups": "H"}
```

- The repeat unit has two `*`, the head first: unit i's second `*` bonds to unit i+1's first `*`.
- Bead numbers (the mapping, as in `map check`) are optional; with them the system also gives the mapping, and the
  chain count follows from the bead target. Annotate every heavy atom or none.
- Size: `"chains": N`, or `"target_cg_beads"` (default 1000; at least 20 chains).
- `"tacticity"`: atactic (default), isotactic or syndiotactic. SMILES stereo marks are ignored.
- Chain ends: the start unit is the repeat unit with its head `*` capped, the end unit with its tail `*` capped.
  - `"end_groups": "H"` (default) suits addition (vinyl) polymers: CH3-CH(X)- ... -CH2-CH2X.
  - `"CH3"`, or per end `{"head": "H", "tail": "CH3"}`, e.g. where OPLS-AA has no neutral type for the H-capped end
    (a -CF2H end).
  - Condensation polymers need real end groups: give the end units as SMILES with one `*`, e.g. a polyester with
    methyl ester ends for the repeat unit `*OCC(=O)*`: `"first": "COCC(=O)[*]", "last": "[*]OCC(=O)OC"`. `system check` warns when an H cap
    would make an aldehyde (acyl end) or a silane (silyl end).
- Copolymers: one species per monomer (each with its bead numbers), and the sequence:
  - `"sequence": "A*5 B*5"` (block), `"A B A B A B"` (alternating) or any explicit order;
  - `"random": {"A": 0.5, "B": 0.5}` with `"repeat_units"`: a new random sequence per chain.

  `system check` lists the bonded types the sequence creates (block junctions, alternation) and warns when a type
  occurs fewer than 100 times in the system. Graft, branched and gradient polymers are not covered (AutoPoly builds
  linear chains).

## Liquids and mixtures

```json
{"system_class": "liquid",
 "species": [{"name": "pentane", "smiles": "[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]",
              "bead_types": {"1": "END", "2": "MID", "3": "END"}}],
 "molecules": 1000}
```

```json
{"system_class": "liquid_mixture",
 "species": [{"name": "pentane", "smiles": "CCCCC", "fraction": 0.5},
             {"name": "benzene", "smiles": "c1ccccc1", "fraction": 0.5}],
 "molecules": 1000}
```

A `"count"` per species replaces the fractions.

## What `system check` reports

- errors (block saving): bad SMILES, wrong number of `*`, bead numbers on some species only, unpaired electrons,
  missing counts, mapping errors;
- warnings (tell the user): acyl / silyl H caps, elements OPLS-AA may lack (Si, P, ...: a published force field may be
  needed), net charges, short chains, fewer than 20 chains, very large systems;
- species: formula, atoms per chain or molecule, mass; the start / middle / end units AutoPoly will use;
- size: chains or molecules, atoms, mass, CG beads, the box edge at 1 g/cm3;
- with bead numbers: the mapping check (bead masses, bond / angle / dihedral types, warnings).
