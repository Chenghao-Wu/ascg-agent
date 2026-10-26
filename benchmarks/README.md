# The ascg benchmark, version 1

40 systems for systematic coarse-graining: 33 polymers (homopolymers and copolymers), 6 molecular liquids and one
liquid mixture, each with its CG mapping and its all-atom (AA) build recipe. It is user data for the general ascg
tools (the plugin names none of these systems):

```bash
export ASCG_LIBRARY=$PWD/benchmarks/ascg_library.json      # catalogue search, map show, aa plan find these systems
```

| File | Content |
|---|---|
| `ascg_library.json` | The systems (schema `ascg-library/1`): key, id, name, class, aliases, resolution, the CG mapping, the recipe |
| `aa_recipes.json` | The AA builds for AutoPoly v2.0 (units as complement SMILES, sequences, chain and molecule counts), the state points and the equilibration per class |
| `ff/` | Force-field overrides where OPLS-AA lacks terms (PDMS: OPLS-AA/Qeq of Xiang et al., J. Phys. Chem. B 2025) |

The numerical core's commands (`build-opls`, `cg-topology`, `map-aa`, ...) and the cluster scripts default to these two
files; the system form (`ascg system save`) derives the same kind of recipe and mapping from SMILES for new systems.
Mappings carry the status `benchmark_v1`: defined by the benchmark, confirmed with the user when a new study uses one.

## Conventions

- State points: polymers 400 and 500 K, liquids and the mixture 300 K, all at 1 atm.
- Equilibration: polymers RadonPy's 21-step schedule at the hottest state; liquids direct compression.
- Size: polymers of 10 repeat units (9 for the periodic EVA) with chain counts for about 1000 CG beads (at least 20
  chains); liquids 1000 molecules, the mixture 500 + 500.
- Chain ends: H caps for addition polymers; methyl or explicit end groups where the recipe notes say why (PLA, PCL,
  PVDF, PDMS, BPA-PC, PET).

## Systems

| Id | Key | Name | Class | Resolution | Composition |
|---|---|---|---|---|---|
| B01 | `pe` | Polyethylene | polymer | 1 bead per repeat unit | `E*10`, 100 chains |
| B02 | `pp` | Polypropylene | polymer | 1 bead per repeat unit | `P*10`, 100 chains |
| B03 | `pib` | Polyisobutylene | polymer | 1 bead per repeat unit | `IB*10`, 100 chains |
| B04 | `pa` | Polyacetylene | polymer | 1 bead per repeat unit | `A*10`, 100 chains |
| B05 | `pom` | Polyoxymethylene | polymer | 1 bead per repeat unit | `OM*10`, 100 chains |
| B06 | `peo` | Poly(ethylene oxide) | polymer | 1 bead per repeat unit | `EO*10`, 100 chains |
| B07 | `pvf` | Poly(vinyl fluoride) | polymer | 1 bead per repeat unit | `VF*10`, 100 chains |
| B08 | `pvc` | Poly(vinyl chloride) | polymer | 1 bead per repeat unit | `VC*10`, 100 chains |
| B09 | `pvdf` | Poly(vinylidene fluoride) | polymer | 1 bead per repeat unit | `VDF*10`, 100 chains |
| B10 | `pva` | Poly(vinyl alcohol) | polymer | 1 bead per repeat unit | `VA*10`, 100 chains |
| B11 | `ppo` | Poly(propylene oxide) | polymer | 1 bead per repeat unit | `PO*10`, 100 chains |
| B12 | `pb1` | Poly(1-butene) | polymer | 2 beads per repeat unit | `B1*10`, 50 chains |
| B13 | `p4m1p` | Poly(4-methyl-1-pentene) | polymer | 2 beads per repeat unit | `MP*10`, 50 chains |
| B14 | `pmve` | Poly(methyl vinyl ether) | polymer | 2 beads per repeat unit | `MVE*10`, 50 chains |
| B15 | `pbd` | Poly(1,3-butadiene) | polymer | 2 beads per repeat unit | `BD*10`, 50 chains |
| B16 | `pi` | Polyisoprene | polymer | 2 beads per repeat unit | `IP*10`, 50 chains |
| B17 | `p23dmbd` | Poly(2,3-dimethyl-1,3-butadiene) | polymer | 2 beads per repeat unit | `DMBD*10`, 50 chains |
| B18 | `ps` | Polystyrene | polymer | 2 beads per repeat unit | `S*10`, 50 chains |
| B19 | `pan` | Polyacrylonitrile | polymer | 2 beads per repeat unit | `AN*10`, 50 chains |
| B20 | `pams` | Poly(alpha-methylstyrene) | polymer | 2 beads per repeat unit | `AMS*10`, 50 chains |
| B21 | `pcr` | Polychloroprene | polymer | 2 beads per repeat unit | `CP*10`, 50 chains |
| B22 | `pla` | Poly(lactic acid) | polymer | 2 beads per repeat unit | `LA*10`, 50 chains |
| B23 | `pcl` | Polycaprolactam | polymer | 3 beads per repeat unit | `CL*10`, 33 chains |
| B24 | `sbs` | Poly(styrene-co-butadiene) | polymer | 1 bead per repeat unit | `S S S BD BD BD BD S S S`, 100 chains |
| B25 | `ep` | Poly(ethylene-co-propylene) | polymer | 1 bead per repeat unit | random E / P 50:50, 10 units, 100 chains |
| B26 | `eva_alt` | Poly(ethylene-co-vinyl alcohol), alternating | polymer | 1 bead per repeat unit | `E VA E VA E VA E VA E VA`, 100 chains |
| B27 | `eva_periodic` | Poly(ethylene-co-vinyl alcohol), periodic | polymer | 1 bead per repeat unit | `E E VA E E VA E E VA`, 111 chains |
| B28 | `pmma` | Poly(methyl methacrylate) | polymer | 2 beads per repeat unit | `MMA*10`, 50 chains |
| B29 | `pdms` | Polydimethylsiloxane (PDMS) | polymer | 1 bead per repeat unit | `DMS*10`, 100 chains |
| B30 | `pvac` | Poly(vinyl acetate) (PVAc) | polymer | 2 beads per repeat unit | `VAC*10`, 50 chains |
| B31 | `pnba` | Poly(n-butyl acrylate) (PnBA) | polymer | 4 beads per repeat unit | `NBA*10`, 25 chains |
| B32 | `bpa_pc` | Bisphenol-A polycarbonate (BPA-PC) | polymer | 4 beads per repeat unit | `BPC*10`, 25 chains |
| B33 | `pet` | Poly(ethylene terephthalate) (PET) | polymer | 4 beads per repeat unit | `ETT*10`, 25 chains |
| B34 | `n_hexane` | n-Hexane | liquid | 3 beads per molecule | 1000 x `CCCCCC` |
| B35 | `cyclohexane` | Cyclohexane | liquid | 3 beads per molecule | 1000 x `C1CCCCC1` |
| B36 | `toluene` | Toluene | liquid | 2 beads per molecule | 1000 x `Cc1ccccc1` |
| B37 | `acetone` | Acetone | liquid | 3 beads per molecule | 1000 x `CC(C)=O` |
| B38 | `ethanol` | Ethanol | liquid | 2 beads per molecule | 1000 x `CCO` |
| B39 | `acetonitrile` | Acetonitrile | liquid | 2 beads per molecule | 1000 x `CC#N` |
| B40 | `n_hexane_toluene_mixture` | n-Hexane + toluene binary liquid mixture | liquid mixture | 3 + 2 beads per molecule | 500 x `CCCCCC` + 500 x `Cc1ccccc1` |
