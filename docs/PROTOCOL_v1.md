# Protocol v1: fitting and validating one CG model

Protocol v1 fixes every numerical setting, budget, acceptance gate and outcome rule of a fit. It is implemented by
`ascg_core.ibi.pipeline` (the `PROTOCOL` dictionary), and the ascg tools run it unchanged. A study locks the protocol by
its hash when it starts. The rules change only through an amendment that a person approves (section 7). Agents may
choose only among the bounded recovery actions of section 7.

## 1. Inputs and preflight

- **Inputs:** a mapped all-atom reference for every approved state point. It holds the CG topology, the target
  distributions, the mapped trajectory, the start configurations and a summary (density, compressibility, Rg, Ree).
- **Preflight** (`pipeline preflight`, deterministic). It checks that:
  - every reference file is present;
  - every bond is in the exclusion list;
  - there are at most 16 bead types (fastMD's limit);
  - the targets cover every pair, bond and angle type at every state;
  - the start configurations match the topology;
  - the AA densities are present;
  - the RDF range is within half the box;
  - dihedrals can be enumerated.

  A failed preflight gives the outcome **unsupported**.
- **Engine gate** (`python -m ascg_core.ibi.selfcheck`, on the host that runs the fit). It checks that:
  - the fastMD energy of a toy melt equals a numpy reference with 1-2 and 1-3 exclusions;
  - the CBT dihedral energy equals its numpy reference;
  - an NPT run of a small CG liquid keeps its volume and pressure stable.

  A failed engine gate is an infrastructure error: the job stops and no outcome is recorded.

## 2. Model

- **Interactions:** tabulated potentials for
  - pairs: cut-off 1.5 nm, C2 switch from 1.3 nm, a repulsive wall below the sampled range;
  - bonds and angles: walls continue each table outside the AA-sampled range;
  - dihedrals: combined bending-torsion (CBT), E = sin³θ₁ sin³θ₂ U(φ), on every path of three bonds. They are on by
    default; `--dihedrals none` turns them off.
- **Exclusions:** 1-2 and 1-3 (the 1-3 pairs as zero-energy bonds); 1-4 pairs interact through the pair table.
- **Temperature dependence:** every table is U(x, T) = A(x) + B(x) T.
  - Two temperatures: exact at both.
  - One temperature (one state, or states that differ only in pressure): B = 0.
  - More temperatures: a least-squares fit weighted by `--weights` (equal by default).
  - `--model shared` fits one potential for all states instead.

## 3. State points

A fit uses any set of N ≥ 1 state points (T, P) that the user approves, for example `400,500`, `450`, `400,400@1000`
or `400@1013.25bar`. A 1 atm state is labelled by its temperature (`T400`); other pressures are labelled `T400_P1000`
(atm).

Every state needs its own mapped AA reference, and every check and gate below applies at every state. The NPT
checks, the held-out validation and the pressure correction use each state's pressure. GROMACS takes pressures in bar
(bar = 1.01325 × atm).

## 4. Fitting (multistate IBI on fastMD)

- **Start:** Boltzmann inversion of each state's AA distributions. Dihedral tables are scaled for the CBT prefactor and
  centred.
- **Each iteration:**
  - an NVT run at the AA mean box for every state (the states run concurrently);
  - equilibration of 100 ps at iteration 0 and 50 ps after that, then 500 ps of production with frames every 1 ps;
  - each run continues from the previous configuration.
- **Updates:**
  - damping λ = 0.2, log-ratio cap 2, support mask at 10⁻³ of the maximum;
  - the per-state corrections are combined into A and B with the state weights.
- **Stage 1** (iterations 1-15): pair tables only, no pressure correction.
- **Stage 2:**
  - schedule pair, pair, pair, bond, angle, dihedral, cycling from iteration 1;
  - runs to iteration 42, rounded up to a whole cycle, so the fit ends on a complete cycle.
- **Pressure correction** (stage 2, every iteration):
  - a ramp A_s (1 − r/r_c) S(r) per state;
  - its amplitude comes from the first-order virial response of the previous run's nonbonded pair distances;
  - it cancels 0.8 of the deviation from the state pressure plus the predicted pressure change of the same
    iteration's pair update;
  - |A_s| ≤ 0.3 k_BT per iteration.
- **In-loop NPT checks:**
  - every 6 iterations (at the end of each cycle) and at the last iteration: 100 ps equilibration + 500 ps;
  - they are density diagnostics. Bond violations in these runs are recorded but do not stop the fit.
- **Bond gate:**
  - a bond's break length is where its table rises 50 k_BT above its minimum;
  - a bond is broken when it is beyond its break length in two consecutive analysed frames, or beyond twice its break
    length in any frame;
  - the gate applies to every NVT fitting run and every held-out validation run;
  - a broken bond stops the task (a deterministic stop).

## 5. Convergence and budget

- **Convergence check** at the end of stage 2: at every state, the in-loop NPT density is within 4 % of AA and the
  worst pair RDF NRMSE is ≤ 0.15.
- **If the check fails:** one extension of 18 iterations (3 cycles), then validation regardless. At most 60 IBI
  iterations.
- **Infrastructure errors** (engine crash, GPU, files): up to 2 retries per stage, resuming after the last complete
  iteration.

## 6. Held-out validation and outcome

- **Runs:** 3 NPT seeds per state, started from the mapped-AA configuration, which the fit never uses. Each run is
  200 ps of equilibration + 1 ns of production.
- **Gates** (every seed, every state): |density error| ≤ 5 %, and every pair RDF NRMSE ≤ 0.15.
- **Reported, not gated:** bond, angle and dihedral NRMSE and JSD; Rg; Ree.
- **Outcome classes:**

  | Outcome | When |
  |---|---|
  | **validated** | the gates pass and the mean Rg is within 10 % of AA at every state |
  | **needs expert input** | the gates pass but Rg is outside the 10 % band |
  | **failed under budget** | the gates are missed after the budget, or a deterministic stop (e.g. a broken bond) |
  | **unsupported** | the preflight failed |

## 7. Decisions and amendments

- **Who does what:** the tools do all numerics. An agent acts only on a task that is awaiting a decision or stopped
  on a gate, with `ascg run act`.
- **Actions:**

  | Action | Parameters |
  |---|---|
  | `extend` | 1-3 cycles |
  | `extend_lower_damping` | cycles; λ 0.05-0.15 |
  | `rewind_smaller_timestep` | iteration to rewind to; time step × 0.5-0.9 |
  | `rewind_lower_damping` | iteration to rewind to; λ 0.05-0.15 |
  | `restart_smaller_timestep` | time step × 0.5-0.9; only when the bond gate stopped the run before iteration 0 completed |

- **Budget:** at most 3 actions and 36 extra iterations per task. Beyond that, or when no action fits, the agent
  escalates to the user.
- **Out of reach for agents:** gates, targets, validation settings and the protocol itself.
- **Amendments:**
  - a rule change is proposed with evidence (`ascg amend propose`);
  - only a person approves it, at a terminal (`ascg amend approve`);
  - after approval, finished tasks can be re-scored from their existing held-out runs without new sampling
    (`ascg amend rescore`).

### Refinements within v1

The rules above are the current ones. Three refinements were made while the benchmark was fitted:

| Version | Rule |
|---|---|
| v1.1 | A bond counts as broken only if it persists (two consecutive analysed frames) or is beyond twice its break length. |
| v1.2 | In-loop NPT checks record bond violations instead of stopping the task. |
| v1.3 | The break length is where the bond table rises 50 k_BT above its minimum (it was 100 kJ/mol). |

v1.1 and v1.2 only loosen the earlier rules. v1.3 is looser at every temperature above 241 K.
