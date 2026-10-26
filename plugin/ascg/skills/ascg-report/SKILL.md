---
name: ascg-report
description: Report coarse-graining results to the user: per-task summaries, study benchmark table and figure, model files, costs; plain language with numbers only from the tools.
---
# Report

- `ascg report task --study S --task K` fetches outcome, plain-language summary, costs, figure and validation into
  `<study>/tasks/K` (checksummed). For a validated model add `--model-files`: the tables, a data file and a ready
  fastMD NPT input per state, into `<study>/tasks/K/report/model_files`. Tell the user where they are.
- `ascg report study --study S` builds `report/benchmark.md` and `benchmark.png`.
- Tell the user, per task:
  - the class;
  - density error and worst RDF per state;
  - chain size vs AA;
  - GPU-hours against the approved budget;
  - any action or escalation, and why.
- The user summary and the technical report carry the fitting history: each invocation's iterations, schedule,
  pressure correction and timestep, the rewinds, and the timestep of the validation and model files. Quote it when
  the fit was rewound or its timestep changed.
- Quote numbers from the files; never round away a failure; say what was not checked.
