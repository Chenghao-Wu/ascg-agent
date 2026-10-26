#!/bin/bash
#SBATCH --job-name=aa-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# AutoPoly v2.0 OPLS-AA build of one task of a recipes file (array index = task), charge neutralization, typing
# audit, and the CG topology when the mappings file has the task. Submit with aa/build.sh. Polymers whose recipe gives
# no chain count get chains = TARGET_BEADS / beads per chain (>= 20); liquids give molecule counts.
set -eo pipefail
source "${ASCG_ROOT:?source cluster/env/slurm.sh before submitting}/cluster/env/slurm.sh"
: "${RECIPES:?}" "${OUT:?}"
MAPPINGS=${MAPPINGS:-$ASCG_ROOT/benchmarks/ascg_library.json}
KEY=$(python -c "import json, sys; print(list(json.load(open(sys.argv[1]))['tasks'])[int(sys.argv[2]) - 1])" \
      "$RECIPES" "$SLURM_ARRAY_TASK_ID")
[ -n "${KEYS:-}" ] && KEY=$(echo "$KEYS" | cut -d+ -f"$SLURM_ARRAY_TASK_ID")      # KEYS=a+b+c (aa/build.sh)
mkdir -p "$OUT"
echo "task $KEY host $(hostname) start $(date)"
SIZE=$(python -c "import json, sys; t = json.load(open(sys.argv[1]))['tasks'][sys.argv[2]]; print('' if 'chains' in t or 'molecules' in t else '--target-beads ${TARGET_BEADS:-1000}')" "$RECIPES" "$KEY")
python -m ascg_core build-opls --recipes "$RECIPES" --mappings "$MAPPINGS" --task-key "$KEY" --out "$OUT" --seed 42 $SIZE > "$OUT/$KEY.build.json"
python -m ascg_core neutralize-aa --recipes "$RECIPES" --system-dir "$OUT/$KEY" --task-key "$KEY" > "$OUT/$KEY.neutral.json"
python -m ascg_core audit-opls --data "$OUT/$KEY/system.neutral.data" > "$OUT/$KEY.audit.json" || true
if python -c "import json, sys; m = json.load(open(sys.argv[1])); sys.exit(sys.argv[2] not in (m.get('tasks') or {s['key'] for s in m.get('systems', [])}))" "$MAPPINGS" "$KEY"; then
  python -m ascg_core cg-topology --recipes "$RECIPES" --mappings "$MAPPINGS" --system-dir "$OUT/$KEY" --task-key "$KEY" > "$OUT/$KEY.cg.json"
fi
echo "done $(date)"
