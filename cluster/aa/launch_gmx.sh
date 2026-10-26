#!/bin/bash
# GROMACS AA runs on a Slurm cluster from any approved account (cluster/site.local.sh):
#   bash cluster/aa/launch_gmx.sh RECIPES SYSTEM_ROOT KEY [KEY ...]
# RECIPES gives each task's class, and per class the approved state points ("states") and the equilibration
# ("eq21" for polymers, "direct" for liquids); STATES=... overrides the states (required when the recipes file has
# none: state points are the user's decision, never a default). SYSTEM_ROOT holds the AutoPoly builds
# (aa/build.sh); runs go to RUNBASE/<basename of SYSTEM_ROOT>/KEY/protocol.
# Per task: write the protocol (prepare-gmx-protocol) unless it has started, then queue
#   MODE=cpu (default): the equilibration job, then one job per state branch (afterok), 16 cores each;
#   MODE=gpu: one job running equilibration and branches in turn on AA_GPU_PARTITION;
#   MAP=1: the mapping job after the branches (RUNBASE/cg_ref/KEY; needs a mapping map-aa can apply).
# DRYRUN=1 prints the sbatch commands instead. Rerunning skips tasks with queued jobs.
set -eo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/cluster/env/slurm.sh"
site_account
REC=$(realpath "${1:?recipes file}"); SYSROOT=$(realpath "${2:?system root}"); shift 2
[ $# -gt 0 ] || { echo "usage: $0 RECIPES SYSTEM_ROOT KEY [KEY ...]"; exit 2; }
OUT=$RUNBASE/$(basename "$SYSROOT"); REF=$RUNBASE/cg_ref
RUNNER=$ASCG_ROOT/cluster/aa/gmx_run.slurm
SB="${DRYRUN:+echo} sbatch --parsable"
mkdir -p "$OUT"
c=$(( ${QOS_OFFSET:-0} )); g=0
# next QoS in turn (sets Q; a $(...) subshell would lose the counter and put every job on the first QoS)
nextq() { local q=${AA_QOS[$((c % ${#AA_QOS[@]}))]}; c=$((c+1)); Q="--qos=$q $(qos_account $q)"; }
for KEY in "$@"; do
  read -r CLASS STATES_R EQUIL EQNS FFO <<<"$(python - "$REC" "$KEY" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); t = r['tasks'][sys.argv[2]]; c = t.get('class', 'polymer')
print(c, r.get('states', {}).get(c, '-'), r.get('equilibration', {}).get(c, 'eq21' if c == 'polymer' else 'direct'),
      10 if c == 'polymer' else 4, t.get('ff_override', '-'))
PY
)"
  STATES_K=${STATES:-$STATES_R}
  [ "$STATES_K" != - ] || { echo "$KEY: no approved state points (set STATES=...)"; continue; }
  ROOT=$OUT/$KEY/protocol
  if [ -n "$(squeue -u $USER -h -o %j | grep -E "^(aa|aaext|gmx)-$KEY(-.*)?$")" ]; then echo "$KEY: MD job already queued"; continue; fi
  [ -f $SYSROOT/$KEY/system.neutral.data ] || { echo "$KEY: no built system in $SYSROOT/$KEY"; continue; }
  EQD=$([ "$EQUIL" = eq21 ] && echo eq21 || echo equil)
  if [ ! -f $ROOT/$EQD/em_sc.log ] && [ -z "${DRYRUN:-}" ]; then     # not started: write the current protocol
    rm -rf $ROOT
    FF=""; [ "$FFO" != - ] && FF="--ff-override $ASCG_ROOT/$FFO"
    python -m ascg_core prepare-gmx-protocol --states $STATES_K --equilibration $EQUIL --eq-ns $EQNS $FF \
      --system-dir $SYSROOT/$KEY --out $ROOT > $OUT/$KEY.protocol.json
  fi
  BR=$(python -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['branch_dirs']))" $ROOT/protocol.json 2>/dev/null \
       || echo "$STATES_K" | tr ',' '\n' | python -c "import sys; from ascg_core.ibi.states import parse; print(' '.join('T' + s.label for s in parse(sys.stdin.read())))")
  if [ "${MODE:-cpu}" = gpu ]; then
    q=${AA_GPU_QOS[$((g % ${#AA_GPU_QOS[@]}))]}; g=$((g+1))
    last=$($SB -J aa-$KEY --qos=$q $(qos_account $q) --partition=$AA_GPU_PARTITION --gres=gpu:1 --cpus-per-task=8 \
           --time=${GPU_TIME:-16:00:00} --export=ALL,MAXH=15.5 -o $OUT/$KEY.gmx-%j.out $RUNNER gpu $ROOT/$EQD $(printf "$ROOT/%s " $BR))
    msg="$KEY ($CLASS, $EQUIL, gpu $q): $last"
  else
    nextq
    EQT=$([ "$EQUIL" = eq21 ] && echo 08:00:00 || echo 04:00:00)
    eq=$($SB -J aa-$KEY-$EQD $Q --partition=$AA_CPU_PARTITION --cpus-per-task=16 --time=$EQT --export=ALL,MAXH=7.5 \
         -o $OUT/$KEY.gmx-$EQD-%j.out $RUNNER cpu $ROOT/$EQD)
    msg="$KEY ($CLASS, $EQUIL): $EQD $eq"; last=""
    for B in $BR; do
      nextq
      j=$($SB -J aa-$KEY-$B $Q --partition=$AA_CPU_PARTITION --cpus-per-task=16 --time=2-00:00:00 --export=ALL,MAXH=46 \
          --dependency=afterok:$eq -o $OUT/$KEY.gmx-$B-%j.out $RUNNER cpu $ROOT/$B)
      msg="$msg, $B $j"; last=${last:+$last:}$j
    done
  fi
  if [ "${MAP:-0}" = 1 ]; then
    mkdir -p $REF
    m=$($SB -J aa-$KEY-map --qos=$MAP_QOS $(qos_account $MAP_QOS) --partition=$AA_CPU_PARTITION --time=06:00:00 \
        --dependency=afterok:$last -o $REF/$KEY.map-%j.out $ASCG_ROOT/cluster/aa/map_aa.slurm $KEY $ROOT $REF $SYSROOT $REC)
    msg="$msg, map $m"
  fi
  echo "$msg"
done
