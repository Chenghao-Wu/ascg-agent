#!/bin/bash
#SBATCH --job-name=ascg-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# Protocol v1 for one task (docs/PROTOCOL_v1.md): python -m ascg_core.ibi.pipeline run = preflight, engine
# selfcheck, stage 1 pair x15, stage 2 pair:3,bond:1,angle:1,dihedral:1 + virial pressure correction to 42 (one
# 18-iteration extension if the in-loop NPT density is off by > 4 %), held-out validation (3 seeds per T), figure,
# outcome class, model export and reports in <task>/report. Both states share the job's GPU. Resubmitting resumes.
#   sbatch --qos=... --partition=... ibi_protocol_v1.slurm REF_DIR TASK_DIR      (ASCG_ENV=<env file> exported)
#   workstation: ASCG_ENV=... CUDA_VISIBLE_DEVICES=N setsid nohup bash ibi_protocol_v1.slurm REF_DIR TASK_DIR &
set -eo pipefail
REF=${1:?ref dir}; TASK=${2:?task dir}
source "${ASCG_ENV:?set ASCG_ENV to the host env file}"
echo "protocol v1 $(basename "$REF") host $(hostname) start $(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$TASK"
python -m ascg_core.ibi.pipeline run --ref "$REF" --task-dir "$TASK" --gpus "${GPUS:-inherit}"
echo "done $(date)"
