#!/bin/bash -l
# run_i1a.sh -- I1a array: 4 truth cases x 2 windows x 8 Latin-hypercube starts (tasks 0..63) + the 4 truth
# evaluations (tasks 64..67).  One pinned thread per task, deterministic environment (det_env.sh).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 --mem=3G \
#       --time=48:00:00 --array=0-67 --job-name=i1a --output=<rundir>/slurm-%a-%A.out \
#       scripts/inverse/run_i1a.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
T=/scratch/negishi/basu56/tslid_runs
CASES=(phase1a_sigc_sweep/sw_s40 phase1b_alpha_sweep/as_s40_a7 phase1a_sigc_sweep/sw_s21 phase1b_alpha_sweep/as_s21_am3)
WINS=(prepeak full)
i=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$DIR"
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) task $i"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
if [ "$i" -ge 64 ]; then
  c=${CASES[$((i - 64))]}; OPT="--truth --window full"
else
  c=${CASES[$((i / 16))]}; w=${WINS[$(((i / 8) % 2))]}; OPT="--window $w --start $((i % 8))"
fi
$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/inverse/i1a_invert.py --deck "$T/$c.inp" \
    --ref "$T/$c" --out "$DIR" $OPT 2>&1 | grep -v Warn | tail -25
echo "$(date) done"
