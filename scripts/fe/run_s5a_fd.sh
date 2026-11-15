#!/bin/bash -l
# run_s5a_fd.sh -- S5a finite-difference step study on the coarse T-peel: theta_j +/- h, h = 1e-3 ... 1e-7,
# rtol 1e-10, replaying the schedule of the base run (run_s5a_coarse.sh, b1).  One parameter per array task
# (0..3 = ln_K0, ln_sigc, ln_Gc, alpha); one pinned thread, deterministic environment.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 --mem=8G \
#       --time=04:00:00 --array=0-3 --job-name=s5a_fd --output=<rundir>/slurm-fd-%a-%j.out \
#       [--dependency=afterok:<coarse job>] scripts/fe/run_s5a_fd.sh <rundir> <job>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; J=${2:?job}
NAMES=(ln_K0 ln_sigc ln_Gc alpha)
P=${NAMES[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) param $P"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py $DIR/$J.inp $DIR --no-abaqus-out --summary-every 100 --rtol 1e-10 --schedule $DIR/${J}_b1_sens.npz"
for h in 1e-3 1e-4 1e-5 1e-6 1e-7; do
  for s in p m; do
    v=$h; [ $s = m ] && v=-$h
    /usr/bin/time -f "TIMING fd_${P}_${h}_$s wall %e s" $R --tag "fd_${P}_${h}_$s" --perturb "$P:$v" | tail -1
  done
done
echo "$(date) done"
