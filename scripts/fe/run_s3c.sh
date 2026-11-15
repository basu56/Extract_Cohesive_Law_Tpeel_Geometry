#!/bin/bash -l
# run_s3c.sh -- S3c runs of tslid.fe on the coarse T-peel deck (sigma_c = 40, alpha = 0, D16 settings).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=16G --time=04:00:00 --job-name=s3c --output=<rundir>/slurm-%j.out \
#       [--export=ALL,THREADS=<n>] \
#       scripts/fe/run_s3c.sh <rundir> <tag> [extra s3_run_tpeel.py options]
# The coarse deck and its ABAQUS run are taken from tslid_runs/solver_S3b_coarse (run_s3b_coarse.sh).
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; TAG=${2:?tag}; shift 2
SRC=/scratch/negishi/basu56/tslid_runs/solver_S3b_coarse
mkdir -p "$DIR" && cd "$DIR" || exit 1
for f in c_s40.inp c_s40.json c_s40_response.csv c_s40.msg c_s40.sta; do [ -e $f ] || ln -s $SRC/$f .; done
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) cpu $(lscpu | sed -n 's/^Model name: *//p')"
cd "$REPO"
source scripts/fe/pin_threads.sh "${THREADS:-0}"      # THREADS=<n>: pin to n allocated CPUs
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py "$DIR/c_s40.inp" "$DIR" --tag "$TAG" "$@" | tail -60
echo "$(date) done"
