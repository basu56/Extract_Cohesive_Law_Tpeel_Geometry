#!/bin/bash -l
# run_s5a_cost.sh -- S5a cost of the direct sensitivities and the production sensitivities of sw_s40.
# One job, one node, one pinned thread, deterministic environment, D24 defaults (rtol 1e-6), no
# ABAQUS-format output: coarse forward / with sensitivities (twice each, alternating), production
# forward / with sensitivities.  The production --sens run gives <job>_sens_sens.npz (S5a item 4).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 --mem=8G \
#       --time=03:00:00 --job-name=s5a_cost --output=<rundir>/slurm-cost-%j.out \
#       scripts/fe/run_s5a_cost.sh <rundir> <coarse deck.inp> <production deck.inp>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; CD=${2:?coarse deck}; PD=${3:?production deck}
mkdir -p "$DIR" && cd "$DIR" || exit 1
for d in "$CD" "$PD"; do for f in "$d" "${d%.inp}.json"; do [ -e "$(basename "$f")" ] || ln -s "$f" .; done; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
run() { local deck=$1 tag=$2; shift 2; /usr/bin/time -f "TIMING $(basename $deck .inp)_$tag wall %e s" \
    $PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py "$DIR/$(basename $deck)" "$DIR" \
    --no-abaqus-out --summary-every 100 --tag "$tag" "$@" | tail -1; }
run "$CD" fwd1
run "$CD" sens1 --sens
run "$CD" fwd2
run "$CD" sens2 --sens
run "$PD" fwd
run "$PD" sens --sens
echo "$(date) done"
