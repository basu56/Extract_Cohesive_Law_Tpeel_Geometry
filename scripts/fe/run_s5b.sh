#!/bin/bash -l
# run_s5b.sh -- one s5b_run.py run (forward + stored states + adjoint [+ direct]) in the S5 deterministic
# environment (1 pinned thread, det_env.sh).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 --mem=8G \
#       --time=03:00:00 --job-name=<name> --output=<rundir>/slurm-<tag>-%j.out \
#       scripts/fe/run_s5b.sh <rundir> <deck.inp> <tag> <abaqus rundir/job> [s5b_run.py options]
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; DECK=${2:?deck}; TAG=${3:?tag}; REF=${4:?ref}; shift 4
mkdir -p "$DIR" && cd "$DIR" || exit 1
J=$(basename "$DECK" .inp)
for f in "$DECK" "${DECK%.inp}.json"; do [ -e "$(basename "$f")" ] || ln -s "$f" .; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
/usr/bin/time -f "TIMING $TAG wall %e s maxrss %M KB" $PIN $HOME/.local/bin/uv run --locked --extra fe python \
    scripts/fe/s5b_run.py "$DIR/$J.inp" "$DIR" "$TAG" --ref "$REF" "$@" 2>&1 | grep -v Warn | tail -40
echo "$(date) done"
