#!/bin/bash -l
# run_s3c_prod.sh -- S3c production timing run: tslid.fe final S3c configuration on a production deck.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=32G --time=03:00:00 --export=ALL,THREADS=1 --job-name=s3c_prod \
#       --output=<rundir>/slurm-prod-%j.out scripts/fe/run_s3c_prod.sh <rundir> <deck.inp> <tag> [options]
# Records stream (s3_run_tpeel.py), so a run stopped by the time limit still gives the cost per
# increment up to that point.
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; DECK=${2:?deck}; TAG=${3:?tag}; shift 3
mkdir -p "$DIR" && cd "$DIR" || exit 1
J=$(basename "$DECK" .inp)
for f in "$DECK" "${DECK%.inp}.json"; do [ -e "$(basename "$f")" ] || ln -s "$f" .; done
cd "$REPO"
source scripts/fe/pin_threads.sh "${THREADS:-0}"
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py "$DIR/$J.inp" "$DIR" --tag "$TAG" \
    --newton energy --assembly element --condense --diverge-after 4 --linsolve pardiso_sym --summary-every 20 "$@" | tail -60
echo "$(date) done"
