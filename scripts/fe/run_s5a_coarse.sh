#!/bin/bash -l
# run_s5a_coarse.sh -- S5a on the coarse T-peel (sigma_c 40, alpha 0; deck c_s40 of S3c), one job,
# one pinned thread, deterministic environment:
#   b1, b2   direct sensitivities, rtol 1e-10, PARDISO (b2 = repeat of b1: bit-identical?)
#   bslu     the same with SuperLU (independent factorization)
#   r0       replay of b1's schedule, unperturbed, no sensitivities (replay reproduces b1?)
#   t6       sensitivities at the production tolerance rtol 1e-6 (own schedule)
#   t6r      rtol 1e-6 replaying b1's schedule (tolerance effect alone)
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 --mem=8G \
#       --time=02:00:00 --job-name=s5a_coarse --output=<rundir>/slurm-coarse-%j.out \
#       scripts/fe/run_s5a_coarse.sh <rundir> <deck.inp>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; DECK=${2:?deck}
mkdir -p "$DIR" && cd "$DIR" || exit 1
J=$(basename "$DECK" .inp)
for f in "$DECK" "${DECK%.inp}.json"; do [ -e "$(basename "$f")" ] || ln -s "$f" .; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py $DIR/$J.inp $DIR --no-abaqus-out --summary-every 100"
run() { local tag=$1; shift; /usr/bin/time -f "TIMING $tag wall %e s" $R --tag "$tag" "$@" | tail -2; }
run b1 --sens --rtol 1e-10
run b2 --sens --rtol 1e-10
run bslu --sens --rtol 1e-10 --linsolve superlu_mmd
run r0 --rtol 1e-10 --schedule "$DIR/${J}_b1_sens.npz"
run t6 --sens --rtol 1e-6
run t6r --sens --rtol 1e-6 --schedule "$DIR/${J}_b1_sens.npz"
echo "$(date) done"
