#!/bin/bash -l
# run_s3c_tol.sh -- S3c tolerance study + coarse energy gate, final S3c configuration, one job, 1 thread.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=16G --time=04:00:00 --job-name=s3c_tol --output=<rundir>/slurm-tol-%j.out \
#       scripts/fe/run_s3c_tol.sh <rundir>
# 1. reference: rtol 1e-10, ABAQUS-compatible outputs (tag tol_ref)
# 2. per rtol in 1e-9 .. 1e-4: (a) replay of the reference increments (--schedule): pure tolerance
#    effect; (b) own incrementation: practical effect incl. changed cutbacks
# 3. energy_checks.py (E4-E6) on the reference run and on the ABAQUS coarse run
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
SRC=/scratch/negishi/basu56/tslid_runs/solver_S3b_coarse
mkdir -p "$DIR" && cd "$DIR" || exit 1
for f in c_s40.inp c_s40.json c_s40_response.csv c_s40_profiles.npz c_s40.msg c_s40.sta; do [ -e $f ] || ln -s $SRC/$f .; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
F="--newton energy --assembly element --condense --diverge-after 4 --linsolve pardiso_sym"
R="$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py $DIR/c_s40.inp $DIR $F"
$R --tag tol_ref --rtol 1e-10 | tail -3
for r in 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4; do
  $R --tag tol_${r}_rep  --rtol $r --schedule $DIR/c_s40_tol_ref_jax.csv --no-abaqus-out | tail -2
  $R --tag tol_${r}_free --rtol $r --no-abaqus-out | tail -2
done
.venv/bin/python src/tslid/analysis/energy_checks.py $DIR --pattern 'c_s40_tol_ref' --out energy_tol_ref.csv
.venv/bin/python src/tslid/analysis/energy_checks.py $DIR --pattern 'c_s40' --out energy_abaqus.csv
echo "$(date) done"
