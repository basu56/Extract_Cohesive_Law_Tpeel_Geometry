#!/bin/bash -l
# run_s3c_ladder.sh -- S3c gain ladder on the coarse deck, all configurations in ONE job (same node,
# same pinned CPU), so the per-item gains are comparable.  1 thread (ABAQUS ran with cpus=1).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=16G --time=04:00:00 --job-name=s3c_ladder --output=<rundir>/slurm-ladder-%j.out \
#       scripts/fe/run_s3c_ladder.sh <rundir> <suffix>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; SUF=${2:?suffix}
SRC=/scratch/negishi/basu56/tslid_runs/solver_S3b_coarse
mkdir -p "$DIR" && cd "$DIR" || exit 1
for f in c_s40.inp c_s40.json c_s40_response.csv c_s40_profiles.npz c_s40.msg c_s40.sta; do [ -e $f ] || ln -s $SRC/$f .; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$PIN $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py $DIR/c_s40.inp $DIR --no-abaqus-out"
E="--newton energy --assembly element"
$R --tag L0_base_$SUF                                                     | tail -3   # S3b path
$R --tag L1_en_$SUF   --newton energy                                     | tail -3   # item 2: jitted q-energy, Armijo
$R --tag L2_div_$SUF  --newton energy --diverge-after 4                   | tail -3   # item 2b: divergence check
$R --tag L3_el_$SUF   $E --diverge-after 4                                | tail -3   # item 3: element assembly
$R --tag L4_elc_$SUF  $E --condense --diverge-after 4                     | tail -3   # item 4: IM condensation
$R --tag L5_elcp_$SUF $E --condense --diverge-after 4 --linsolve pardiso_sym | tail -3 # item 5: PARDISO (sym, reuse)
$R --tag L5u_elp_$SUF $E --diverge-after 4 --linsolve pardiso_sym           | tail -3 # item 5 without condensation
echo "$(date) done"
