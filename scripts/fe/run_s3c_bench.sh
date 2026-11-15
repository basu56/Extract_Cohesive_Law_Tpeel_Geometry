#!/bin/bash -l
# run_s3c_bench.sh -- S3c micro-benchmarks (assembly, linear solvers) on the production sw_s40 deck
# and the coarse deck.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=<1|4> \
#       --mem=32G --time=01:00:00 --job-name=s3c_bench --output=<rundir>/slurm-bench-%j.out \
#       scripts/fe/run_s3c_bench.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
mkdir -p "$DIR"
C=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$C OMP_NUM_THREADS=$C
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) cpus $C"
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$HOME/.local/bin/uv run --locked --extra fe --with pypardiso==0.4.7 python scripts/fe/s3c_bench.py"
$R /scratch/negishi/basu56/tslid_runs/solver_S3b_coarse/c_s40.inp "$DIR/bench_coarse_cpu$C.json" --reps 5
$R /scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/sw_s40.inp "$DIR/bench_sw_s40_cpu$C.json" --reps 3
echo "$(date) done"
