#!/bin/bash -l
# run_s1_compare.sh -- S1 comparison (tslid.fe vs ABAQUS) on a compute node, uv env.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --time=01:00:00 --output=<rundir>/compare-%j.out scripts/fe/run_s1_compare.sh <rundir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
echo "$(date) host=$(hostname) commit=$(git rev-parse --short HEAD) cpu=$(lscpu | sed -n 's/^Model name: *//p')"
$HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s1_compare.py "${1:?rundir}" results/solver/S1_bulk
echo "$(date) done"
