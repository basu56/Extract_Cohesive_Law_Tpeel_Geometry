#!/bin/bash -l
# run_fe_tests.sh -- run the solver test suite on a Negishi compute node (S0 cluster check).
# Submit from the Negishi repo root:
#   sbatch --account=tomar --partition=cpu scripts/slurm/run_fe_tests.sh [pytest -m expr]
# Default marker expression is "not slow" (same as CI); pass "" to run everything.
# Also runs tests/test_laws.py under the system python3 (the ABAQUS-side interpreter).
#SBATCH --job-name=fe_tests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=fe_tests_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
MARK=${1-not slow}
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
UV=$HOME/.local/bin/uv
echo "$(date) host=$(hostname) commit=$(git rev-parse --short HEAD) cpu=$(lscpu | sed -n 's/^Model name: *//p')"
echo "== uv env: $($UV run --locked --extra fe python --version)"
$UV run --locked --extra fe pytest tests/test_laws.py tests/fe -m "$MARK" -v --durations=5
echo "== system python: $(/usr/bin/python3 --version 2>&1)"
/usr/bin/python3 -m unittest discover -s tests -p "test_laws.py"
echo "$(date) done"
