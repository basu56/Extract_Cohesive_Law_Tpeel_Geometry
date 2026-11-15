#!/bin/bash -l
# run_s3a_kin.sh -- S3a rerun with the COH2D4 kinematics found by the probes (frame auto =
# corotational / corotational_linear): NLGEOM 16-increment elastic deck with SVK and Hencky, and the
# NLGEOM = NO deck with linear kinematics.  ABAQUS runs reused from run_s3a*.sh.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=32G --time=02:00:00 --job-name=s3a_kin --output=<rundir>/slurm-kin-%j.out scripts/fe/run_s3a_kin.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
cd "$REPO"
echo "$(date) repo $(git rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3a_elastic.py"
$R "$DIR/s3a_lin_tight.inp" "$DIR" --kinematics linear --tag co_lin | grep -v newton
for k in svk hencky; do
  $R "$DIR/s3a_el16_tight.inp" "$DIR" --kinematics $k --tag co_el16_$k | grep -v newton
done
echo "$(date) done"
