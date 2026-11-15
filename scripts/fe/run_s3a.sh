#!/bin/bash -l
# run_s3a.sh -- S3a elastic gate on the production sw_s40 mesh: ABAQUS (default and tight
# controls) + extraction, then tslid.fe (SVK and Hencky), on one compute node.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=32G --time=02:00:00 --job-name=s3a --output=<rundir>/slurm-%j.out \
#       scripts/fe/run_s3a.sh <rundir> [ref json]
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
REF=${2:-/scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/sw_s40.json}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR host $(hostname) cpu $(lscpu | sed -n 's/^Model name: *//p')"
python3 "$REPO/scripts/fe/build_s3a_elastic.py" "$REF" . --n-inc 4 --delta 1.6
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for j in s3a_el_default s3a_el_tight; do
  if [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; then echo "skip $j"; continue; fi
  /usr/bin/time -f "$j wall %e s" abaqus job=$j input=$j.inp cpus=1 interactive > $j.log 2>&1
  tail -1 $j.log; grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" $j.odb
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000001 $j.odb
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000002 $j.odb
done
module purge >/dev/null 2>&1
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
for k in svk hencky; do
  $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3a_elastic.py "$DIR/s3a_el_tight.inp" "$DIR" --kinematics $k
done
echo "$(date) done"
