#!/bin/bash -l
# run_s3a_scale.sh -- S3a: elastic NLGEOM T-peel in 16 increments (delta = 0.1 ... 1.6 mm), tight
# ABAQUS controls, to measure how the tslid.fe - ABAQUS difference scales with load (strain), for
# SVK and (closed-form) Hencky; plus cohesive-row S/E for the interface-kinematics check.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=32G --time=03:00:00 --job-name=s3a_scale --output=<rundir>/slurm-scale-%j.out \
#       scripts/fe/run_s3a_scale.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
REF=/scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/sw_s40.json
cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
python3 "$REPO/scripts/fe/build_s3a_elastic.py" "$REF" . --n-inc 16 --delta 1.6 --nlgeom YES --name s3a_el16
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
j=s3a_el16_tight
if ! { [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; }; then
  /usr/bin/time -f "$j wall %e s" abaqus job=$j input=$j.inp cpus=1 interactive > $j.log 2>&1
fi
tail -1 $j.log; grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1
abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" $j.odb
abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000001 $j.odb
abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000002 $j.odb
abaqus python "$REPO/scripts/fe/extract_tpeel_coh.py" $j.odb s3a_el_tight.odb s3a_lin_tight.odb
module purge >/dev/null 2>&1
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
for k in svk hencky; do
  $HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3a_elastic.py "$DIR/$j.inp" "$DIR" --kinematics $k --tag el16_$k
done
echo "$(date) done"
