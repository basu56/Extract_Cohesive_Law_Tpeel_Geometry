#!/bin/bash -l
# run_s3a_diag.sh -- S3a diagnostics after the elastic gate missed 1e-5: (1) NLGEOM = NO deck vs
# tslid.fe linear kinematics + linearised coupling, (2) NLGEOM = YES at 1 % of the load,
# (3) cohesive-row S/E of the gate decks.  Same mesh (sw_s40).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=32G --time=02:00:00 --job-name=s3a_diag --output=<rundir>/slurm-diag-%j.out \
#       scripts/fe/run_s3a_diag.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
REF=/scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/sw_s40.json
cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
python3 "$REPO/scripts/fe/build_s3a_elastic.py" "$REF" . --n-inc 4 --delta 1.6 --nlgeom NO --name s3a_lin
python3 "$REPO/scripts/fe/build_s3a_elastic.py" "$REF" . --n-inc 4 --delta 0.016 --nlgeom YES --name s3a_small
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for j in s3a_lin_tight s3a_small_tight; do
  if [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; then echo "skip $j"; continue; fi
  /usr/bin/time -f "$j wall %e s" abaqus job=$j input=$j.inp cpus=1 interactive > $j.log 2>&1
  tail -1 $j.log; grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" $j.odb
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000001 $j.odb
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000002 $j.odb
done
abaqus python "$REPO/scripts/fe/extract_tpeel_coh.py" s3a_el_tight.odb s3a_lin_tight.odb s3a_small_tight.odb
module purge >/dev/null 2>&1
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3a_elastic.py"
$R "$DIR/s3a_lin_tight.inp" "$DIR" --kinematics linear --coupling linear --tag lin
$R "$DIR/s3a_small_tight.inp" "$DIR" --kinematics svk --tag small_svk
$R "$DIR/s3a_el_tight.inp" "$DIR" --kinematics svk --tag svk_tr
echo "$(date) done"
