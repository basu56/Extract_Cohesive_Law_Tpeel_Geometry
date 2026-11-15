#!/bin/bash -l
# run_s3b_coarse.sh -- S3b robustness: coarse T-peel (sigma_c = 40, alpha = 0, D16 load settings),
# full step, tslid.fe monolithic and staggered, plus ABAQUS on the same coarse deck.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --mem=16G --time=04:00:00 --job-name=s3b_coarse --output=<rundir>/slurm-%j.out \
#       scripts/fe/run_s3b_coarse.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) cpu $(lscpu | sed -n 's/^Model name: *//p')"
COARSE="--ny 4 --h-y0 0.5 --h-coh 0.5 --h-far 1.5 --h-leg 1.5"
python3 "$REPO/src/tslid/build_tpeel_inp.py" --outdir . --law alpha --alpha 0 --Gc 0.4 --delta 1.6 \
    --viscosity 1e-4 --n-inc 800 --sigc 40 --job c_s40 $COARSE
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
j=c_s40
if ! { [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; }; then
  /usr/bin/time -f "$j wall %e s" abaqus job=$j input=$j.inp cpus=1 interactive > $j.log 2>&1
fi
grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1
abaqus python "$REPO/src/tslid/extract_tpeel_odb.py" $j.odb > /dev/null 2>&1 || echo "extract failed"
module purge >/dev/null 2>&1
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
R="$HOME/.local/bin/uv run --locked --extra fe python scripts/fe/s3_run_tpeel.py"
$R "$DIR/c_s40.inp" "$DIR" --history monolithic | tail -40
$R "$DIR/c_s40.inp" "$DIR" --history staggered | tail -40
echo "$(date) done"
