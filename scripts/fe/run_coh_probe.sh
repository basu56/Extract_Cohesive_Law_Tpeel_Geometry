#!/bin/bash -l
# run_coh_probe.sh -- S3a COH2D4 kinematics probes (scripts/fe/build_coh_probe.py) + extraction,
# and the cohesive-row S/LE of the NLGEOM elastic T-peel runs.
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 \
#       --time=00:30:00 --job-name=cohprobe --output=<rundir>/slurm-%j.out scripts/fe/run_coh_probe.sh <rundir> [tpeel rundir]
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
TP=${2:-}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname)"
python3 "$REPO/scripts/fe/build_coh_probe.py" .
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for inp in pg*.inp; do
  j=${inp%.inp}
  abaqus job=$j input=$inp cpus=1 interactive > $j.log 2>&1
  echo "$j: $(grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1)"
  abaqus python "$REPO/scripts/fe/extract_se_odb.py" $j.odb
  abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" $j.odb
done
if [ -n "$TP" ]; then
  cd "$TP" && abaqus python "$REPO/scripts/fe/extract_tpeel_coh.py" s3a_el16_tight.odb s3a_el_tight.odb
fi
echo "$(date) done"
