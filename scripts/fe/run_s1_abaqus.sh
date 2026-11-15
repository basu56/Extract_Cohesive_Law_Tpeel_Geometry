#!/bin/bash -l
# run_s1_abaqus.sh -- S1 tests 3-4: ABAQUS CPE4I cantilever decks + extraction, and UR3 of the
# T-peel load point from the production run sw_s40 (for the rotation estimate).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=1 \
#       --time=02:00:00 --job-name=s1_aba --output=<rundir>/slurm-%j.out scripts/fe/run_s1_abaqus.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR host $(hostname)"
B="python3 $REPO/scripts/fe/build_cantilever_inp.py --outdir ."
for ny in 2 4 8 20; do
  $B --job lin_ny${ny}_F --ny $ny --load force --P 230
  $B --job lin_ny${ny}_D --ny $ny --load disp --v 0.67
done
$B --job skew_ny4_F --ny 4 --hx 1.0 --h-y0 0 --skew 0.3 --load force --P 230
$B --job nl_ny20_F --ny 20 --load force --P 1400 --nlgeom YES --n-inc 20
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for inp in lin_*.inp skew_*.inp nl_*.inp; do
  j=${inp%.inp}
  if [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; then echo "skip $j"; continue; fi
  /usr/bin/time -f "$j wall %e s" abaqus job=$j input=$inp cpus=1 interactive > $j.log 2>&1
  tail -1 $j.log; grep -h 'SUCCESSFULLY\|NOT BEEN COMPLETED' $j.sta | tail -1
done
abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" lin_*.odb skew_*.odb nl_*.odb
TP=/scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/sw_s40.odb
cp "$TP" tpeel_sw_s40.odb && abaqus python "$REPO/scripts/fe/extract_cantilever_odb.py" --history-node 2000001 tpeel_sw_s40.odb \
  && rm tpeel_sw_s40.odb
ls -la *.csv
echo "$(date) done"
