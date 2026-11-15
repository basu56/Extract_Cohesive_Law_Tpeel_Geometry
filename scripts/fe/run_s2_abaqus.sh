#!/bin/bash -l
# run_s2_abaqus.sh -- S2: new single-COH2D4 ABAQUS runs (mixed mode, viscous, quadrature probe) and
# per-integration-point extraction of these and of the Phase 0 runs (phase0_single_element).
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=4 \
#       --time=01:00:00 --job-name=s2_aba --output=<rundir>/slurm-%j.out scripts/fe/run_s2_abaqus.sh <rundir>
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR host $(hostname)"
B="python3 $REPO/scripts/fe/build_se_deck.py --outdir ."
R="--sigc 20 --Gc 0.4"
for m in s ns; do
  $B --job s2_${m}_lin_a0 $R --alpha 0 --softening linear --mode $m
  $B --job s2_${m}_exp_a3 $R --alpha 3 --softening exponential --mode $m
  $B --job s2_${m}_tab_a3 $R --alpha 3 --softening tabular --mode $m
done
$B --job s2_visc1e-2_exp_a3 $R --alpha 3 --softening exponential --viscosity 1e-2 --T 1
$B --job s2_visc1e-4_exp_a3 $R --alpha 3 --softening exponential --viscosity 1e-4 --T 1
$B --job s2_visc1e-4_rate_exp_a3 $R --alpha 3 --softening exponential --viscosity 1e-4 --rate 1
$B --job s2_probe $R --alpha 0 --probe 1e-4
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for inp in s2_*.inp; do
  j=${inp%.inp}
  if [ -f $j.sta ] && grep -q 'COMPLETED SUCCESSFULLY' $j.sta; then continue; fi
  abaqus job=$j input=$inp cpus=1 interactive > $j.log 2>&1
  printf '%-26s %s\n' $j "$(tail -1 $j.sta 2>/dev/null || echo 'NO .sta')"
done
abaqus python "$REPO/scripts/fe/extract_se_odb.py" s2_*.odb
P0=/scratch/negishi/basu56/tslid_runs/phase0_single_element
for odb in $P0/se_*.odb; do
  b=$(basename $odb .odb); cp $odb ./p0_$b.odb; cp $P0/$b.json ./p0_$b.json
done
abaqus python "$REPO/scripts/fe/extract_se_odb.py" p0_se_*.odb && rm -f p0_se_*.odb
head -3 s2_probe_ip.csv; ls *_ip.csv | wc -l
echo "$(date) done"
