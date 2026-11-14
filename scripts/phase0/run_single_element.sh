#!/bin/bash
# Phase 0, step 3: single cohesive-element verification of the alpha-law representations.
#   sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=8 --time=01:00:00 \
#          --job-name=tsl_se --output=<rundir>/slurm-%j.out \
#          scripts/phase0/run_single_element.sh <rundir>
# Reference law: K = 8500 MPa/mm, sigc = 20 MPa, Gc = 0.4 N/mm (phi_e = 0.06);
# a high-strength case Gc = 0.1, sigc = 29 (phi_e = 0.49, near the E2 limit).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
SE="python3 $REPO/src/tslid/single_element.py --outdir ."
echo "repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR"

R="--sigc 20 --Gc 0.4"
$SE --job se_lin_a0   $R --alpha 0  --softening linear
$SE --job se_tab_a0   $R --alpha 0  --softening tabular
for a in 1 3 7; do
  $SE --job se_exp_a$a $R --alpha $a --softening exponential
  $SE --job se_tab_a$a $R --alpha $a --softening tabular
done
$SE --job se_exp_am2  $R --alpha -2 --softening exponential   # does ABAQUS accept alpha < 0?
$SE --job se_tab_am2  $R --alpha -2 --softening tabular
$SE --job se_exp_a3_cyc $R --alpha 3 --softening exponential --cyclic 0.4
$SE --job se_tab_a3_cyc $R --alpha 3 --softening tabular --cyclic 0.4
H="--sigc 29 --Gc 0.1"
$SE --job se_hi_exp_a3 $H --alpha 3 --softening exponential
$SE --job se_hi_tab_a3 $H --alpha 3 --softening tabular

bash "$REPO/scripts/slurm/run_abaqus_dir.sh" "$DIR" 'se_*' 1
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for odb in se_*.odb; do
  abaqus python "$REPO/src/tslid/extract_single_element.py" "$odb"
done
for j in se_*.inp; do
  j=${j%.inp}; printf '%-16s %s\n' "$j" "$(tail -1 $j.sta 2>/dev/null || echo 'NO .sta (datacheck/abort: see .dat)')"
done
