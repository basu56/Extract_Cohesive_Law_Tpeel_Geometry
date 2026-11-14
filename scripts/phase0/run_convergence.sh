#!/bin/bash -l
# Phase 0, step 5: numerical error floor of P(delta) at fixed Gc = 0.40 N/mm, alpha = 0 (D15).
#   (from the repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=11 \
#          --time=12:00:00 --job-name=tsl_conv --output=<rundir>/slurm-%j.out \
#          scripts/phase0/run_convergence.sh <rundir>
# Varies one numerical setting at a time from the base (h_coh 0.1 mm, n_inc 400, viscosity 1e-3 s),
# at sigma_c = 40 (reference), 58 (shortest l_cz, hardest for the mesh) and 21 (longest l_cz, = c H).
# delta = 1.6 mm: the crack reaches the free end at ~1.4 mm for these laws.
set -u
REPO=${TSLID_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR"
GEN="python3 $REPO/src/tslid/build_tpeel_inp.py --outdir . --law alpha --alpha 0 --Gc 0.4 --delta 1.6"
for s in 40 58 21; do
  $GEN --job cv_s${s}_base  --sigc $s
  $GEN --job cv_s${s}_h050  --sigc $s --h-coh 0.05
done
$GEN --job cv_s40_h025  --sigc 40 --h-coh 0.025
$GEN --job cv_s58_h025  --sigc 58 --h-coh 0.025
$GEN --job cv_s40_n1600 --sigc 40 --n-inc 1600
$GEN --job cv_s40_v1e-4 --sigc 40 --viscosity 1e-4
$GEN --job cv_s40_v1e-5 --sigc 40 --viscosity 1e-5
ls cv_*.inp | wc -l
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
bash "$REPO/scripts/slurm/run_abaqus_dir.sh" "$DIR" 'cv_*' 1
for odb in cv_*.odb; do
  abaqus python "$REPO/src/tslid/extract_tpeel_odb.py" "$odb" > /dev/null 2>&1 || echo "extract failed: $odb"
done
module load anaconda/2024.10-py312 >/dev/null 2>&1
python3 "$REPO/src/tslid/analysis/energy_checks.py" "$DIR" --pattern 'cv_*' --out conv_energy_summary.csv
for s in 40 58 21; do
  J=$(ls cv_s${s}_*_response.csv 2>/dev/null | sed 's/_response.csv//' | paste -sd, -)
  [ -n "$J" ] && python3 "$REPO/src/tslid/analysis/compare_ld.py" "$DIR" --jobs "$J" --ref cv_s${s}_base --out conv_s${s} 2>&1 | grep -v -i warn
done
