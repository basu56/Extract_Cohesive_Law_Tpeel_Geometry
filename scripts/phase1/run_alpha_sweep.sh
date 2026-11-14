#!/bin/bash -l
# Phase 1b (D15, D17): shape sweep -- vary alpha at fixed Gc = 0.40 N/mm and fixed sigma_c.
#   (from the repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=14 \
#          --time=12:00:00 --job-name=tsl_alpha --output=<rundir>/slurm-%j.out \
#          scripts/phase1/run_alpha_sweep.sh <rundir>
# alpha in {-3, -1.5, 0, 1.5, 3, 5, 7} at sigma_c = 40 (l_cz/H ~ 0.42 at alpha 0) and
# sigma_c = 21 (l_cz/H ~ 1.0 at alpha 0). phi_e does not depend on alpha (0.235 / 0.065).
# Production settings D16: h_coh 0.1 mm, viscosity 1e-4 s, n_inc 800, delta 1.6 mm.
# Reference per sigma_c: alpha = 0 (identical to sw_s40 / sw_s21 of the sigma_c sweep).
set -u
REPO=${TSLID_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR"
ALPHAS="-3 -1.5 0 1.5 3 5 7"
tag() { echo "$1" | sed 's/^-/m/; s/\./p/'; }      # -1.5 -> m1p5
for s in 40 21; do
  for a in $ALPHAS; do
    python3 "$REPO/src/tslid/build_tpeel_inp.py" --outdir . --law alpha --Gc 0.4 --delta 1.6 \
        --viscosity 1e-4 --n-inc 800 --sigc $s --alpha=$a --job as_s${s}_a$(tag $a)
  done
done
ls as_*.inp | wc -l
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
bash "$REPO/scripts/slurm/run_abaqus_dir.sh" "$DIR" 'as_*' 1
for odb in as_*.odb; do
  abaqus python "$REPO/src/tslid/extract_tpeel_odb.py" "$odb" > /dev/null 2>&1 || echo "extract failed: $odb"
done
module load anaconda/2024.10-py312 >/dev/null 2>&1
python3 "$REPO/src/tslid/analysis/energy_checks.py" "$DIR" --pattern 'as_*' --out alpha_energy_summary.csv
for s in 40 21; do
  J=$(for a in $ALPHAS; do j=as_s${s}_a$(tag $a); [ -f ${j}_response.csv ] && echo $j; done | paste -sd, -)
  python3 "$REPO/src/tslid/analysis/compare_ld.py" "$DIR" --jobs "$J" --ref as_s${s}_a0 --label alpha \
      --out gc040_s${s}_alpha 2>&1 | grep -v -i warn
done
