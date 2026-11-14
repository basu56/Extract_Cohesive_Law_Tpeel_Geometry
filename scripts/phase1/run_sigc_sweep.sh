#!/bin/bash -l
# Phase 1a (D15): sigma_c sweep at fixed Gc = 0.40 N/mm, alpha = 0, within the c = 1 bounds (D12).
#   (from the repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=9 \
#          --time=12:00:00 --job-name=tsl_sigc --output=<rundir>/slurm-%j.out \
#          scripts/phase1/run_sigc_sweep.sh <rundir>
# Production settings from the convergence study (results/phase0/convergence): h_coh 0.1 mm,
# viscosity 1e-4 s (E5 ~ 7e-5), n_inc 800 (finer peak sampling), delta 1.6 mm.
set -u
REPO=${TSLID_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR"
SIGS="21 25 30 35 40 45 50 55 58"
for s in $SIGS; do
  python3 "$REPO/src/tslid/build_tpeel_inp.py" --outdir . --law alpha --alpha 0 --Gc 0.4 --delta 1.6 \
      --viscosity 1e-4 --n-inc 800 --job sw_s${s} --sigc $s
done
ls sw_*.inp | wc -l
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
bash "$REPO/scripts/slurm/run_abaqus_dir.sh" "$DIR" 'sw_*' 1
for odb in sw_*.odb; do
  abaqus python "$REPO/src/tslid/extract_tpeel_odb.py" "$odb" > /dev/null 2>&1 || echo "extract failed: $odb"
done
module load anaconda/2024.10-py312 >/dev/null 2>&1
python3 "$REPO/src/tslid/analysis/energy_checks.py" "$DIR" --pattern 'sw_*' --out sweep_energy_summary.csv
J=$(for s in $SIGS; do [ -f sw_s${s}_response.csv ] && echo sw_s${s}; done | paste -sd, -)
python3 "$REPO/src/tslid/analysis/compare_ld.py" "$DIR" --jobs "$J" --ref sw_s40 --out gc040_sigc_sweep 2>&1 | grep -v -i warn
