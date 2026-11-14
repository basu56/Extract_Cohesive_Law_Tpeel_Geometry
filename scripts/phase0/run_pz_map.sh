#!/bin/bash -l
# Phase 0 pilot: process-zone length map l_pz(Gc, sigc) for the linear-softening law (alpha = 0),
# to set the lower sigc bound from l_pz <= c H (DECISIONS.md D12) and to see where the mesh
# under-resolves the process zone (upper sigc end, n_el_pz small).
#   (from the repo root) sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=16 \
#          --time=08:00:00 --job-name=tsl_pz --output=<rundir>/slurm-%j.out \
#          scripts/phase0/run_pz_map.sh <rundir>
# Settings: D11 geometry (H 6.324, a0 18, t 0.4, clamp/load, elastic), K0 8500, delta 3 mm,
# 1 mm/min, viscosity 1e-3 s, h_coh 0.1 mm (campaign03 defaults). sigc capped by E2 at
# sqrt(K Gc) = 29.2 / 46.1 / 58.3 MPa for Gc = 0.1 / 0.25 / 0.4.
set -u
REPO=${TSLID_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
DIR=${1:?rundir}
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "repo $(git -C "$REPO" rev-parse --short HEAD) dir $DIR"
GEN="python3 $REPO/src/tslid/build_tpeel_inp.py --outdir . --law alpha --alpha 0 --delta 3.0"
for Gc in 0.1 0.25 0.4; do
  for s in 5 10 15 20 25 30 40 50 57; do
    $GEN --job "pz_g${Gc/./p}_s${s}" --Gc $Gc --sigc $s 2>&1 | grep -E '^wrote|violated' \
      || true
  done
done
ls pz_*.inp | wc -l
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
bash "$REPO/scripts/slurm/run_abaqus_dir.sh" "$DIR" 'pz_*' 1
for odb in pz_*.odb; do
  abaqus python "$REPO/src/tslid/extract_tpeel_odb.py" "$odb" > /dev/null 2>&1 || echo "extract failed: $odb"
done
module load anaconda/2024.10-py312 >/dev/null 2>&1
python3 "$REPO/src/tslid/analysis/energy_checks.py" "$DIR" --pattern 'pz_*'
