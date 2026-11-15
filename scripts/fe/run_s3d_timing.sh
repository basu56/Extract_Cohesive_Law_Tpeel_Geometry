#!/bin/bash -l
# run_s3d_timing.sh -- D25 timing protocol: one job, one node (exclusive), sequential and pinned:
#   1. ABAQUS, default deck (cpus=1)        2. ABAQUS, field output thinned to tslid.fe's (cpus=1)
#   3. tslid.fe, 1 thread                    4. tslid.fe, 4 threads
#   (Negishi repo root) sbatch --account=tomar --partition=cpu --exclusive --time=06:00:00 \
#       --job-name=s3d_timing --output=<rundir>/slurm-timing-%j.out scripts/fe/run_s3d_timing.sh <rundir> <deck.inp>
# Wall times: /usr/bin/time for every run, plus WALLCLOCK from the ABAQUS .msg.
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DIR=${1:?rundir}; DECK=${2:?deck}
J=$(basename "$DECK" .inp)
mkdir -p "$DIR" && cd "$DIR" || exit 1
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) host $(hostname) cpu $(lscpu | sed -n 's/^Model name: *//p')"
cp "$DECK" tm_abq_default.inp
cp "${DECK%.inp}.json" tm_abq_default.json
python3 "$REPO/scripts/fe/thin_output_deck.py" "$DECK" tm_abq_thin.inp
cp "${DECK%.inp}.json" tm_abq_thin.json
ln -sf "$DECK" "$J.inp"; ln -sf "${DECK%.inp}.json" "$J.json"
source "$REPO/scripts/fe/pin_threads.sh" 1; PIN1=$PIN
source "$REPO/scripts/fe/pin_threads.sh" 4; PIN4=$PIN
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
for j in tm_abq_default tm_abq_thin; do
  rm -f $j.lck
  /usr/bin/time -f "TIMING $j wall %e s" $PIN1 abaqus job=$j cpus=1 interactive > $j.log 2>&1
  grep TIMING $j.log; grep -h "WALLCLOCK\|COMPLETED SUCCESSFULLY\|NOT BEEN COMPLETED" $j.msg $j.sta | tail -2
  du -h $j.odb
done
module purge >/dev/null 2>&1
cd "$REPO"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
for t in 1 4; do
  P=PIN$t
  export MKL_NUM_THREADS=$t OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t
  /usr/bin/time -f "TIMING tslid_t$t wall %e s" ${!P} $HOME/.local/bin/uv run --locked --extra fe python \
      scripts/fe/s3_run_tpeel.py "$DIR/$J.inp" "$DIR" --tag tm_t$t --summary-every 50 2>&1 | grep -E "TIMING|wall_run|\"increments\"|iterations_all|wall_first|wall_setup|t_io"
done
echo "$(date) done"
