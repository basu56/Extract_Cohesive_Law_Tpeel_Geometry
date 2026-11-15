#!/bin/bash -l
# run_s5_gate.sh -- S5.0 gate 2: forward results unchanged by "parameters as inputs".
#   (Negishi repo root) OLD=<worktree at the pre-S5 commit> sbatch --account=tomar --partition=cpu \
#       --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 --job-name=s5g_<job> \
#       --output=<rundir>/slurm-gate-<job>-%j.out scripts/fe/run_s5_gate.sh <rundir> <deck.inp>
# One job, one node, one pinned thread, deterministic environment: old code (OLD/src, its driver with
# only the CSV format widened to 17 digits), new code, new code again (determinism).  D24 defaults,
# no ABAQUS-format output (the same for all three).
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
OLD=${OLD:?OLD worktree}
DIR=${1:?rundir}; DECK=${2:?deck}
mkdir -p "$DIR" && cd "$DIR" || exit 1
J=$(basename "$DECK" .inp)
for f in "$DECK" "${DECK%.inp}.json"; do [ -e "$(basename "$f")" ] || ln -s "$f" .; done
cd "$REPO"
source scripts/fe/pin_threads.sh 1
source scripts/fe/det_env.sh
echo "$(date) repo $(git -C "$REPO" rev-parse --short HEAD) old $(git -C "$OLD" rev-parse --short HEAD) host $(hostname)"
export UV_CACHE_DIR=/scratch/negishi/basu56/.uv-cache UV_PYTHON_INSTALL_DIR=/scratch/negishi/basu56/.uv-python
UV="$HOME/.local/bin/uv run --locked --extra fe"
for tag in old new new2; do
  if [ $tag = old ]; then PP="$OLD/src"; DRV="$OLD/scripts/fe/s3_run_tpeel.py"; else PP=""; DRV=scripts/fe/s3_run_tpeel.py; fi
  PYTHONPATH=$PP $UV python -c "import tslid.fe.assembly as a; print('$tag uses', a.__file__)"
  /usr/bin/time -f "TIMING $tag wall %e s" env PYTHONPATH=$PP $PIN $UV python "$DRV" "$DIR/$J.inp" "$DIR" \
      --tag $tag --no-abaqus-out --summary-every 100 | tail -3
done
echo "$(date) done"
