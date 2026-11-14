#!/bin/bash
# run_abaqus_negishi.sh -- Negishi (RCAC) version of scripts/run_abaqus_dir.sh: run every
# <pattern>.inp in a directory, NPAR at a time, with the abaqus/2026 module (Bell's
# /apps/external/abaqus/2025 does not exist on Negishi; 2025 is not installed there).
#   sbatch --account=tomar --partition=cpu --ntasks=1 --cpus-per-task=10 --time=06:00:00 \
#          scripts/stageF/run_abaqus_negishi.sh <rundir> [pattern] [cpus-per-job]
# Each job's stdout goes to <job>.log; a job is skipped if its .sta reports success.
# License: FlexNet 1736@mooring.ecn.purdue.edu (same ECN server as Bell).
module load intel/19.1.3.304 abaqus/2026 >/dev/null 2>&1
ABA=$(command -v abaqus)
[ -x "$ABA" ] || { echo "abaqus/2026 module failed to load"; exit 1; }
DIR=${1:?rundir}
PAT=${2:-*}
CPJ=${3:-1}
NPAR=$(( ${SLURM_CPUS_PER_TASK:-1} / CPJ ))
[ "$NPAR" -lt 1 ] && NPAR=1
cd "$DIR" || exit 1
echo "$(date) host=$(hostname) dir=$DIR pattern=$PAT cpus/job=$CPJ parallel=$NPAR abaqus=$ABA"
for INP in $PAT.inp; do
    JOB="${INP%.inp}"
    if [ -f "$JOB.sta" ] && grep -q 'COMPLETED SUCCESSFULLY' "$JOB.sta"; then
        echo "skip $JOB (done)"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 5; done
    echo "$(date +%T) start $JOB"
    ( "$ABA" job="$JOB" input="$INP" cpus="$CPJ" interactive > "$JOB.log" 2>&1
      echo "$(date +%T) end   $JOB: $(tail -1 "$JOB.sta" 2>/dev/null)" ) &
done
wait
echo "$(date) all done"
