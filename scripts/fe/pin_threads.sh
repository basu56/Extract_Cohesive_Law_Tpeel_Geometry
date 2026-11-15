# pin_threads.sh -- source with a thread count N: sets PIN (a taskset prefix restricted to the first N
# CPUs that SLURM allocated to this job) and MKL/OMP/OpenBLAS thread counts = N.
# Negishi allocates CPUs by memory (16G -> 9, 32G -> 17) and does not confine affinity (0-255), so
# without this a run may use any number of cores.  N empty or 0 -> no pinning, PIN="".
_N=${1:-0}
PIN=""
if [ "$_N" -gt 0 ] 2>/dev/null && [ -n "${SLURM_JOB_ID:-}" ]; then
  _ids=$(scontrol show job -d "$SLURM_JOB_ID" | sed -n 's/.*CPU_IDs=\([^ ]*\).*/\1/p' | head -1)
  _list=()
  IFS=, read -ra _parts <<< "$_ids"
  for _p in "${_parts[@]}"; do
    if [[ $_p == *-* ]]; then for ((_i=${_p%-*}; _i<=${_p#*-}; _i++)); do _list+=("$_i"); done; else _list+=("$_p"); fi
  done
  _sel=$(IFS=,; echo "${_list[*]:0:$_N}")
  PIN="taskset -c $_sel"
  export MKL_NUM_THREADS=$_N OMP_NUM_THREADS=$_N OPENBLAS_NUM_THREADS=$_N
  echo "pinned to CPUs $_sel of $_ids ($_N threads)"
fi
