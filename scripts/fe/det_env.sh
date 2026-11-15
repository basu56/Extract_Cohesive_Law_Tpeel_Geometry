# det_env.sh -- source after pin_threads.sh 1: deterministic numerics for S5 (D27).
# One pinned thread (pin_threads.sh), MKL conditional numerical reproducibility (fixed code path, also
# across node types), XLA CPU without the multithreaded Eigen pool.  Bit-identical repeats are shown,
# not assumed (S5a log).
export MKL_CBWR=COMPATIBLE
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
echo "deterministic: MKL_CBWR=$MKL_CBWR XLA_FLAGS=$XLA_FLAGS MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
