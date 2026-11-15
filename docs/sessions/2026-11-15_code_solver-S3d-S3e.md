# Session 2026-11-15 (Claude Code, Bell login + Negishi) — solver S3d and S3e

**Input (user, S3c review).**
- D24 confirmed, plus a SuperLU fallback with a test; no SuiteSparse.
- Residual check for the condensed step accepted.
- S3d and S3e as instructed, with the D25 timing protocol.
- After S3e: S4a (21 sweep runs vs ABAQUS), then S5; S4b deferred.
- Stop after S3e.

**Done.**
- Logged: D24 agreed, D25, and the instructions update.
- `make_solver` fallback with a test; driver defaults = D24; `compare_solvers.py`.
- S3d `sw_s40`:
  - All gates and the target met (8.6e-5).
  - Toggles: SVK stays, the corotational frame stays.
  - S3a traction difference located at the crack tip.
  - D25 timing: 0.23× (1 thread) and 0.17× (4 threads) of ABAQUS's wall time, in one job.
- S3e: all P gates and the target met in the four extreme cases.
- Energy split corrected; all S3d/S3e runs repeated.
- SOLVER_LOG S3d and S3e entries; S3 summary row; dated correction of the S3c SMT claim; results
  README; STATUS.

**Not done / failed.**
- `as_s21_a7` E5 fails in the `energy_checks.py` window. The cause is the window ending
  mid-collapse; it is explained, `energy_checks.py` is unchanged, and the user decides.
- An exclusive timing node is impossible under the group cap of 64 hp_cpu; 16 dedicated cores
  were used instead.
- The first S3d/S3e runs used the superseded energy split (kept in `v1/`).

**Stopped for user review** (end of S3e).
