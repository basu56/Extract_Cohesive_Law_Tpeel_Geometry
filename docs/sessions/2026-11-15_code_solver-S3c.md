# Session 2026-11-15 (Claude Code, Bell login + Negishi) — solver S3c

**Input.**
- User decisions on S3a/S3b: accept the NLGEOM floor as D23, with the gate revised to ≤ 5e-5.
- The S3c order: records as the run goes, then profile; line search and overhead; element
  Hessians vs jacfwd; IM condensation; linear solvers with reuse; tolerance and energy gate.
- Project the production time and stop; if > 2× ABAQUS, decide together.

**Done.**
- Logged: D23 and the S3c order (`S3_INSTRUCTIONS.md` update). The staggered S3b run timed out
  (dated note). The S3b cost attribution is corrected (dated note).
- S3c items 1–6, each measured separately (SOLVER_LOG S3c, figure `results/solver/S3_tpeel/s3c.png`).
- New code: `assembly.py`, `linsolve.py`, `tpeel_output.py`, `newton_energy`, the streaming driver,
  CPU pinning, ladder/tolerance/production scripts. Tests: `tests/fe/test_s3c_assembly.py`.
- Production `sw_s40` run in full: 490 s on 1 thread vs ABAQUS 2321 s, with P and energies
  matching. These are preliminary S3d numbers.
- D24 proposed: solver configuration and rtol 1e-6.

**Additions beyond the instructed list (flagged).**
- Divergence check in `newton_energy`. The profile showed that abandoned attempts dominate.
- Exact closed-form inviscid ALLDMD in the output writer.
- Schedule replay for the tolerance study, to separate the tolerance effect from incrementation.
- CPU pinning: Negishi allocates CPUs by memory and has no cpuset.

**Not done / failed.**
- UMFPACK and CHOLMOD not benchmarked (no SuiteSparse; a micromamba install was blocked).
- The first condensation version was slower than no condensation (batched LAPACK on 4 × 4).
  Replaced by a closed-form inverse; both sets of measurements are kept (`s3c_explore_*`).
- The first benchmark jobs ran with 17 unpinned threads (renamed `*_unpinned17`, superseded).

**Stopped for user review** (end of S3c). Next: confirm D24, go-ahead for S3d.
