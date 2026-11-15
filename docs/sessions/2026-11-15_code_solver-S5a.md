# 2026-11-15 — Claude Code — S4a review, S5.0 and S5a

## Context / goal
User review of S4a (accepted; E6 decision; apply S5 instructions; S5.0 → S5a with deterministic FD,
then stop).

## What was done
- D27 recorded. `apply_solver_S5.sh` applied (D28) and removed.
- S5.0:
  - law parameters as kernel arguments (AlphaLawParams pytree);
  - active-branch dmax;
  - partials tests.
- S5a:
  - `sensitivity.py` (direct recursion, condensed multi-RHS solve);
  - driver `--sens`/`--perturb`, exact-time schedule replay;
  - tests;
  - Negishi jobs 45934469–45934476: forward gate (5 cases, old/new/new), coarse
    determinism/tolerance, FD step study (40 replay runs), cost (coarse + production).
- `docs/solver/ADJOINT.md` (direct and adjoint derivation, signs checked).

## Results
SOLVER_LOG S5.0/S5a; `results/solver/S5_sensitivity/README.md`.

## Decisions
D27 (user), D28 (S5 design, Claude Web, applied).

## Failed / aborted / not done
- S5.0 gate 2 (≤ 1e-14) not met: roundoff-level (≤ 6.5e-12 in the window) differences from
  compiling the law as runtime arguments.
- ln σ_c FD gate marginal (1.02e-5).
- Test-design issues found and fixed during the work:
  - a two-block load step exactly at δ_f;
  - test points where a partial vanishes analytically;
  - an FD reference Newton that was too loose.

## Next steps and open questions
User decisions listed in SOLVER_LOG S5a "Open items"; then S5b (adjoint) and S5c (Fisher).
