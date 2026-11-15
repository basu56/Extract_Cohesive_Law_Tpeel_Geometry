# 2026-11-14 — Claude Code — differentiable solver S0 (environment and skeleton)

## Context / goal
Apply the Claude Web solver plan (D19–D20) and complete S0 of `docs/solver/S0_S2_INSTRUCTIONS.md`.

## What was discussed or done
- The user asked for a separate "Differentiable Solver" folder outside the repo but also a push to the
  repo; these conflict. The user chose the planned in-repo layout (`src/tslid/fe/`, `tests/fe/`, `docs/solver/`)
  and Bell login node + Negishi instead of the laptop for development.
- Ran `apply_solver_S0_S2.sh` (commit ce37d58, pushed to GitHub and Negishi); removed the uploaded
  duplicate folder `Differentiable solvers/` (untracked; contents are now in `docs/`).
- Installed uv 0.9.5 in `~/.local/bin` on Bell and Negishi; cache/Pythons on scratch.
- Built the S0 skeleton, smoke tests, CI, SLURM test script, `docs/solver/ENVIRONMENT.md`.

## Results
- [D] 16/16 tests on Bell login, Negishi node (job 45855196) and GitHub CI; `test_laws` 11/11 on Python 3.6.8.
- Details and timings: `docs/SOLVER_LOG.md` S0 entry. Commits 9962f73, 6b39bea, 3c3fd07.

## Decisions
- None new. Implementation choice (not a Dn): exact Python pin 3.12.12, because the lock resolves
  different jax versions for 3.11 and 3.12 and Negishi picked a different 3.12 patch.

## Failed / aborted / not done
- First `uv sync` on Bell failed transiently (`os error 11`); session interrupted during the retry, which had
  in fact completed.
- First Negishi job 45855040 passed but on Python 3.12.13; superseded after pinning.
- Laptop environment not tested. GPU model/access TBD.

## Next steps and open questions
- S1 on the user's go-ahead.
- GPU access for account `tomar` on Negishi `gpu` (before S4).
