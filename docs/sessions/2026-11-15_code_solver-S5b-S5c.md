# 2026-11-15 — Claude Code — S5.0/S5a review (D29), S5b, S5c

## Context / goal
User accepted S5.0/S5a with decisions (D29), including an observation operator; S5b (adjoint) and S5c
(Fisher) to do, then stop for review.

## What was done
- D29 recorded; S5a note on kink-straddled increments (none excluded).
- Observation operator: bin = one 0.1 mm element passage, 0.003 mm, from the crack-tip record.
- Adjoint, piecewise-linear law, iterative refinement of the derivative solves; tests.
- Negishi jobs:
  - 45945542 (production rtol 1e-10);
  - 45947320–45947322 and 45950446/7 (S5b);
  - 45950706/7 (production on a shifted and a finer schedule).
- Analyses: s5b_analysis, s5c_fisher, s5c_width_study.

## Results
SOLVER_LOG S5b, S5c; `results/solver/S5_sensitivity/README.md`.

## Decisions
D29 (user).

## Failed / aborted / not done
- The rtol 1e-10 production run turned out to have the same schedule as rtol 1e-6 inside the window
  (both at dtmax), so it is not a schedule-stability test. Two replayed schedules were added.
- Fisher eigenvalue stability ≤ ~1 %: not met (4–15 %).
- Piecewise-law adjoint vs direct 4.1e-10 (gate 1e-10).
- Adjoint cost measured without the refinement solves.

## Next steps and open questions
User: the Fisher stability route (finer increments, wider bins, or accept); then the parametric
inversion baseline.
