# 2026-11-15 — Claude Code — S5b/S5c review (D30), S5 close-out

## Context / goal
User accepted S5b/S5c with decisions (D30). Remaining: the first-failure Fisher diagnostic, the cost
rerun with refinement, and the S5 close-out.

## What was done
- D30 recorded.
- `scripts/fe/s5c_first_failure.py`: Fisher on the window up to the first D = 1, schedules A, C, D.
- Cost job rerun with the refinement solves (job 45956871); `s5b_cost.csv` updated (the old one is in
  `solver_S5b/v1_cost_norefine/` on Negishi).
- SOLVER_LOG: dated notes under S5b and S5c, including a correction of the S5c cause statement; summary
  row closed; STATUS.

## Results
- First failure at δ = 0.198 mm (after the peak). Window up to it: λ_min stable to 1–4 %, λ_max 7–9 %,
  weak direction ≤ 0.16°. The hypothesised ≤ 1 % was not reached.
- The pre-peak schedule dependence comes from σ_c-front passages (one per 0.0034 mm). The earlier
  attribution to element failures was wrong.
- Adjoint with refinement: production 0.67–0.77× forward.

## Decisions
D30 (user).

## Failed / aborted / not done
- First-failure diagnostic: the expected ≤ 1 % was not met (reported).

## Next steps and open questions
- Instructions from Claude Web: parametric inversion baseline with Monte Carlo noise, then the
  piecewise-law Fisher identifiability map.
- S4b: batching / GPU / increments, factorization reuse, h_coh 0.05 sensitivity mesh study.
