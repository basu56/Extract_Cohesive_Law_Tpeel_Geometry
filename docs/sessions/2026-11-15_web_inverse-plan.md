# 2026-11-15 — Claude Web — S5b/S5c review, inverse-phase instructions

## Context / goal
Review S5b/S5c (2a7b3d8) and plan the phase after S5.

## What was discussed or done
- S5b/S5c accepted (D30): Fisher option (c), eigenvalue uncertainty about 10 % with a stable
  weak direction; a diagnostic window ending at the first full failure; D29 updated; the
  piecewise adjoint at 4.1e-10 accepted; cost rerun with refinement.
- Interpretation [I]:
  - within the α family, all of (σ_c, G_c, α) are locally identifiable at s = 1e-3 P_peak (α the
    weakest, ± 0.05–0.15);
  - the hardest-to-detect change in (σ_c, α) does not keep the peak fixed (slope 0.07 vs 0.11);
  - this is local and holds only within the three-parameter family.
- Next phase designed:
  - I1, parametric inversion baseline (cross-solver bias prediction, Monte Carlo against
    Cramér–Rao, mesh-mismatch truth, bilinear model-form error);
  - I2, piecewise-law identifiability map (support parameter, L² metric, spectrum and
    identifiable modes, twin laws with the full model, mode count against ℓ_cz/H, extra
    observable).

## Results
None new.

## Decisions
Inverse phase I1–I2 design (agreed).

## Failed / aborted / not done
None.

## Next steps and open questions
- Claude Code: after the D30 close-out review, I1a → stop → I1b–I1d → stop → I2 → stop.
- Open: crack length as a differentiable observable (not available yet).
- Open: the twin-law maximization (optional in I2b).
- Later: neural law and Hao et al.'s reproduction, informed by the I2 map.
