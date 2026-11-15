# 2026-11-14 — Claude Web — α-sweep review, Phase 2 strategy, tatva assessment, S0–S2 instructions

## Context / goal
Review the σ_c and α sweeps; plan Phase 2 (dataset, model, literature, roadmap); choose a
differentiable-solver foundation; write instructions for the first solver steps.

## What was discussed or done
- Literature [D, from the papers]:
  - Gavris & Sun (CMAME 2026): differentiable explicit MPM, MLP law, displacement fields, one simulation per identification.
  - Hao, Han & Bahmani (arXiv 2610.33711): differentiable implicit FE, GSM law (PICNN energy + MNN inverse resistance), discrete adjoint through the damage history.
  - For the DCB with ℓ_cz/H ≈ 0.88 and force-only data, they recover G_c < 0.2 % but the median shape error is ≈ 18 %.
  - Their prior is trained on 9 traction–separation pairs. Their tests use noise-free data on the same mesh used for identification.
  - Neither paper trains on a dataset of structural simulations.
- Roadmap agreed in outline:
  1. finish the forward maps;
  2. classical parametric baseline (Fisher matrix and profile likelihood);
  3. differentiable solver;
  4. reproduce Hao et al.;
  5. our contributions: identifiability maps computed from the adjoint, choosing extra observables, robustness to cross-solver mismatch and noise, bridging laws; neural-operator surrogate later.
- tatva v0.11.6 reviewed (repo and docs examples) [D]. It provides energy-based global AD, sparse colouring, mixed-dimensional coupling and custom elements. It has no solver, no history management, no cohesive element class and no adjoint.
- Prototype on tatva: 2D DCB with a single-step implicit gradient. It matched finite differences to 0.08 % [D, unverified physics]; added as `scripts/fe/prototype_dcb_tatva.py`.
- α-sweep re-analysis (D18).

## Results
- D18 [I]: P_peak ≈ f(ℓ_cz/H) within ≈ 1.2 % for α ∈ [−3, 3]; 1.6–5 % residuals at α ≥ 5.

## Decisions
- D18 proposed; D19, D20 agreed.

## Failed / aborted / not done
- Could not push from Claude Web (no credentials); this script is the delivery.

## Next steps and open questions
- Claude Code: S0 per `docs/solver/S0_S2_INSTRUCTIONS.md`; stop after S0 for review.
- Open: which Purdue cluster/partition offers GPUs (S0 records it).
- Open: COH2D4 quadrature and frame under NLGEOM (S2/S3).
- Open: dedicated test of D18.
