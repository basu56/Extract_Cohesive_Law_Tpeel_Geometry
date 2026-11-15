#!/usr/bin/env bash
# Claude Web 2026-11-15: S5 instructions, S5 design decision, session log.
# Run from the GitHub-connected clone on Bell:
#   cd /scratch/bell/basu56/Extract_Cohesive_Law_Tpeel_Geometry && bash apply_solver_S5.sh
set -euo pipefail
[ -f CLAUDE.md ] && [ -f docs/CONTEXT_PROTOCOL.md ] || { echo "Run from the repo root"; exit 1; }
git pull --ff-only origin main
grep -q "S5 design — direct sensitivities" docs/DECISIONS.md && { echo "Already applied (S5 design present)"; exit 1; }
[ -e docs/solver/S5_INSTRUCTIONS.md ] && { echo "docs/solver/S5_INSTRUCTIONS.md exists; not overwriting"; exit 1; }
# next free decision number
N=$(( $(grep -oE "^## [0-9-]+ — D[0-9]+" docs/DECISIONS.md | grep -oE "D[0-9]+$" | tr -d D | sort -n | tail -1) + 1 ))
echo "S5 design decision will be D$N"
mkdir -p docs/solver docs/sessions
cat > docs/solver/S5_INSTRUCTIONS.md <<'CLAUDE_WEB_EOF_0'
# Differentiable solver — Claude Code instructions for S5 (sensitivities and adjoint)

Written by Claude Web, 2026-11-15. Session log: `docs/sessions/2026-11-15_web_S5-plan.md`.
Decision: the "S5 design" entry in `DECISIONS.md`.

Read first: `CLAUDE.md`, `docs/STATUS.md`, `docs/SOLVER_LOG.md` (S3b–S3e), `docs/solver/TPEEL.md`.
The project-wide rules of `S0_S2_INSTRUCTIONS.md` still apply: units, x64, no ABAQUS-side
changes, gradient safety, [D]/[H]/[I] labels, tatva pinned and unmodified, `sbatch` for heavy
jobs.

**Start S5 only after S4a has been reviewed.**

## Goal

Make the T-peel response differentiable with respect to the cohesive-law parameters,
**through the whole loading history**. Then use the derivatives for the first identifiability
result.

This is the step the solver was built for. Every derivative must be checked against an
independent method before it is used.

We build **two independent methods** and check each against the other:

- **Direct (forward) sensitivities (S5a).** For a few parameters (the α family:
  σ_c, G_c, α, and K0), propagate dq/dθ forward through the increments. Each increment costs one
  extra right-hand side per parameter on the factorization that is already available.
  - This gives the full sensitivity matrix S = dy/dθ (every output, every increment), which is
    exactly what the misfit-curvature (Fisher) and Gauss–Newton analyses need.
  - It needs no storage of the history.
- **Discrete adjoint (S5b).** The gradient of one scalar objective J at a cost independent of
  the number of parameters. Needed later for neural or tabular laws with tens to hundreds of
  parameters (Hao et al.).
  - It needs the converged states stored, or checkpointed.

Both methods give the derivative of the **discrete** response as it is computed: the realized
increment schedule, the converged states and the monolithic incremental potential. They do not
give the derivative of the continuous problem.

Follow the order S5.0 → S5a → stop for review → S5b → S5c → stop for review.

## Notation and the discrete equations

Increments n = 1 … N with prescribed end loads λ_n (displacement control; λ_n does not depend
on θ).

- q_n: converged free unknowns. This includes the incompatible-mode amplitudes and the
  reference-point unknowns (`TPeelModel.free`).
- h_n: history after increment n, h_n = (dmax_n, Dv_n) at the cohesive quadrature points
  (`zero_history` / `update_history`).
- Equilibrium (monolithic, D22): R_n(q_n, h_{n−1}, θ) = 0, the gradient of the incremental
  potential at frozen history (`frozen_args`) with the viscous weights (a_n, b_n) of Δt_n.
- Tangent: K_n = ∂R_n/∂q_n, the exact Hessian of the incremental potential. It is
  **symmetric**, so the adjoint system has the same matrix and the PARDISO-sym factorization
  can be reused.
- History update: h_n = H_n(q_n, h_{n−1}, θ) (S2 `update_damage`: dmax_n = max(dmax_{n−1}, δ_m(q_n)),
  then the backward-Euler D_v).
- Outputs: y_n = Y_n(q_n, h_{n−1}, θ). In S5 these are P_n (RF2 at RP1) and the mouth openings
  (MOUTH_U, MOUTH_L U2). P_n depends on θ explicitly through the energy at the prescribed degree
  of freedom, not only through q_n.
- Objective (S5b): J = Σ_n φ_n(y_n). Data on another δ grid are interpolated onto our δ_n. The
  weights do not depend on θ under displacement control.

**Direct sensitivities (S5a).** With S^q_n = dq_n/dθ and S^h_n = dh_n/dθ, and S^h_0 = 0:

- K_n S^q_n = −(∂R_n/∂h · S^h_{n−1} + ∂R_n/∂θ)
- S^h_n = ∂H_n/∂q · S^q_n + ∂H_n/∂h · S^h_{n−1} + ∂H_n/∂θ
- dy_n/dθ = ∂Y_n/∂q · S^q_n + ∂Y_n/∂h · S^h_{n−1} + ∂Y_n/∂θ

**Discrete adjoint (S5b).** Backward n = N … 1, with η_N = 0 and g_n = ∂φ_n/∂y_n:

- K_nᵀ μ_n = (∂Y_n/∂q)ᵀ g_n + (∂H_n/∂q)ᵀ η_n
- η_{n−1} = (∂Y_n/∂h)ᵀ g_n + (∂H_n/∂h)ᵀ η_n − (∂R_n/∂h)ᵀ μ_n
- dJ/dθ += (∂Y_n/∂θ)ᵀ g_n + (∂H_n/∂θ)ᵀ η_n − (∂R_n/∂θ)ᵀ μ_n

These are standard; Hao, Han & Bahmani (arXiv 2610.33711) Eqs. 57–62 use the same structure.
**[H]: re-derive them for our exact R, H, Y**, check the signs, and write the derivation in
`docs/solver/ADJOINT.md`.

Compute every partial with JAX: `jvp` for the direct method and `vjp` for the adjoint, applied
to the per-increment functions R, H and Y. **Do not autodiff the Newton loop or the driver.**
Only converged states enter. Cutbacks, line search, abandoned attempts and the Newton tolerance
affect the derivative only through where the converged states sit.

## Rules specific to S5

- **Frozen schedule.** Derivatives are taken on the realized increment schedule of the forward
  run. Every finite-difference check reruns the perturbed problems on **that same schedule**,
  using the `schedule=` replay of `incremental.run` (S3c). Without replay, the adaptive cutbacks
  make finite differences noisy.
- **Loading–unloading kink.** max(dmax_{n−1}, δ_m) has a kink.
  - Use the derivative of the active branch: loading if δ_m(q_n) > dmax_{n−1}, else unloading
    or no change.
  - Record how many quadrature points per increment sit within a relative 1e-8 of the switch.
  - If finite differences straddle a switch, the check is invalid there. Report it, don't hide it.
  - D = 1 (fully failed) has zero derivative; confirm that this is NaN-free.
- **Tolerance.** The derivative formulas assume exact equilibrium. Verification uses tight
  replays (rtol 1e-10). Separately, quantify the derivative error at the production tolerance
  (D24, rtol 1e-6) against tight: ‖S(1e-6) − S(1e-10)‖/‖S(1e-10)‖.
- **Parameters as inputs.** The law parameters must be explicit, differentiable arguments of the
  production path, not read from `self.law` inside the energy. The production path is the D24
  element-assembly path with incompatible-mode condensation.
- **Scaling.** Report sensitivities with respect to log parameters (d/d ln σ_c, d/d ln G_c,
  d/d ln K0) and with respect to α. Use the same convention for the curvature analysis.

## S5.0 — Parameters as differentiable inputs

1. Pass the law parameters as a pytree argument through:
   - the incremental potential;
   - residual and tangent in the element-assembly path;
   - `update_history`;
   - the outputs.

   Do this for `AlphaLawParams` now. For the many-parameter test in S5b, either use
   `TabularLawParams`, if its incremental potential is closed-form and differentiable in the
   table values, or add a piecewise-linear traction envelope with knot tractions as parameters
   and a closed-form envelope integral. Choose, and document why.
2. **Gate:** the forward results for the five S3 reference cases are unchanged (P histories
   identical to ≤ 1e-14 relative), and the timing is unchanged within noise.
3. **Gate:** the per-increment partials ∂R/∂θ, ∂H/∂θ and ∂Y/∂θ match central finite
   differences at frozen q and h to ≤ 1e-7 relative, at points on loading, unloading, softening,
   fully failed and compressive quadrature points.

## S5a — Direct sensitivities

1. **Implement** `src/tslid/fe/sensitivity.py`: the forward recursion above.
   - Use the same factorization as the converged Newton step: one factorization, p right-hand
     sides.
   - Return S_y = dy_n/dθ for all n, with condensation in the linear solve exactly as in Newton.
   - Run it during the forward solve, after each converged increment. No storage is needed.
2. **Tests, in this order:**
   1. **Single cohesive element** (S2 setting, displacement-driven, with and without viscosity).
      dP/dθ from the recursion equals the derivative of the analytic traction history computed
      by AD of `laws_jax` along the same path, to ≤ 1e-10. This includes the cyclic
      unload/reload runs.
   2. **Two-block test** (S2): ≤ 1e-9 against central finite differences with schedule replay.
   3. **Coarse T-peel**, σ_c = 40, α = 0, θ = (ln K0, ln σ_c, ln G_c, α), against central finite
      differences with schedule replay at rtol 1e-10.
      - Do a step study: relative steps from 1e-3 to 1e-7.
      - **Gate:** curve-wise ‖S_dir − S_FD‖₂ / ‖S_FD‖₂ ≤ **1e-5** per parameter, over the
        compare window.
      - Report point-wise errors and any kink-straddling increments.
3. **Cost** (coarse and production): the overhead of the sensitivities relative to the forward
   run, on 1 thread, pinned, in one job. The expectation [H] is a small fraction, because the
   factorization is reused.
4. **Production `sw_s40`:** compute S_y for P with 4 parameters at rtol 1e-6. Compare against
   tight-tolerance sensitivities on the coarse mesh for the tolerance effect. Save
   `results/solver/S5_sensitivity/s5a_sw_s40_S.npz`.

**Stop for review after S5a**, with a SOLVER_LOG entry. Its "Gain" is exact sensitivities of the
whole P–δ curve, at a cost to be measured. Compare that cost with ABAQUS finite differences,
which need at least 2 runs × 4 parameters at about 38 min each.

## S5b — Discrete adjoint

1. **Implement** `src/tslid/fe/adjoint.py`: the backward recursion above.
   - Storage: keep (q_n, h_{n−1}, a_n, b_n, λ_n) per increment, in memory or on disk. At
     production size this is about 1.3 MB per increment, about 1.1 GB per run [H]; measure it.
   - Recompute K_n during the backward sweep from the stored state, or keep factorizations if
     memory allows. Benchmark both.
   - Add uniform checkpointing (store every k-th state and recompute forward in between) as an
     option. Implement it only if storage turns out to be a problem.
2. **Objective for the tests:** J = ½ Σ_n w_n (P_n − P_n^data)², where the data are the ABAQUS
   record of the same case interpolated onto our δ_n, with w_n the trapezoid weights in δ.
3. **Tests:**
   1. **Adjoint vs direct (S5a)** on the coarse T-peel: dJ/dθ_adj = S_Pᵀ W (P − P^data) to
      ≤ **1e-10** relative. Same converged states, two independent linear-algebra paths.
   2. **Adjoint vs finite differences of J**, with replay: ≤ 1e-6 relative.
   3. **Many-parameter test:** the S5.0 tabular or piecewise law with 20–50 parameters on the
      coarse T-peel.
      - The adjoint gradient against the direct method run with all parameters (affordable on
        coarse) to ≤ 1e-10.
      - Report how adjoint cost scales with p against direct cost. The expectation [H] is that
        the adjoint is flat in p and the direct method linear in p.
4. **Cost:** adjoint wall time relative to the forward run, and peak memory, at coarse and
   production size.

## S5c — First identifiability result: misfit curvature in parameter space

1. **Fisher matrix on `sw_s40`** (production, from S5a). F = S_Pᵀ W S_P / s², where s is a
   load-noise level.
   - Report for s = 1e-3 P_peak and 1e-4 P_peak (the 0.04 % floor is about 4e-4). State that
     these are assumed noise levels, not measurements.
   - Report the eigenvalues and eigenvectors of F, using the log-parameter scaling of the rules
     above, for θ = (ln σ_c, ln G_c, α), and separately with ln K0 added.
   - Condition number, and Cramér–Rao standard deviations, with the assumptions written out.
2. **Comparison with the brute-force ABAQUS result (D18).**
   - From the matched-peak (σ_c, α) pairs (S4a output, computed in both solvers), estimate the
     trade-off direction d(ln σ_c)/dα along which the peak is preserved.
   - Compare it with the weakest eigenvector of the (ln σ_c, α) block of F, and with the
     direction implied by ∂P_peak/∂θ from S5a.
   - Report the angle between the directions. [H] They should be close. If they aren't, report
     it and look for the reason before interpreting anything.
3. **Where the information sits along δ.** Plot the columns of S_P (d ln σ_c, α, d ln G_c)
   against δ with the P–δ curve. Mark the pre-peak window, δ ≈ 0.1–0.22 mm, where D18's
   matched pairs differ.
4. **Independent cross-check.** Finite-difference slopes from the existing sweeps (the
   σ_c and α runs around σ_c = 40, α = 0, in both solvers) against S5a's dP/dθ at matching δ.
   This is coarse (large steps), so expect agreement only to first order. Report it, not as a
   gate.
5. **Outputs:** `results/solver/S5_sensitivity/` with README, CSVs and figure `s5c.png`.
   SOLVER_LOG entries for S5b and S5c.

**Stop for review after S5c.**

## Exit criteria for S5

- S5.0 forward results are unchanged, and the partials are verified.
- S5a matches finite differences (≤ 1e-5, coarse, replay), with cost measured.
- S5b matches S5a to ≤ 1e-10 and finite differences to ≤ 1e-6, with cost and memory measured;
  the many-parameter scaling is reported.
- S5c Fisher analysis is done, and its weak direction is compared against D18's matched-pair
  direction.
- `docs/solver/ADJOINT.md` is written. SOLVER_LOG has entries S5.0–S5c, with the summary row
  updated.

## Out of scope for S5

- Any optimization or inversion: that comes after S5 (the parametric baseline, then Hao et al.).
- Neural laws.
- Noise models beyond the assumed s levels.
- Batching or GPU (S4b).
- Derivatives with respect to the increment schedule, mesh or geometry.
- Any change to ABAQUS-side code or to D16, D24 and D25.
CLAUDE_WEB_EOF_0
cat > docs/sessions/2026-11-15_web_S5-plan.md <<'CLAUDE_WEB_EOF_1'
# 2026-11-15 — Claude Web — S3d/S3e review, S5 instructions

## Context / goal
Review S3d/S3e (commit 55eb375) and write the S5 instructions.

## What was discussed or done
- S3d/S3e accepted. All five cases pass; the largest window difference is 3.6e-4 P_peak
  (`sw_s58`, the shortest zone).
- Same-node timing: 0.23× ABAQUS on 1 thread, 0.17× on 4. Thinning ABAQUS's output saves 5 %.
- Energy-check window: Claude Web recommended evaluating E4–E6 over the load-comparison window,
  as an option in `energy_checks.py` with the default unchanged (sent to Claude Code as "D25";
  numbered by Claude Code).
- S4 split: S4a (21-run sweep reproduction plus matched-pair check) now; S4b (batching, GPU,
  increment strategy) after S5.
- S5 design:
  - two independent derivative methods, direct and adjoint, checked against each other;
  - finite-difference checks with schedule replay;
  - Fisher analysis against D18.
- Key design point [I]: for the α family (4 parameters) the direct method is cheaper and gives
  the full Jacobian needed for Fisher and Gauss–Newton. The adjoint is needed for
  many-parameter laws.

## Results
None new.

## Decisions
S5 design (agreed).

## Failed / aborted / not done
None.

## Next steps and open questions
- Claude Code: finish S4a → review → S5.0/S5a → review → S5b/S5c → review.
- Open: whether storage of the production adjoint (~1 GB per run, estimated) needs
  checkpointing.
CLAUDE_WEB_EOF_1
cat >> docs/DECISIONS.md <<'CLAUDE_WEB_EOF_2'

## 2026-11-15 — D__N__: S5 design — direct sensitivities and discrete adjoint
Decision (user, on Claude Web's plan):
- Differentiate the **discrete** T-peel response (realized schedule, converged states,
  monolithic incremental potential) with respect to the law parameters, by **two independent
  methods**:
  - **(a) direct (forward) sensitivities.** p extra right-hand sides per increment on the reused
    factorization. Gives the full Jacobian dy/dθ for the α family.
  - **(b) discrete adjoint.** Backward sweep with history costates; cost independent of p;
    needed for tabular and neural laws.
- (a) is verified against finite differences with schedule replay; (b) against (a) to roundoff.
- Per-increment partials come from JAX jvp/vjp. Newton and the driver are never autodiffed.
- Loading–unloading kinks: derivative of the active branch, with near-switch points counted and
  reported.
- First result (S5c): Fisher matrix on `sw_s40`, its weak direction compared with D18's
  matched-pair direction.
Order: S5.0 (parameters as inputs) → S5a → review → S5b → S5c → review.
Instructions: `docs/solver/S5_INSTRUCTIONS.md`. S5 starts after the S4a review.
Status: **agreed**.
CLAUDE_WEB_EOF_2
cat >> docs/STATUS.md <<'CLAUDE_WEB_EOF_3'

## Update — Claude Web, 2026-11-15 (S5 plan)
- S5 instructions are in `docs/solver/S5_INSTRUCTIONS.md` (decision "S5 design").
- **Start S5 only after the S4a review.** Order: S5.0 → S5a → stop → S5b → S5c → stop.
CLAUDE_WEB_EOF_3
sed -i "s/D__N__/D$N/" docs/DECISIONS.md
sed -i "s/the \"S5 design\" entry in \`DECISIONS.md\`/D$N (S5 design)/" docs/solver/S5_INSTRUCTIONS.md
git add docs/solver/S5_INSTRUCTIONS.md docs/sessions/2026-11-15_web_S5-plan.md docs/DECISIONS.md docs/STATUS.md
git commit -m "S5 instructions (D$N): direct sensitivities + discrete adjoint, FD with schedule replay, Fisher vs D18"
git push origin main
if git remote | grep -qx negishi; then git push negishi main; fi
echo "Done: $(git rev-parse --short HEAD) (decision D$N)"
