# Differentiable solver — Claude Code instructions for S3 (T-peel forward solve vs ABAQUS)

Written by Claude Web, 2026-11-15. Session log: `docs/sessions/2026-11-15_web_S3-plan.md`.
Decision: D21 update. Read `CLAUDE.md`, `docs/STATUS.md`, `docs/SOLVER_LOG.md` (S0–S2 entries),
`docs/solver/ELEMENTS.md` and `docs/solver/INTERFACE.md` first. The project-wide rules of
`S0_S2_INSTRUCTIONS.md` still apply:
- units mm–N–MPa;
- x64 precision;
- no ABAQUS-side changes;
- gradient safety;
- [D]/[H]/[I] labels;
- tatva pinned and unmodified;
- heavy jobs via `sbatch`.

## Goal

Solve the full 2D plane-strain T-peel with `tslid.fe` and reproduce the ABAQUS P–Δ record for
the same law, mesh and settings. Measure honestly what it costs.

S3 is the first point where we can say whether the differentiable solver is (a) correct for
the real problem and (b) competitive in time. Both answers go into `SOLVER_LOG.md`, whatever
they are.

## Settled inputs

**Bulk (D21 update).** Q4IM with **SVK** energy is the default. Hencky is a cross-check,
run once in S3d.

**Interface (S2).** Nodal (Newton–Cotes) quadrature, as in COH2D4. The α law
(`laws_jax`), QUADS with equal strengths, damage driven by the effective separation, no damage
in compression, and viscous regularization with the backward-Euler D_v update already verified
in S2.

**ABAQUS reference runs.** These are already on Negishi scratch, with production settings
D16 (h_coh 0.1, viscosity 1e-4 s, n_inc 800, Δ = 1.6 mm, 1 mm/min):

| role | job | σ_c (MPa) | α | ℓ_cz/H | rundir |
|---|---|---|---|---|---|
| reference | `sw_s40` | 40 | 0 | 0.42 | σ_c sweep |
| extreme, short zone | `sw_s58` | 58 | 0 | 0.17 | σ_c sweep |
| extreme, long zone | `sw_s21` | 21 | 0 | 1.00 | σ_c sweep |
| concave shape | `as_s21_am3` | 21 | −3 | 0.77 | α sweep |
| convex long tail | `as_s21_a7` | 21 | 7 | 2.28 | α sweep |

All use G_c = 0.40 N/mm, K0 = 8500 MPa/mm. Confirm the job names and paths from
`scripts/phase1/` and the rundirs before use. If a reference is missing, report it rather than
substitute another.

## Sub-steps

Write one SOLVER_LOG entry per sub-step. Use the entry template, and fill **Gain** for each
sub-step separately, because the user wants to see what each sub-step bought.

Stop for user review in three places:
- after **S3b**;
- at the end of **S3e**;
- immediately if any gate below fails.

### S3a — T-peel mesh, loading and an elastic check

1. **Import the mesh from the ABAQUS input deck.** Do not rebuild it. Write a dependency-free
   parser in `src/tslid/fe/inp_import.py` for the flat decks that `build_tpeel_inp.py` writes.
   It reads:
   - nodes;
   - CPE4I elements (with orientation checked counter-clockwise);
   - COH2D4 connectivity, mapped to matched bottom/top node pairs on the bond faces y = ±t/2 (geometric thickness 0.4 mm, constitutive thickness 1.0);
   - node sets and surfaces (`RP1`, `RP2`, `S_LEGTOP_U`, `S_LEGTOP_L`, `MOUTH_U`, `MOUTH_L`, and the cohesive element sets);
   - section thickness W.

   Generate the deck with the same `build_tpeel_inp.py` arguments as the reference job (read
   them from `<job>.json`), so the mesh is identical node for node. Unit test: element count,
   node count, total bulk area, and bonded length a0 → L match the deck.

2. **Kinematic couplings.** Each `*Coupling, *Kinematic` makes a leg end face rigid with
   its reference point.
   - Implement this with 3 master unknowns per reference point (u1, u2, θ). The face nodes
     follow finite rigid-body motion: u(X) = u_RP + (R(θ) − I)(X − X_RP).
   - Eliminate the slave dofs; don't use penalties. Use a `tatva` `Lifter` or our own
     reduction, whichever is cleaner; document the choice.
   - Boundary conditions as in the deck: RP2 encastre (`bc-bottom clamp`); RP1 with u2
     prescribed and u1 and θ free (`bc-top load`).
   - The reaction P is RF2 at RP1, the total force in N (thickness W is included).

3. **Elastic gate.** Run one short ABAQUS T-peel with `--law elastic` (no damage, NLGEOM = YES, a few increments) and the same
   case in `tslid.fe`.
   - **Gate:** RF2 at RP1, the mouth openings and the RP1 rotation agree within **1e-5**
     relative.
   - This isolates mesh import, couplings, bulk and the interface elastic stiffness before
     any damage. If it fails, stop and report.

### S3b — Damage and displacement in one Newton solve (incremental potential)

S2 found that the staggered scheme (solve → update history → repeat) needed up to 17 passes
per step. Replace it, as the default, with a monolithic scheme that keeps tatva's energy
structure.

1. **Incremental potential.** At increment n+1, with history (δ_max,n, D_v,n) frozen and
   r = Δt/μ, the cohesive traction of the ABAQUS update (S2's `update_damage`) is radial in the
   effective separation δ_m. It is therefore the gradient of a potential in δ_m:

   Φ_n(δ) = [ (1 − D_v,n) ½ K0 δ_m² + r Ψ_env,n(δ_m) ] / (1 + r) + ½ K0 ⟨−δ_n⟩₊²,

   with the envelope term

   Ψ_env,n(δ_m) = ½ (1 − D_n) K0 min(δ_m, δ_max,n)² + ∫ from δ_max,n to max(δ_m, δ_max,n) of T(s) ds,

   where T(s) = (1 − D(s)) K0 s is the monotonic law. As μ → 0 (r → ∞), Φ reduces to the
   Ortiz–Pandolfi irreversible cohesive potential (Ortiz & Pandolfi 1999, IJNME).

   **[H] — derive this yourself** from the verified `update_damage` and confirm it. Do not
   take it on trust. Requirements:
   - Use closed-form integrals of the α envelope; no numerical quadrature in s. The tabular
     law can come later; it is not needed in S3.
   - Make it gradient-safe at δ_m = δ_max,n, δ_0 and δ_f, and at α = 0.
   - After convergence, update the history with the existing `update_damage`.

2. **Gate.** The monolithic scheme, with **one Newton solve per increment**, must reproduce:
   - all 20 S2 single-element runs (inviscid and viscous) within the S2 tolerances;
   - the S2 two-block test.

   Report Newton iterations per increment against the staggered passes. Keep the staggered
   scheme as an option (`history="staggered"`) for cross-checks.

3. **Newton robustness.** The softening tangent can be indefinite.
   - Add a backtracking line search on the residual norm or the incremental energy, plus
     increment cutback (halve on non-convergence, regrow after easy increments).
   - Mimic ABAQUS's limits: maximum increment Δt_max = T/800, initial increment as in the deck
     (`dt0 = min(dtmax, T/10) · 0.1`), minimum T·1e-9.
   - Log cutbacks.

**Stop after S3b**, with SOLVER_LOG entries for S3a and S3b.

### S3c — Coarse T-peel with damage, profiling, performance

Develop on a **coarse T-peel** locally: the same builder with fewer elements through the
thickness and a coarser h_coh, giving at least 5 elements per ℓ_cz at σ_c = 40. Get it running
end to end, then:

1. **Profile before optimizing.** Report the split of wall time between JIT compilation,
   residual, tangent assembly, linear solve, history update and I/O for one full run. Check
   that nothing recompiles across increments: pass history and load as arguments, with static
   shapes.

2. **Optimize in this order.** Measure the gain of each change separately and log it.
   1. **Static condensation of the incompatible modes.** Keep the IM amplitudes as unknowns
      of the energy, but condense them in the linear solve.
      - The IM block is block-diagonal (4 × 4 per element). Form the Schur complement
        S = K_uu − K_ua K_aa⁻¹ K_au and the condensed residual with batched 4 × 4 solves
        (`vmap`), solve for Δu, then recover Δa.
      - This is exactly Newton on the full system; the global system drops from about 160k
        to about 54k unknowns at production size.
      - Unit test: the condensed and uncondensed solves give the same Newton step to 1e-12.
   2. **Linear solver.** Benchmark SciPy SuperLU (current), PARDISO (`pypardiso`), UMFPACK,
      and CHOLMOD where the matrix is SPD. Prefer one that reuses the symbolic factorization
      between iterations. Report time per solve at coarse and production size. If you add a
      dependency, add it to the `fe` group and lockfile and keep CI green.
   3. **Convergence tolerance.** ABAQUS's default criterion is far looser than our S1/S2
      tolerance (rtol 1e-10). Choose the loosest tolerance that changes P by ≤ 1e-5 of P_peak
      compared with a tight solve, and document it. Do **not** reduce the number of
      increments in S3. That is an S4 question, with its own accuracy check.

3. **Gate (coarse).** Energy balance E4 and dissipation E5/E6 are computed exactly as
   `src/tslid/analysis/energy_checks.py` does for ABAQUS, and pass.

**Update 2026-11-15 (user, after the S3a/S3b review; D23).** S3c runs in this order, logging
the gain of each item separately in SOLVER_LOG:
1. Write records as the run goes (a timed-out run still leaves data), then profile.
2. Reduce line-search residual evaluations and per-increment Python overhead (JIT the Newton
   step, energy-based Armijo acceptance, try the full step first). This is about 61 % of the
   coarse wall time.
3. Benchmark element-local Hessian assembly (batched per-element Hessians, then scatter) against
   the 34-colour `sparse.jacfwd`, with the same matrix to 1e-12.
4. Condense the incompatible modes (item 2.1 above).
5. Benchmark the linear solvers, with symbolic-factorization reuse (item 2.2).
6. Tolerance study (item 2.3) and the coarse energy gate (item 3).
At the end of S3c, project the production wall time from the measurements and stop for review.
If it is more than about 2× ABAQUS, decide together with the user rather than optimise further.
S3d adds one item: report where along the bond the S3a cohesive-traction difference
(2.6–3.0e-4) sits, crack tip vs bond-end corner.

### S3d — Production reference case vs ABAQUS (`sw_s40`)

1. **Run on Negishi** (`sbatch`, CPU partition), on the production mesh imported from the
   `sw_s40` deck, with the same settings as D16.

2. **Write ABAQUS-compatible outputs:** `<job>_response.csv`, `<job>.json` and
   `<job>_profiles.npz`, with the same column names and arrays as
   `src/tslid/extract_tpeel_odb.py` writes. Read that file for the exact schema. The point is
   that `compare_ld.py` and `energy_checks.py` run on our output unchanged.
   - Energies: ALLWK (external work), ALLSE (bulk + interface elastic), ALLDMD (damage
     dissipation) and the viscous dissipation. Put the viscous part where `energy_checks.py`
     expects it; see the S2 note that ABAQUS reports cohesive viscous dissipation via ALLCD,
     not ALLVD.

3. **Cross-solver comparison script:** `src/tslid/analysis/compare_solvers.py`. Interpolate
   both P(δ) records onto a common δ grid and report:
   - whole-curve max |ΔP|/P_peak and rms over 0 ≤ δ ≤ the window end of `compare_ld.py`
     (σ_c front at L − 15 mm);
   - ΔP_peak, Δδ_peak and the elastic slope;
   - ℓ_cz and crack-tip position vs δ;
   - E4–E6 for both solvers.

4. **Gates:**
   - whole-curve max |ΔP| ≤ **0.5 % of P_peak**;
   - |ΔP_peak| ≤ **0.1 %**;
   - Δδ_peak within one increment;
   - ℓ_cz within 2 elements;
   - E4–E6 pass for both solvers.

   **Target:** whole-curve ≤ 0.1 %, and toward the 0.04 % floor if reachable. If the gates
   pass but the target is missed, report where along δ the difference sits before trying to
   close it.

5. **Two toggles on this case only.** One run each, compared with the default:
   - **Kinematics: SVK vs Hencky** (D21 update). If the whole-curve |ΔP| between them is
     ≤ 0.01 % of P_peak, SVK stays the default. Otherwise report it and ask the user.
   - **Interface frame: global axes vs the deformed mid-surface frame** (what COH2D4 does
     under NLGEOM).
     - Implement the rotating frame first: separations rotated into the frame of the deformed
       mid-surface, built from the average of the two faces' current positions.
     - Report its effect on P–Δ. Adopt it as the default if it moves P toward ABAQUS. Report
       either way.

6. **Timing,** on the same Negishi node type as the ABAQUS reference:
   - JAX on 1 core and on 4 cores, against ABAQUS's ~20–25 min;
   - total Newton iterations, cutbacks, and the first-call (JIT) cost reported separately.

   Measured numbers only.

### S3e — The four extreme cases

Run `sw_s58`, `sw_s21`, `as_s21_am3` and `as_s21_a7` with the S3d defaults. Apply the same
gates and comparison table.

These test the short-zone limit (fewest elements per ℓ_cz) and the long-tail limit (zone
longer than H, end effects). Report per case; do not average.

Reproducing the full σ_c and α sweeps belongs to S4.

## S3 exit criteria

- S3a elastic gate passes.
- S3b single-element and two-block gates pass with one Newton solve per increment.
- S3d and S3e gates pass for all five cases, or the failures are explained and the user has
  agreed how to proceed.
- SOLVER_LOG has entries S3a–S3e and an updated summary row for S3.
- Timing is recorded against ABAQUS on the same hardware.
- `docs/STATUS.md` and a session log are written per the context protocol.

## Out of scope for S3

- Gradients through the load history (S5). Keep the code differentiable: no Python-side
  branching on traced values in the energy, and history passed as arrays. Do not build the
  adjoint yet.
- Batching across laws and GPU runs (S4).
- Fewer increments or a graded mesh (S4, each with an accuracy check against this S3 baseline).
- Any change to the ABAQUS-side code or to production settings D16.
