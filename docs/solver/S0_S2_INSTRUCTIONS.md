# Differentiable solver — Claude Code instructions for S0–S2

Written by Claude Web on 2026-11-14 (session log `docs/sessions/2026-11-14_web_solver-plan.md`,
decisions D19–D20). Read `CLAUDE.md`, `docs/STATUS.md`, and this file before starting.

## Purpose and scope

We are building a differentiable forward solver for the T-peel model on top of **tatva**
(ETH SMEC, `github.com/smec-ethz/tatva`, JAX, energy-based FEM with global AD). It runs
alongside ABAQUS, not instead of it. ABAQUS remains the reference and the source of
synthetic "truth" data.

Our code is a separate layer in this repo; tatva is a pinned dependency.

This file covers only **S0 (environment), S1 (bulk element), S2 (cohesive law and
interface)**. S3 (T-peel forward solve vs ABAQUS), S4 (throughput) and S5 (adjoint
gradients) come later and get their own instructions.

**Stop after each step.** Write its `docs/SOLVER_LOG.md` entry, commit, push, and report to
the user. Do not start the next step until the user says so.

## What tatva gives us, and what it does not

These facts were checked against v0.11.6 on 2026-11-14.

**Tatva provides:**
- `Mesh`, `Operator` (`eval`, `grad`, `integrate`, `map`)
- elements Line2, Line3, Tri3, Tri6, Quad4, Tet4, Hex8, plus custom elements by subclassing `element.Element`
- sparse Jacobians by graph colouring (`tatva.sparse.jacfwd`, `ColoredMatrix`)
- `compound` multi-field unknowns
- `lifter` constraints
- MPI

**Tatva does not provide:**
- a nonlinear solver, load stepping or cutbacks
- history-variable management
- a cohesive element class (the cohesive models are notebooks in the separate `tatva-docs` repo)
- any adjoint or inverse machinery

**Two traps to avoid:**
- A Newton loop written with `lax.while_loop` cannot be reverse-differentiated. Keep the solver differentiable by implicit differentiation later (S5), not by unrolling.
- The API is young: there were module renames in 0.11.5. **Pin `tatva==0.11.6`.**

**Licence:** tatva is LGPL-3.0. Use it as a dependency and **do not modify or vendor it**. If a
tatva bug blocks you, work around it in our code and record it in
`docs/solver/TATVA_NOTES.md` (bug, minimal reproducer, workaround). Do not open upstream
issues or contact the authors without the user's go-ahead. The user has decided not to
contact the tatva group for now (D19).

A starting prototype is in `scripts/fe/prototype_dcb_tatva.py`. It is a 2D DCB with Quad4
bulk, Line2 cohesive and a bilinear law with history frozen per step, plus an implicit
single-step gradient dP/dσ_c that matched central finite differences to 0.08 %. It is an
**unverified prototype**: dense tangent, plain Quad4, not validated against anything. Use it
for API patterns only.

## Project-wide rules for this work

- **Units:** mm, N, MPa, N/mm. Same as the ABAQUS decks; state this in every module docstring.
- **Precision:** `jax.config.update("jax_enable_x64", True)` in `src/tslid/fe/__init__.py`. All verification is in float64.
- **No ABAQUS-side changes.** `src/tslid/laws.py`, `build_tpeel_inp.py` and the extractors must remain importable by the existing Python. `src/tslid/__init__.py` must **not** import `tslid.fe`, so that `import tslid.laws` never pulls in JAX.
- **Gradient safety.** Every function in the law and energy path must give finite, correct `jax.grad` everywhere, including δ = 0, δ = δ0, δ = δ_f, α = 0 and compression.
  - Use the "double-`where`" pattern for branches.
  - Use a series expansion of g(α) and s(x; α) near α = 0 (|α| < 1e-3).
  - Add tests for all of these points.
- **Labels.** Mark results [D] demonstrated, [H] hypothesis, [I] interpretation (CLAUDE.md rule 2). Report failed runs.
- **Compute on Negishi.** Heavy jobs go through `sbatch` (CLAUDE.md cluster rules). Short unit tests may run on the laptop or in CI. The development loop is **local** (user's laptop with Claude Code), and production runs go to the cluster (D19).
- **Outputs.** Small CSVs and figures go to `results/solver/<step>/` with a README (same style as `results/phase0/`). Raw data goes to scratch.

## S0 — Environment and skeleton

**Goal:** an environment that is identical on the laptop and on Negishi, and a package
skeleton with CI.

1. **JAX-side environment with `uv`.**
   - Add a repo-root `pyproject.toml` that declares the `tslid` package (src layout) with an optional dependency group `fe`: `jax`, `tatva==0.11.6`, `jax-autovmap`, `numpy`, `scipy`, `matplotlib`, `pytest`.
   - Require Python ≥ 3.11 for this group only. Commit `uv.lock`.
   - The ABAQUS-side scripts keep working with the existing interpreters. Verify by running `tests/test_laws.py` with both environments.
2. **Package layout.** Docstring stubs only, no logic beyond S0:
   ```
   src/tslid/fe/
     __init__.py        # x64 on; version info; no heavy imports
     elements.py        # S1: custom bulk elements
     kinematics.py      # S1: small-strain and finite-rotation (SVK) energies
     laws_jax.py        # S2: JAX version of the alpha family (mirrors tslid.laws)
     interface.py       # S2: cohesive interface energy, separations, history update
     meshing.py         # S1/S3: structured meshes; later the T-peel mesh
     solve.py           # S1+: Newton with sparse tangent; load stepping (S3)
   tests/fe/
   results/solver/
   docs/solver/
   ```
3. **CI.**
   - Add `.github/workflows/tests.yml`: Ubuntu, `uv sync --extra fe`, then run `pytest tests/test_laws.py tests/fe -m "not slow"` on CPU.
   - Mark anything over ~60 s as `slow`.
   - Add the test badge to `README.md`.
4. **Smoke tests in `tests/fe/test_s0_smoke.py`:**
   - tatva imports, and its version equals 0.11.6;
   - x64 is active;
   - a Quad4 `Operator` integrates 1 over a unit square to 1 within 1e-14.
5. **Cluster check.**
   - Run the same tests in a short `sbatch` job on Negishi (CPU partition), using the uv environment.
   - Record the exact commands in `docs/solver/ENVIRONMENT.md`: install, sync, run tests, and how to get a GPU build of JAX later. GPU is not needed yet; record which Purdue cluster and partition offers GPUs, or write TBD.

**S0 exit criteria:** CI is green; the same tests pass on the laptop and on Negishi;
`tests/test_laws.py` still passes in the old environment; the SOLVER_LOG entry for S0 is written.

## S1 — Bulk element and kinematics

**Goal:** know the bulk discretization error before any cohesive comparison.

ABAQUS uses **CPE4I** (incompatible modes) with **NLGEOM = YES**. The adherends are elastic
Al: E = 70 000 MPa, ν = 0.33, plane strain, arm thickness H = 6.324 mm, out-of-plane width
W = 15 mm. The production mesh (D16) uses ny = 20 elements through each arm, h_y0 = 0.1 mm at
the bond face, and h_coh = 0.1 mm along the bond.

Plain bilinear Quad4 shear-locks in bending, so an unexplained stiffness mismatch with
ABAQUS would most likely come from here.

1. **Kinematics** (`kinematics.py`):
   - small-strain linear elasticity;
   - finite-rotation St. Venant–Kirchhoff, total Lagrangian, in plane strain.

   ABAQUS NLGEOM uses a hypoelastic, corotational-type update for elastic solids. At the
   small strains of elastic Al (≲ 0.5 %) this difference should be second order in strain
   [H]. Quantify it in test 4 below; do not assume it.

2. **Elements** (`elements.py`). Implement and compare:
   - **(a) Q4:** tatva's built-in Quad4, full 2 × 2 integration. Baseline.
   - **(b) Q4-IM:** an incompatible-modes / enhanced-strain quad matching CPE4I as closely as possible.
     - Recommended implementation in the global-energy setting: carry the 4 incompatible-mode amplitudes per element as **extra unknowns** in a `tatva.compound` field. Each amplitude couples only to its own element, so sparsity is preserved and the total energy is minimized jointly. No static condensation code is needed, and AD gives the tangent.
     - For finite rotation, use an enhanced deformation gradient (Simo–Armero type, enhancement mapped with the centroid Jacobian).
     - Before coding, document in `docs/solver/ELEMENTS.md` exactly which formulation you implement and how it relates to ABAQUS's CPE4I description in the ABAQUS manual. Cite the manual section; do not guess.
   - **(c, optional) Q8 or Tri6:** quadratic elements, only if (b) fails the thresholds.

3. **Tests** (`tests/fe/test_s1_*.py`; figures and CSV in `results/solver/S1_bulk/`):
   1. **Patch test.** Linear displacement field on a distorted 5-element patch, for (a) and (b). Constant-strain energy is exact to 1e-12 relative.
   2. **Linear cantilever.**
      - Geometry: length 60 mm, thickness H = 6.324 mm, plane strain, clamped at one end, tip shear load. Use the end-load magnitude that gives a tip deflection of the order seen in the T-peel arm; take it from an existing T-peel run at δ ≈ 1 mm, or state your choice.
      - Meshes: ny ∈ {2, 4, 8, 20}, element aspect ratios as in the production mesh.
      - Report tip deflection error versus Timoshenko beam theory with E′ = E/(1−ν²). Report it, but don't use it as a pass/fail, because of end effects.
   3. **Same cantilever in ABAQUS.** CPE4I, NLGEOM = NO, built by a small new script in `scripts/fe/` (do not modify the T-peel builder). Run with sbatch.
      - **Threshold:** element (b) at ny = 20 matches ABAQUS tip deflection and reaction within **0.05 %**.
      - Also report (a) at every ny, so the locking error is documented.
   4. **Large-rotation cantilever.** Tip rotation of about 0.1–0.3 rad: estimate the arm rotation in the T-peel from an existing run and use at least that. Compare SVK (b) with ABAQUS CPE4I at NLGEOM = YES.
      - **Threshold:** **0.1 %** on tip displacement components and reaction.
      - If this fails, report the size of the difference before changing anything. It may be the hypoelastic-vs-SVK effect.
   5. **Timing.** Wall time per Newton solve for (a) and (b) at the cantilever size, on the laptop CPU. Recorded for information only.

**S1 exit criteria:** tests 1, 3 and 4 pass with element (b), or the user agrees on an
alternative element; `ELEMENTS.md` is written; the S1 SOLVER_LOG entry is written.

## S2 — Cohesive law and interface

**Goal:** a JAX cohesive interface that reproduces the ABAQUS cohesive behaviour exactly at
the material point and in the 14 single-element runs.

What the ABAQUS deck specifies, from `build_tpeel_inp.py` and `laws.py`; re-read the code and
confirm each item:
- `*Elastic, type=TRACTION` with K_nn = K_ss = K_tt = K0 = 8500 MPa/mm, constitutive thickness 1.0 (so separation = nominal strain);
- `*Damage Initiation, criterion=QUADS` with equal normal and shear strengths σ_c;
- `*Damage Evolution, type=DISPLACEMENT` with LINEAR (α = 0), EXPONENTIAL (α ≠ 0) or TABULAR;
- no mixed-mode behaviour keyword, so evolution is mode-independent in the effective separation;
- damage does not act on compressive normal traction;
- `*Section Controls, viscosity=μ` (μ = 1e-4 s in production, D16; the single-element runs used none).

Confirm the exact definitions in the ABAQUS manual (effective separation
δ_m = √(⟨δ_n⟩² + δ_s²), the QUADS form, the EXPONENTIAL formula, and the viscous
regularization ODE for D_v) and cite the sections in `docs/solver/INTERFACE.md`.

1. **`laws_jax.py`.** A JAX version of the α family: g(α), s(x; α), δ0, δ_f, traction(δ), damage D(δ_max).
   - It must be NaN-free and correctly differentiable at α = 0 (series branch) and at the kinks.
   - Parameters are (K0, σ_c, G_c, α), all differentiable inputs.
2. **`interface.py`.**
   - Separations on the bond line are computed from matched node pairs on the upper (y = +t/2) and lower (y = −t/2) bond faces. Geometric thickness is 0.4 mm and constitutive thickness 1.0.
   - The interface energy density, with damage frozen at the history value, is
     ψ = (1 − D) · ½ K0 (⟨δ_n⟩₊² + δ_s²) + ½ K0 ⟨−δ_n⟩₊²,
     with D = D(δ_m,max).
   - The traction comes from `jax.grad` of ψ. Verify that it equals the ABAQUS secant law.
   - A separate `update_history(δ, history)` updates δ_m,max (and the viscous D_v when μ > 0).
   - Normal and shear are measured in the **global frame** for S2. Add a switch for the deformed mid-surface frame (what COH2D4 does under NLGEOM) to be tested in S3. Record this as an open item.
   - Quadrature: find out from the manual which integration COH2D4 uses (Gauss or nodal/Newton–Cotes) and match it. Write the answer and its source in `INTERFACE.md`.
3. **Tests** (`tests/fe/test_s2_*.py`; outputs in `results/solver/S2_interface/`):
   1. **Material point vs `tslid.laws`.** Traction and damage for all parameter sets of the 14 single-element runs (`results/phase0/single_element/single_element_summary.csv`) plus α ∈ {−3, −1.5, 0, 1.5, 3, 5, 7} at σ_c = 21 and 40, G_c = 0.4. **Threshold: ≤ 1e-12 σ_c.**
   2. **Single cohesive element vs ABAQUS.**
      - Reproduce the 14 single-element runs: one element, 1 mm², opened to 1.2 δ_f in 2000 increments, and the two cyclic runs that unload at x = 0.4 to zero and reload.
      - Compare against the ABAQUS histories on Negishi scratch (`/scratch/negishi/basu56/tslid_runs/phase0_single_element/`; extract with the existing extractor if needed).
      - **Thresholds:** traction ≤ 1e-6 σ_c versus ABAQUS for EXPONENTIAL and LINEAR, and ≤ 2e-5 σ_c for TABULAR (ABAQUS's own TABULAR error was 1.6e-5); dissipated energy vs G_c ≤ 1e-5; secant unloading error ≤ 1e-6 σ_c.
   3. **Mixed mode at a material point.** Pure shear, and proportional mixed mode at δ_s = δ_n, compared against a short ABAQUS single-element run of each (new deck from the existing single-element script, not a modified one). Same thresholds. This checks QUADS initiation and the effective-separation evolution, which the Mode I runs do not exercise.
   4. **Viscous regularization.** Implement the D_v ODE with backward Euler as ABAQUS does. On the single element, check that μ → 0 recovers test 2 and that μ = 1e-4 s gives the documented behaviour. Timing follows the T-peel step definition (rate 1 mm/min).
   5. **Gradients.** AD versus central finite differences for d(traction)/d(σ_c, G_c, α, K0) and d(dissipated energy)/dθ, at the points listed under gradient safety.
      - **Threshold:** relative error ≤ 1e-6 (choose the FD step and report it).
      - Check that d(dissipated energy)/dG_c = 1 within 1e-8.
   6. **Assembly sanity.** A tiny two-block, bulk-plus-interface problem (opening of a 2-element bar) in which force equals traction × area exactly. This is not the T-peel; that is S3.

**S2 exit criteria:** tests 1–6 pass; `INTERFACE.md` is written; the S2 SOLVER_LOG entry is
written.

## After S2

Stop, report, and wait. S3 will build the T-peel mesh and quasi-static loading. It will
choose between staggered and monolithic history handling, and it will compare P–Δ with the
ABAQUS σ_c = 40, α = 0 run, then both sweeps. The target is 0.5 % of P_peak, to be
tightened toward the 0.04 % floor where possible.
