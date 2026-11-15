# Project status (handoff page)

_Last updated: 2026-11-15 (S4a) by Claude Code. Overwrite this page each session._

## Where we are

**Phase 0 — infrastructure and verification.**

Agreed: D1–D7, D9 (G_c 0.1–0.4 N/mm, K = 8500), D10 (energy constraints + tolerances),
D11 (elastic adherends, H 6.324, a0 18, t 0.4, clamp/load), D12 (σ_c lower bound from
ℓ_pz ≤ c·H, c = 1 proposed), D13 (Negishi only).

Done:
- `src/tslid/laws.py` (α-family, E1–E3), 11 unit tests pass.
- Generator `--law alpha`; rejects laws violating E1–E3.
- **Single-element verification passed (14/14)** — `results/phase0/single_element/`.
  ABAQUS EXPONENTIAL == s(x; α), α < 0 accepted, TABULAR within 1.6e-5 σ_c, dissipation = G_c.

Done (pilot): **ℓ_cz map, α = 0, 21/21 runs** — `results/phase0/pz_map/`. ℓ_cz/H is a
single function of ℓ_ch/H; ℓ_cz ≈ 0.02–0.14 ℓ_ch. c = 1 → σ_c,min 10.5 / 16.6 / 21.0 MPa
at G_c 0.1 / 0.25 / 0.4. Energy checks pass, except E4 at σ_c = 5 (ℓ_cz > 2.7H, out of bounds).

## Warning — zone lengths

SDEG-based lengths (pilot preview, campaign03 Stage F) **underestimate** the zone by up to
~25 %. Use ℓ_cz from the opening profile (D14, agreed). ℓ_pz is undefined until the user
settles it.

Pilot preview [D, α = 0, 17 runs]: ℓ_cz/H collapses onto one curve in ℓ_ch/H:
ℓ_ch/H 1.5 / 3.1 / 5.5 / 12.4 / 50 → ℓ_cz/H 0.18 / 0.42 / 0.66 / 1.05 / 1.94.

## Done (evening): convergence study — `results/phase0/convergence/`, D16
Error floor [D]: mesh and increment change P by ≤ 0.04 % of P_peak; viscosity 1e-3 biases P by
≈ 0.1 % just after the peak; 1e-4 ≡ 1e-5 (< 0.01 %). All energy checks pass (E4 ≤ 5.4e-5).
Production: h_coh 0.1, viscosity 1e-4, n_inc 800, Δ 1.6 (D16).

First result (D15, pilot runs, viscosity 1e-3): `results/phase1a_sigc/` — at G_c 0.4, σ_c 25–57
changes P_peak by −5.8 … +3.5 % (≫ floor), post-peak P–Δ differs by ≤ 0.12 % (≈ floor).

## Done: dense σ_c sweep (D15) — `results/phase1a_sigc/` (second section of README)
G_c 0.4, α 0, σ_c 21–58, production settings, 9/9 runs, all energy checks pass. P_peak −8.4 … +3.8 %
vs σ_c 40; post-peak P–Δ identical to ≤ 0.21 % (σ_c ≥ 30: ≤ 0.08 %, ≈ floor); the residual decays
away from the peak. σ_c information is confined to δ ≲ 0.4 mm.

## Done: α sweep (D17) — `results/phase1b_alpha/README.md`
SLURM 45840508, 14/14, all energy checks pass. [D] α changes only the peak region, like σ_c. The P_peak
shift is as large as for σ_c: −6.2 … +2.6 % (σ_c 40) and −8.9 … +4.7 % (σ_c 21).
[D] σ_c–α trade-off: same-G_c laws with matched P_peak (e.g. σ_c 25/α 0 vs σ_c 21/α −1.5,
ΔP_peak 0.1 %) differ only on the rising branch before the peak (max 1.7–5 %). After the peak, ≤ 0.11 %.
Out of the c = 1 bound: σ_c 21 with α ≥ 1.5, and σ_c 40 with α 7 (flagged).
compare_ld.py window now ends at the σ_c front (fixes an end-effect artefact for long zones).

## Running

Nothing.

## Runtime note
A base run takes ~20–25 min on 1 CPU. The cost is 26k CPE4I elements (adherend columns follow
h_coh = 0.1 through the full arm thickness) and full-model U/RF field output every increment
(~390 MB odb). ℓ_cz is NOT the cause: ≥ 11 elements per ℓ_cz at h = 0.1, and h = 0.05 changes
nothing. Speed-ups before large data sets: thin the output (no physics change); a fine bond-line
strip with coarser tied arms (needs a convergence check).

## Next actions

0. Repo is public; Claude Web can read it.
1. Decide with the user: the c = 1 bound and α (it was set at α = 0), and how to treat the σ_c–α trade-off
   (a 2D (σ_c, α) map of the peak-region misfit at fixed G_c would quantify the non-uniqueness).
1b. Cheap (Web suggestion): R-curve G(ℓ) from existing σ_c-sweep runs; plateau should return 0.40, transient ∝ ℓ_cz.
2. Possibly repeat the σ_c sweep at G_c 0.1 and 0.25 (same design) to show the result holds across G_c.
3. Speed up the model (output thinning, then mesh grading + check) before generating data sets.
4. User: ℓ_pz definition.

Git: the Negishi clone now tracks `main` (its old `master` branch is stale; leave it).

## Open questions for the user

- α range for Phase 1 (now that α < 0 works): e.g. −3 … 7?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7).
- Figures follow `FIGURE_DESIGN.md` (IJF guide) conventions unless told otherwise.

## Differentiable solver (D19, D20) — `docs/SOLVER_LOG.md`
- Claude Web plan applied (commit ce37d58): S0–S2 instructions `docs/solver/S0_S2_INSTRUCTIONS.md`,
  D18 (proposed: P_peak tracks ℓ_cz/H), D19–D20 agreed, prototype `scripts/fe/prototype_dcb_tatva.py`.
- **S0 done** [D]: uv env (Python 3.12.12, jax 0.12.0.1, tatva 0.11.6) on Bell and Negishi, CI green,
  16/16 tests everywhere; `test_laws` still OK on Python 3.6. Setup: `docs/solver/ENVIRONMENT.md`.
- **S1 done** [D]: Q4IM ≡ CPE4I (1e-8 linear; NLGEOM v, M ≤ 4e-5 to 0.1 rad). Flag: u_tip 1.25 % (SVK) /
  0.30 % (Hencky) of its own value, constitutive and 2nd order; T-peel max rotation 0.0325 rad. D21 proposed
  (Q4IM + Hencky). `results/solver/S1_bulk/`.
- **S2 done** [D]: JAX cohesive interface = ABAQUS COH2D4 to ≤ 6.8e-8 σ_c (20 inviscid runs: mode I/shear/mixed,
  LIN/EXP/TAB, cyclic); viscous BE exact until D = 1 (post-failure tail ≤ 3.4e-5 σ_c, unexplained); COH2D4 uses
  **nodal** quadrature; exact gradients in (K0, σ_c, G_c, α). `results/solver/S2_interface/`.
- D21 updated (Claude Web 2026-11-15, agreed): Q4IM + **SVK** default, Hencky cross-check in S3d. D22 agreed
  (S3 design). S3 instructions: `docs/solver/S3_INSTRUCTIONS.md`.
- **S3a done, gate missed (explained)** [D], `results/solver/S3_tpeel/`:
  - The T-peel is imported from the deck node for node (`inp_import.py`). Couplings are eliminated as finite
    rigid motions (`tpeel.py`).
  - Linear elastic T-peel = ABAQUS to 3e-8 (RF2, UR3, mouth, whole field, cohesive tractions).
  - NLGEOM at T-peel loads: RF2 2–3e-5, UR3 5e-5, mouth 1–1.5e-5 (gate 1e-5). SVK and Hencky agree, so the
    bulk energy is not the cause [H: the S1 Q4IM-vs-CPE4I NLGEOM floor].
  - **Finding:** COH2D4 with a geometric gap is corotational. It measures the current gap vector in the
    deformed mid-surface frame minus the reference gap (7 probes, `docs/solver/TPEEL.md` §3). The S2 jump
    form would add φ × 0.4 mm of spurious shear, comparable to δ₀. Implemented and now the default.
  - The power-series Hencky diverged on the T-peel; replaced by a closed form.
- **S3b done** [D]: incremental potential (Ortiz–Pandolfi type incl. the viscous BE update), one Newton solve
  per increment. It equals the S2 scheme to 1e-12 σ_c on all 20 ABAQUS single-element runs; two-block test
  43 vs 183 iterations.
  - Driver with line search and cutback (`incremental.py`).
  - Coarse damaged T-peel completes: P_peak −2.2e-5 vs ABAQUS on the same deck; 670 s vs ABAQUS 256 s,
    unoptimised.
  - Staggered coarse comparison (job 45868159) timed out at 4 h (≥ 28× slower than monolithic).
- ABAQUS reference wall times are 36–56 min (not 20–25), from the .msg files.
- User review of S3a/S3b (2026-11-15): NLGEOM floor accepted, gate revised to ≤ 5e-5 RF2 (**D23**). S3c order
  set by the user (`S3_INSTRUCTIONS.md` update).
- **S3c done** [D] (SOLVER_LOG S3c, `results/solver/S3_tpeel/` S3c section, `docs/solver/TPEEL.md` §6):
  - Profile: 61 % of the S3b coarse run was abandoned attempts before cutbacks, not line search (S3b note
    corrected).
  - Gain ladder, coarse, 1 thread: 926 → 657 (energy Newton) → 572 (divergence check) → 179 (element-local
    assembly) → 102 (IM condensation) → **91 s** (PARDISO-sym with reuse). ABAQUS: 256 s.
  - Element tangent = jacfwd to 6e-14; condensed step backward error 5e-16.
  - rtol 1e-6 = loosest tolerance with |ΔP| ≤ 1e-5 P_peak (2.3e-6).
  - Coarse E4–E6 pass (4.3e-5 / 3.0e-4 / −1.1e-3).
  - **Production `sw_s40` measured: 490 s on 1 thread (335 s on 4) vs ABAQUS 2321 s.** P_peak +5.7e-6, window
    max |ΔP| 8.4e-5 P_peak, E4–E6 equal to ABAQUS's (preliminary for S3d).
  - **D24 proposed**: this configuration + rtol 1e-6; `pypardiso` added to the `fe` extra.
  - UMFPACK / CHOLMOD not benchmarked (no SuiteSparse on Negishi; a micromamba install was blocked by
    session permissions).
- User review of S3c: **D24 agreed** (+ SuperLU fallback, done), **D25** (residual check for the
  condensed step; D25 timing protocol; after S3e: S4a = the 21 sweep runs vs ABAQUS, then S5; S4b
  batching/GPU/increments after S5).
- **S3d done** [D] (`sw_s40`; SOLVER_LOG S3d):
  - All gates pass and the 0.1 % target is met: max |ΔP| 8.6e-5 P_peak, ΔP_peak +5.7e-6, ℓ_cz equal,
    E4–E6 equal to ABAQUS's.
  - Toggles: Hencky vs SVK 2.7e-5, so SVK stays. Frames `global` = `midsurface` (2e-9 N); both are
    farther from ABAQUS than the corotational default, which stays.
  - D23 item: the S3a traction difference sits at the crack tip.
  - **Timing (D25, one job, one node): tslid.fe 532 s (1 thread) and 383 s (4) vs ABAQUS 2292 s
    (default deck) and 2183 s (thinned output): 0.23× / 0.17×.**
- **S3e done** [D] (`sw_s58`, `sw_s21`, `as_s21_am3`, `as_s21_a7`):
  - All P gates and the 0.1 % target are met (worst `sw_s58`, max |ΔP| 3.6e-4).
  - **One explained failure: `as_s21_a7` E5 = 3.4e-2 in the `energy_checks.py` window.** Its window
    ends mid end-of-ligament collapse for the α = 7 long tail; ABAQUS skips the collapse in one
    frame.
  - Over the compare_ld window, E4/E5 pass in all five cases and E5 equals ABAQUS's within 3 %.
- Energy split in `tpeel_output.py` corrected (viscous-overstress split; the S3c split
  double-counted a lagging D_v). All S3d/S3e runs repeated with it.
- User review of S3d/S3e: **D26**:
  - E4–E6 are computed in the load-comparison window (`energy_checks.py --window front`; the
    default is unchanged, verified bit for bit).
  - S3 accepted.
  - S4a defined.
  - S5 waits for the Claude Web instructions.
- **S4a done** [D] (`results/solver/S4a_sweeps/`, SOLVER_LOG S4a):
  - **23/23 sweep runs (21 laws, G_c 0.4) pass all load gates and the 0.1 % target** (max |ΔP|
    ≤ 5.5e-4 P_peak, ΔP_peak ≤ 8.3e-5, ℓ_cz ≤ 0.04 el.).
  - E4/E5 pass in all; E6 agrees with ABAQUS to ≤ 1e-4.
  - **E6 fails for `as_s21_a7` identically in both solvers (−0.0212)** in the D26 window.
  - D18 matched-peak pairs: the pair differences (1.7–5.1 % of P_peak) agree between solvers to
    ≤ 1.05e-4 P_peak (≤ 0.4 % of the signal).
  - Duplicate laws: ABAQUS bit-identical, tslid.fe 1e-10.
- **Stopped for user review after S4a** (D26).
- **Next:**
  - User: decide on E6 for long-tail laws in the D26 window (accept as a case property, use the
    default window for E6, or define the E6 steady range by the tip).
  - S5: `apply_solver_S5.sh` / `S5_INSTRUCTIONS.md` arrived (repo root); not applied, waiting for
    the go-ahead.
  - S4b (batching, GPU, increments) after S5.
- Open: GPU model / access on Negishi `gpu` partition (S4); laptop env untested; ABAQUS manual citations
  unverified (user checking 3DS access).

## Update — Claude Web, 2026-11-15 (S5 plan)
- S5 instructions are in `docs/solver/S5_INSTRUCTIONS.md` (decision "S5 design").
- **Start S5 only after the S4a review.** Order: S5.0 → S5a → stop → S5b → S5c → stop.

## Update — Claude Code, 2026-11-15 (S4a review, S5.0 + S5a)
- **D27** (S4a review): S4a accepted; E6 on long-tail laws is a property of the check (open item:
  E6 steady range by the broken tip, for the R-curve analysis); deterministic FD required. S5
  instructions applied (**D28**, S5 design); `apply_solver_S5.sh` removed.
- **S5.0 + S5a done** [D] (SOLVER_LOG S5.0/S5a; `results/solver/S5_sensitivity/`; derivation
  `docs/solver/ADJOINT.md`):
  - Law parameters are jit arguments of the production kernels.
  - Direct sensitivities (`src/tslid/fe/sensitivity.py`) for θ = (ln K0, ln σ_c, ln G_c, α).
  - Verified: partials ≤ 1e-7; single element vs AD ≤ 1e-10; two-block vs FD ≤ 1e-9; coarse T-peel
    vs FD with replay, best steps 1.3e-6 / **1.02e-5** / 6.1e-7 / 4.0e-6 (ln σ_c marginally above
    the 1e-5 gate; FD-limited: event increments vs the 1/h noise floor).
  - Deterministic runs are bit-identical.
  - **S5.0 forward gate not met at ≤ 1e-14:** old vs new code differ by 1 ulp from increment 2, up
    to 3.8e-12 – 6.5e-12 P_peak in the window; two schedules depart after the window.
  - Cost: production sw_s40 forward 541 s → with 4 sensitivities 1051 s (1 thread).
  - Post-peak sensitivities are spiky (element failures) and schedule-dependent at the spikes;
    pre-peak smooth.
- **Stopped for user review after S5a.**
- **Next (user):** accept or not the roundoff-level S5.0 change; accept the ln σ_c FD result or
  rerun FD with tighter Newton; S5c treatment of the spikes; then S5b → S5c.
