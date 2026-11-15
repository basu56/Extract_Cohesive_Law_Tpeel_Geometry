# S3 — T-peel forward solve vs ABAQUS (S3a, S3b)

Units: mm, N, MPa, s. Formulation: `docs/solver/TPEEL.md`. Instructions: `docs/solver/S3_INSTRUCTIONS.md`.
Negishi rundirs:
- S3a: `/scratch/negishi/basu56/tslid_runs/solver_S3a_elastic/`, `solver_S3a_cohprobe/`;
- S3b: `solver_S3b_coarse/`.

All runs are on AMD EPYC 7763 nodes.

| file | content |
|---|---|
| `s3a_elastic.csv` | S3a gate: every (ABAQUS run, tslid.fe run) pair per frame. Relative errors of RF2, UR3 (RP1), mouth opening, and lower mouth U2 relative to the opening; absolute RP1 U1 error (mm); full-field max\|ΔU\|/max\|U\|; cohesive tractions (max\|Δt\| / max\|S22\|); Newton iterations |
| `s3b_coarse.csv` | S3b coarse damaged T-peel, one row per solver run: increments, cutbacks, Newton iterations, line-search halvings, wall time, peak and whole-curve P differences vs ABAQUS on the same deck |
| `s3b_coarse_pd.npz` | P–Δ curves of those runs (float32) |
| `s3ab.png/pdf` | (a) S3a RF2 error vs load; (b) coarse P–Δ; (c) coarse ΔP vs Δ |

Scripts:
- `scripts/fe/build_s3a_elastic.py`, `run_s3a.sh`, `run_s3a_diag.sh`, `run_s3a_scale.sh`,
  `run_s3a_kin.sh`, `s3a_elastic.py`, `s3a_compare.py`;
- `build_coh_probe.py`, `run_coh_probe.sh`, `extract_tpeel_coh.py`;
- `run_s3b_coarse.sh`, `s3_run_tpeel.py`, `s3ab_plot.py`.

## S3a — mesh import, couplings, elastic gate

**Decks.** The elastic decks call the unchanged `build_tpeel_inp.py` with every parameter of
`sw_s40` except `--law elastic`, `--n-inc` and `--job`. [D] Their nodes, elements and sets are
byte-identical to `sw_s40.inp` (md5).

The `_tight` copies add one card after `*Static`, `*Controls, parameters=field, field=global` /
`1e-9, 1e-9`, so that ABAQUS's own Newton tolerance does not limit the comparison. [D] With
ABAQUS's default controls, RF2 at δ = 1.6 mm changes by 1.05e-4.

**Elastic loading.** The bond is linear elastic with K0 everywhere. δ = 1.6 mm gives RF2 = 7026 N,
12× the real T-peel peak (562 N in `sw_s40`). The 16-increment deck therefore covers the real load
range in its first two frames.

**Results [D]** (relative to ABAQUS with tight controls):

| case | RF2 | UR3 | mouth opening | U field | cohesive tractions |
|---|---|---|---|---|---|
| NLGEOM = NO, linear kinematics, COH2D4 linearised kinematics (`co_lin`) | ≤ 3.3e-8 | ≤ 3.0e-8 | ≤ 2.9e-8 | ≤ 4.2e-8 | ≤ 5.0e-8 |
| NLGEOM = NO, cohesive "global" frame (`lin`) | ≤ 3.3e-8 | ≤ 3.0e-8 | ≤ 2.9e-8 | **6.6e-3** (upper adherend offset 0.026 mm) | ≤ 5.0e-8 |
| NLGEOM, SVK, COH2D4 kinematics, P = 331 / 675 N | 2.2e-5 / 2.7e-5 | 5.2e-5 / 6.2e-5 | 1.3e-5 / 1.5e-5 | 3.3e-5 / 4.1e-5 | 2.6e-4 / 3.0e-4 |
| NLGEOM, Hencky, COH2D4 kinematics, P = 331 / 675 N | 2.6e-5 / 3.6e-5 | 4.6e-5 / 4.4e-5 | 1.0e-5 / 9.2e-6 | 4.3e-5 / 4.6e-5 | 2.5e-4 / 2.6e-4 |
| NLGEOM, SVK, "global" frame, P = 331 / 675 N | 1.2e-4 / 2.2e-4 | 8.9e-5 / 1.4e-4 | 1.3e-5 / 1.7e-5 | 6.6e-3 | 2.0e-4 (shear 3.9e-3 / 7.8e-3) |
| NLGEOM, SVK, COH2D4 kinematics, P = 7026 N | 3.7e-4 | 8.0e-4 | 2.0e-4 | 1.5e-3 | 1.4e-3 |

**Gate (1e-5 relative on RF2, mouth openings and UR3): not met under NLGEOM.**

- *Linear.* The linear case is exact to the ODB's single precision. This verifies the mesh
  import, the couplings, the bulk element, the interface stiffness, the BCs and the reaction.
- *At T-peel loads.* With the correct cohesive kinematics, the misses are 2–3e-5 on RF2, about
  5e-5 on UR3 and 1–1.5e-5 on the mouth opening.
- *Not the bulk law.* SVK and Hencky give the same sign and nearly the same size at 331 N
  (RF2 −2.25e-5 vs −2.58e-5; UR3 +5.2e-5 vs +4.6e-5), so the bulk constitutive law is not the
  cause.
- *Same size as S1.* This is the size of the S1 NLGEOM bending floor: Q4IM vs CPE4I tip
  deflection 3.6e-5, accepted in D21. [H] The CPE4I large-deformation enhanced-strain formulation
  differs slightly from our additive (Simo–Armero) enhancement. This has not been isolated further.
- *SVK error curve.* The SVK curve in panel (a) dips because the error changes sign near 3000 N.

**Main finding: COH2D4 kinematics with a geometric gap.** See `docs/solver/TPEEL.md` §3 for the
probe table. COH2D4 measures the current gap vector in the deformed mid-surface frame, minus the
reference gap. With NLGEOM = NO it uses the linearisation of that. It does not measure the nodal
displacement jump.

The S2 interface (zero-gap single element) could not see this. In the T-peel, the bonded laminate
rotates rigidly (ω ≈ 0.066 rad in the linear run), so ω × 0.4 mm = 0.026 mm of tangential jump
carries no traction in ABAQUS.

`tslid.fe` now uses `frame="corotational"` (NLGEOM) or `"corotational_linear"` (NLGEOM = NO),
chosen automatically from the deck. All 7 probes are reproduced to roundoff.

**Hencky.** The original power-series Hencky (S1) diverged on the first T-peel iterate after the
RP jump, where |C − I| ~ 1 in the leg-top elements. It converged to a spurious state
(RF2 ~ 1e8 N). It was replaced by a closed-form 2×2 logarithm, which matches `scipy.linalg.logm`
to ≤ 5e-12 at all strain levels. The series form is kept as `"hencky_series"`.

**Cost of one elastic increment** (production, 159,014 unknowns, 34 colours): about 7 s with
3 Newton iterations, of which tangent 3.2 s and solve 4.0 s (SuperLU). Setup (import + colouring)
takes 2.5–6 s. ABAQUS took 123 s for the 16-increment tight deck.

## S3b — monolithic history (incremental potential), robustness

**Single element and two-block [D]** (`tests/fe/test_s3b_monolithic.py`):

- Monolithic tractions equal the S2 update scheme to ≤ 1e-12 σ_c on all 20 ABAQUS single-element
  runs (inviscid, viscous, TABULAR).
- They meet the S2 tolerances: inviscid ≤ 2e-7 σ_c; viscous ≤ 1e-6 until D = 1 and ≤ 5e-5 after;
  dissipation vs ABAQUS ≤ 1e-7 G_c.
- ∇Φ equals the ABAQUS update traction to ≤ 1.2e-14 σ_c on random states: α ∈ {−3, ±1e-3, 0, 0.3,
  3, 7}, analytic and tabular, inviscid and viscous, compression and unloading.
- The envelope integral gives G_c exactly for the α law.
- Two-block bar: R = tA, the law is reproduced to 1e-12 σ_c, and the bar fails completely (D = 1).

| two-block bar (23 steps) | Newton iterations per step | total |
|---|---|---|
| staggered α = 0 (S2) | 1–16 passes, 183 iterations in total | 183 |
| monolithic α = 0 | 2 (1 after failure) | 43 |
| staggered α = 3 | 1–17 passes | 160 |
| monolithic α = 3 | 3–4 | 64 |

**Coarse T-peel with damage** (ny = 4, h_coh = 0.5; σ_c = 40, α = 0, μ = 1e-4 s, 800-increment
target, δ = 1.6 mm; the same builder as production; 7,306 unknowns):

| run | increments | cutbacks | Newton total (max/incr) | line-search halvings | wall | P_peak vs ABAQUS | rms ΔP/P_peak |
|---|---|---|---|---|---|---|---|
| ABAQUS, same deck | 1479 | 232 | 2910 | – | 256 s | 562.397 N | – |
| tslid.fe monolithic, COH2D4 kinematics | 1778 | 321 | 2551 (8) | 602 | 670 s | −2.2e-5 | 2.9e-4 |
| tslid.fe monolithic, "global" frame | 1766 | 315 | 2550 (15) | 590 | 515 s | −1.2e-4 | 4.1e-4 |
| tslid.fe staggered, "global" frame | did not finish: 4 h SLURM limit (job 45868159), no records written | | | | > 14,400 s | – | – |

- The full step completes with ABAQUS-like incrementation. Both codes cut back only during crack
  propagation (t > 37 s), where the 0.5 mm cohesive elements fail one at a time. The peak and Δ at
  the peak agree to within one increment.
- Panel (c) spikes: the whole-curve maximum, 0.59 % at δ = 1.39, comes from single-element snap
  events landing at slightly different Δ in the two codes on this coarse bond mesh.
- The production-mesh gates are S3d's.
- Partial staggered run (Bell login, first 150 increments, "global" frame): the monolithic run
  reached t = 17.06 s with 241 Newton iterations and 0 cutbacks. The staggered run reached only
  t = 11.88 s in the same 150 increments.

**Cost split (coarse, monolithic, COH2D4 kinematics).**
- Of the 670 s wall time: tangent 190 s, linear solve 47 s, history update 22 s. The remaining
  about 410 s is residual evaluations in the line search, the lift and J products, and Python
  overhead per increment.
- S3c will profile and optimise this as instructed.
- *Correction 2026-11-15 (S3c profile, `docs/SOLVER_LOG.md`):* the ~410 s are mostly the 321
  abandoned attempts before cutbacks (~456 s of 753 s in the profiled rerun), not line-search
  residuals or Python overhead. The timings above counted converged increments only.
- First call (JIT) 2.0 s.

## S3c — performance, tolerance, energy gate (coarse) and production timing

Details and all numbers: `docs/SOLVER_LOG.md` (S3c). Formulation: `docs/solver/TPEEL.md` §6.
Negishi rundir: `/scratch/negishi/basu56/tslid_runs/solver_S3c/` (subdirs `ladder/`, `tol/`, `gate/`,
`prod/`, `bench2/`). Timed runs are pinned to 1 thread unless stated (`scripts/fe/pin_threads.sh`).

| file | content |
|---|---|
| `s3c_ladder_runs.csv` | gain ladder, coarse: one row per configuration and copy (A, B), plus ABAQUS. Options, increments, cutbacks, iterations incl. abandoned attempts, wall split, phase timers, P vs ABAQUS (full record and compare_ld window `_win`) |
| `s3c_tol_runs.csv` | tolerance study: reference `tol_ref` (rtol 1e-10), replays `tol_<rtol>_rep` on its increments (`max_dP_ref_same_t_win`), and free runs `tol_<rtol>_free` |
| `s3c_bench.csv` | micro-benchmarks, coarse and production, 1 and 4 threads: tangent (jacfwd / element / condensed), residual, linear solvers (first call incl. analysis, mean of reuse calls, backward error) |
| `s3c_explore_runs.csv`, `s3c_explore_bench.csv` | exploratory runs: unpinned, or first condensation version. Kept for the record; superseded by the files above |
| `s3c_energy_gate_coarse.csv`, `s3c_energy_abaqus_coarse.csv` | `energy_checks.py` output, final configuration vs ABAQUS, coarse deck |
| `s3c_energy_prod.csv`, `s3c_energy_abaqus_prod.csv` | same for production `sw_s40` (rtol 1e-6) |
| `s3c_prod.csv`, `s3c_prod_pd.npz` | production `sw_s40`: ABAQUS and three tslid.fe runs (rtol 1e-10 / 1e-6, 1 / 4 threads), timings, accuracy; P–Δ curves (float32) |
| `s3c.png/pdf` | (a) gain ladder; (b) tolerance study; (c) production P difference vs ABAQUS |

Scripts: `scripts/fe/run_s3c*.sh`, `s3_run_tpeel.py`, `s3c_bench.py`, `s3c_summary.py`, `s3c_plot.py`.

**Results [D].**
- Coarse wall time, 1 thread: 926 s (S3b path) → 657 → 572 → 179 → 102 → **91 s**. ABAQUS takes
  256 s on the same deck.
- The steps are: energy Newton, divergence check, element assembly, IM condensation, PARDISO-sym.
- The response is the same for all steps (window max 0.52 % vs ABAQUS, the coarse snap event).
- Tolerance: rtol = 1e-6 is the loosest that keeps |ΔP| ≤ 1e-5 P_peak on the same increments
  (2.3e-6).
- Coarse energy gate: E4 4.3e-5, E5 3.0e-4, E6 −1.1e-3. All pass; ABAQUS gives 2.4e-4, 5.1e-4,
  −1.2e-3.
- Production `sw_s40`: **490 s on 1 thread** (335 s on 4) against ABAQUS 2321 s.
  - P_peak +5.7e-6; window max |ΔP| 8.4e-5 P_peak, rms 1.3e-5.
  - E4–E6 equal to ABAQUS's.
  - Preliminary for S3d.

## S3d / S3e — production reference cases vs ABAQUS

Details: `docs/SOLVER_LOG.md` (S3d, S3e). Comparison: `src/tslid/analysis/compare_solvers.py`.
Tables and figure: `scripts/fe/s3de_compare.py`. Negishi rundirs `tslid_runs/solver_S3d/` and
`solver_S3e/`. Superseded outputs from before the energy-split fix are in `v1/` (identical P).

| file | content |
|---|---|
| `s3d_compare.csv` | `sw_s40`: default vs ABAQUS; toggles (Hencky, frame `global`, frame `midsurface`) vs ABAQUS and vs the default. All `compare_solvers.py` fields and gates |
| `s3e_compare.csv` | `sw_s58`, `sw_s21`, `as_s21_am3`, `as_s21_a7`: default vs ABAQUS |
| `s3de.png/pdf` | (P − P_ABAQUS)/P_peak vs Δ per case (compare_ld window); `sw_s40` toggles vs the default |
| `s3d_traction_location.csv/.png` | D23 item: where along the bond the S3a elastic traction difference sits |
| `s3d_timing.csv` | D25 timing protocol (one job, one node): ABAQUS default deck, ABAQUS with thinned field output, tslid.fe on 1 and 4 threads |

**Results [D]** (default = D24 configuration):

| case | max \|ΔP\|/P_peak | rms | ΔP_peak | Δℓ_cz (el.) | E4–E6 |
|---|---|---|---|---|---|
| `sw_s40` | 8.6e-5 | 1.3e-5 | +5.7e-6 | 0.00 | pass |
| `sw_s58` | 3.6e-4 | 8.5e-5 | −1.8e-5 | +0.01 | pass |
| `sw_s21` | 4.1e-5 | 7.7e-6 | −2.5e-5 | −0.01 | pass |
| `as_s21_am3` | 7.0e-5 | 1.1e-5 | −1.5e-5 | 0.00 | pass |
| `as_s21_a7` | 2.8e-5 | 7.9e-6 | −1.9e-5 | −0.01 | E5 fails in the energy_checks window (end-of-ligament collapse; explained in SOLVER_LOG S3e); passes over the compare_ld window |

All P gates (0.5 %, 0.1 %, one increment, 2 elements) and the 0.1 % target are met in every case.
- Toggles on `sw_s40`:
  - Hencky vs SVK: ≤ 2.7e-5 P_peak, so SVK stays.
  - Cohesive frame `global` = `midsurface` (≤ 2e-9 N apart): 1.2e-4 P_peak from the default
    corotational frame, and farther from ABAQUS (max 1.5e-4 vs 8.6e-5). Corotational stays.
- S3a traction difference: at the crack tip (x = 18.0–18.1 mm). 35–46 % of Σ|Δt| lies within
  2 mm of the tip, none within 2 mm of the bond end.
