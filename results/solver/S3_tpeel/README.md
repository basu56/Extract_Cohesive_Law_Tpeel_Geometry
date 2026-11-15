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
| tslid.fe staggered, "global" frame | TBD (job 45868159 still running when this was written) | | | | | | |

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
- First call (JIT) 2.0 s.
