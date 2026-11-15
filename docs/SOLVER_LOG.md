# Solver log — differentiable T-peel solver on tatva

One entry per step (S0, S1, …). Write an entry when the step's exit criteria are met, or when
a step is abandoned or changed. **Never edit an old entry.** If something later turns out to be
wrong, add a dated correction under that entry. This file records what each step gained us,
both for the manuscript and for the short contribution on the tatva module (D19, D20).

Labels: [D] demonstrated by a test or run in this repo, [H] hypothesis, [I] interpretation.

## Running summary

Update this table at each entry.

| Step | Status | Key verification result | Runtime / throughput | Gain (one line) |
|---|---|---|---|---|
| S0 environment | **done** 2026-11-14 | 16/16 tests on Bell, Negishi node, GitHub CI; `test_laws` OK on Py 3.6 | suite 14–15 s (Bell login, Negishi node) | one locked JAX/tatva env on every machine, with CI |
| S1 bulk element | **done**, 1 flagged item 2026-11-14 | Q4IM = CPE4I to 1e-8 (linear); NLGEOM v, M ≤ 4e-5; u_tip 1.25 % SVK / 0.30 % Hencky (own value) | ny=20 linear 1.8 s/solve; NL 2.6 s/incr (ABAQUS 50 s total) | locking-free bulk element matching CPE4I; constitutive gap quantified |
| S2 cohesive interface | **done** 2026-11-14 | 20 inviscid COH2D4 runs ≤ 6.8e-8 σ_c; viscous ≤ 4.8e-8 until D = 1; COH2D4 = nodal quadrature | 2000-increment element history ~0.5 s | ABAQUS cohesive behaviour reproduced exactly, with exact gradients in (K0, σ_c, G_c, α) |
| S3 T-peel forward vs ABAQUS | **S3a–S3e done** 2026-11-15 (S3a gate revised to 5e-5, D23); **1 explained E5 failure (as_s21_a7, energy_checks window at the end-of-ligament collapse)**; stopped for review | linear T-peel = ABAQUS to 3e-8; 5 reference cases: max \|ΔP\| ≤ 3.6e-4 P_peak, ΔP_peak ≤ 2.5e-5, ℓ_cz ≤ 0.01 el.; E4–E6 = ABAQUS over the compare_ld window | sw_s40, one job/node: 532 s (1 thread) and 383 s (4) vs ABAQUS 2292 s (0.23× / 0.17×) | ABAQUS-equivalent T-peel forward solve at < 1/4 of its cost, element-local and differentiable |
| S4 throughput (batched / GPU) | not started | | | |
| S5 gradients (adjoint) | not started | | | |

Reference ABAQUS costs:
- [D] ~20–25 min per T-peel run on 1 CPU (26k CPE4I, full output; `docs/STATUS.md`).
  - Correction 2026-11-15 [D]: the production reference runs took 2154–3338 s wall (36–56 min)
    on Negishi; WALLCLOCK in the .msg of `sw_s40` is 2321 s, `sw_s58` 3338 s, `sw_s21` 2154 s,
    `as_s21_am3` 2254 s and `as_s21_a7` 2395 s.
- [D] Licence-limited to ~12 concurrent jobs, i.e. ≈ 30 runs/h.
- [D] No gradients: finite differences need ≥ 2 runs per parameter.

## Entry template

Copy this template for each step.

```markdown
## Sx — <title> — <date> — commit <hash>

**Built.** What code and tests were added (files).

**Verification.** Each test with its threshold and measured value, [D].
Failed or partial tests are listed here too.

**Runtime.** Wall time of the relevant solve(s); hardware (laptop CPU / Negishi CPU / GPU);
for comparisons, the ABAQUS number on the same problem. Measured, not estimated.

**Gain.** What this step makes possible that was not possible before. Be concrete:
"we now know the Quad4 locking error at ny = 20 is X %", not "better accuracy".

**Cost / limitations.** What it does not do yet; known differences from ABAQUS; tatva
workarounds (see docs/solver/TATVA_NOTES.md).

**Open items.** Carried to the next step.
```

<!-- Entries start below this line. Newest at the bottom. -->

## S0 — Environment and skeleton — 2026-11-14 — commit 3c3fd07 (code), log in the following commit

**Built.**
- `pyproject.toml` (package `tslid`, src layout, extra `fe`: jax, tatva==0.11.6, jax-autovmap,
  numpy, scipy, matplotlib, pytest; `slow` marker), `uv.lock`, `.python-version` = 3.12.12.
- `src/tslid/fe/` with `__init__.py` (x64 on, `TATVA_VERSION`) and docstring stubs for
  `elements`, `kinematics`, `laws_jax`, `interface`, `meshing`, `solve`.
- `tests/fe/test_s0_smoke.py` (5 tests), `.github/workflows/tests.yml` + README badge,
  `scripts/slurm/run_fe_tests.sh`, `docs/solver/ENVIRONMENT.md`.
- Locked: Python 3.12.12, jax 0.12.0.1, tatva 0.11.6; uv 0.9.5.

**Verification.** [D]
- tatva version == 0.11.6: pass. x64 active (`jnp.zeros(1).dtype == float64`): pass.
- Quad4 `Operator.integrate(1.0)` on the unit square, 2×2 and 5×5 node meshes: |I − 1| ≤ 1e-14, pass.
- Extra test (not in the instructions): `import tslid, tslid.laws` does not load jax: pass.
- Full suite `pytest tests/test_laws.py tests/fe -m "not slow"`, 16 tests:
  - Bell login node (AMD EPYC 7502): 16 passed.
  - Negishi compute node a023 (AMD EPYC 7763), job 45855196: 16 passed.
  - GitHub Actions (ubuntu-latest), commits 9962f73, 6b39bea, 3c3fd07: success.
- `tests/test_laws.py` under the old interpreter (system Python 3.6.8, unittest), Bell and Negishi: 11/11 OK.
- Partial / deviation: the "laptop" leg was replaced by the Bell login node (user's choice
  this session); the laptop has not been tested. First Negishi job 45855040 also passed 16/16
  but ran Python 3.12.13 (`.python-version` was "3.12"); the patch was then pinned and re-run.

**Runtime.** Test suite wall time: 14.3 s (Bell login), 14.8 s (Negishi node, job 45855196;
26 s job elapsed). The first Negishi job took 67.7 s, 25.6 s of it in the first Quad4 test —
first-use JIT/import cost on a fresh environment [I]. No solver timings yet (S1).

**Gain.** Every machine (Bell, Negishi, CI) now runs the same locked JAX + tatva stack in
float64, and every push is tested automatically. The ABAQUS-side code is unaffected:
`tslid.laws` still runs on Python 3.6 and does not import JAX (guarded by a test).

**Cost / limitations.**
- `requires-python = ">=3.11"` applies to the whole package (not only the `fe` extra, which
  pyproject cannot express). Harmless because the ABAQUS side never installs the package;
  it uses `sys.path`.
- The lock forks by Python (3.11 → jax 0.10.2; 3.12 → jax 0.12.0.1), hence the exact Python pin.
- No tatva workarounds needed; `docs/solver/TATVA_NOTES.md` not created.
- Bell login node: one transient `os error 11` during the first `uv sync` (re-run worked).

**Open items.**
- GPU: Negishi partition `gpu` (g000–g004, 3 GPUs each) exists; GPU model and whether
  account `tomar` may use it are TBD (needed for S4).
- Laptop environment not yet verified.
- S1 waits for the user's go-ahead.

## S1 — Bulk element and kinematics — 2026-11-14 — code d57cfca, log in the following commit

**Built.**
- `kinematics.py`: plane-strain energies `linear`, `svk`, and `hencky` (log strain, power series
  of ln C, NaN-free at H = 0; added to test the SVK-vs-hypoelastic hypothesis).
- `elements.py`: `BulkModel` with Q4 (tatva Quad4, 2 × 2) and Q4IM (4 Wilson/Taylor incompatible
  modes as extra global unknowns, enhanced displacement gradient with j₀/j scaling; Simo–Armero
  F = F_c + H_enh under finite rotation).
- `solve.py`: `SparseNewton`, a Python-loop Newton with the tangent from `tatva.sparse.jacfwd`
  (pattern from the element dof map, distance-2 colouring: 18 colours for Q4, 34 for Q4IM) and a
  SciPy SuperLU solve.
- `meshing.py` (beam using the T-peel `graded()`, edge loads, 5-element patch); `benchmarks.py` (cantilever).
- ABAQUS side: `scripts/fe/build_cantilever_inp.py`, `extract_cantilever_odb.py`, `run_s1_abaqus.sh`;
  comparison `s1_compare.py`, `s1_plot.py`, `run_s1_compare.sh`.
- `docs/solver/ELEMENTS.md`; tests `tests/fe/test_s1_patch.py` (6), `test_s1_cantilever.py` (6 + 2 slow).

**Verification.** [D] Details in `results/solver/S1_bulk/README.md`.
1. Patch test (distorted 5-element patch, Q4 and Q4IM × linear and SVK): energy error ≤ 2e-16,
   nodal error ≤ 1e-18 mm, mode amplitudes 0. Threshold 1e-12: **pass**. SVK rigid rotation 0.3 rad
   gives zero energy.
2. Cantilever vs Timoshenko (reported only): Q4IM −0.12 … −0.48 %, ABAQUS identical; Q4 −19 % … −0.67 %.
3. Linear vs ABAQUS CPE4I (job 45856435): Q4IM tip deflection and reaction within ≤ 2.1e-8 at every
   ny (threshold 5e-4 at ny = 20): **pass**. Q4 locking: −19.3 / −11.1 / −2.29 / −0.195 % at
   ny = 2 / 4 / 8 / 20. Distorted mesh: Q4IM 4.7e-6 vs CPE4I (information).
4. NLGEOM to θ = 0.101 rad (P = 1400 N, 20 increments, tight ABAQUS tolerances):
   v_tip ≤ 3.6e-5, clamp moment ≤ 3.0e-5 (SVK); **pass**.
   **Flag:** u_tip relative to its own value is −1.25 % (SVK) and +0.30 % (Hencky), so it **fails 0.1 %**
   if the threshold is applied to that component alone. Relative to |v_tip| it is 4.5e-4 / 1.1e-4.
   Cause [D]: constitutive (changing the strain measure alone changes it 4× and flips its sign;
   it grows ∝ load²). Residual cause [H]: hypoelastic Cauchy vs hyperelastic Kirchhoff stress.
   Measured T-peel load-point rotation max = 0.0325 rad (sw_s40 UR3). At that rotation all
   differences are ≤ 1.3e-4 of |v| (SVK) and ≤ 4e-5 (Hencky).
5. Timing: see Runtime.
- Fast suite: 17 passed on Bell (2 slow deselected; the slow NLGEOM tests replicate the job above).

**Runtime.** Negishi node a001 (EPYC 7763, 4 cores): linear ny = 20 Q4IM (73 242 dofs) 1.8 s per
solve after JIT, 3.1 s first call; ABAQUS CPE4I 6.2 s wall (includes pre-processing and licence).
NLGEOM 20 increments: SVK 54 s including JIT (2.6 s per increment, 3 Newton iterations each);
ABAQUS 49.7 s. Hencky 202 s (≈ 4× SVK per tangent). Laptop not measured.

**Gain.** We now have a bulk element that reproduces ABAQUS CPE4I to 1e-8 in the linear regime and
to ≤ 4e-5 in P-relevant quantities under NLGEOM up to 3× the T-peel rotation. So any later P–Δ
mismatch in S3 cannot come from the bulk discretisation. We also know that plain Q4 at the
production mesh is 0.195 % too stiff (5× the floor), and that the remaining NLGEOM gap is
constitutive and second order in strain.

**Cost / limitations.**
- SVK and Hencky are hyperelastic; ABAQUS is hypoelastic. The gap is second order in strain and
  visible only in the axial tip displacement.
- The Hencky series is valid for strains ≲ 5 % (elastic Al only) and costs about 4× SVK.
- ABAQUS manual citations in `ELEMENTS.md` are from memory and unverified (manual not reachable).
- Each solve recompiles when the mesh changes (JIT); per-solve times exclude this.
- No tatva workarounds needed.

**Open items.**
- User: accept the u_tip flag (proposed: judge NLGEOM by v, M and |Δu|/|v|), and choose the S3 bulk
  energy. Proposed D21: Q4IM + Hencky, with SVK as a cross-check.
- User: confirm the ABAQUS manual sections cited in `ELEMENTS.md`.
- Laptop timing not done.

## S2 — Cohesive law and interface — 2026-11-14 — code and log in the S2 commit

**Built.**
- `laws_jax.py`: the α family in JAX, every parameter differentiable. s and g are ratios of power
  series for |α| < 0.5 (no 0/0, no cancellation), with double-`where` branches.
- `interface.py`: `CohesiveInterface` (separations from matched node pairs, global or deformed
  mid-surface frame, nodal or Gauss quadrature, energy with frozen damage, `update_history` with
  backward-Euler viscous D_v), `AlphaLawParams`, `TabularLawParams` (ABAQUS TABULAR table).
- `benchmarks.py`: `single_cohesive` (lax.scan over a prescribed path), `se_case` (rebuilds any
  ABAQUS single-element run from its JSON), `two_block_bar`.
- `solve.py`: multi-width dof maps; convergence scale now includes the initial residual, plus a
  roundoff-level step criterion (needed once the interface is fully debonded).
- ABAQUS side: `scripts/fe/build_se_deck.py` (new decks: shear, mixed, viscous, probe),
  `extract_se_odb.py`, `run_s2_abaqus.sh`; `s2_compare.py`.
- `docs/solver/INTERFACE.md`; tests `test_s2_law.py`, `test_s2_gradients.py`,
  `test_s2_single_element.py`, `test_s2_assembly.py`.

**Verification.** [D] Details in `results/solver/S2_interface/README.md`.
1. Material point vs `tslid.laws`: ≤ 9.8e-16 σ_c (threshold 1e-12): **pass**.
2. 14 Phase 0 runs: traction ≤ 6.8e-8 σ_c (thresholds 1e-6 / 2e-5): **pass**. Unloading ≤ 1.7e-8: **pass**.
   Dissipation vs G_c: ≤ 7.8e-6 monotonic; **1.04e-5 for the cyclic run (threshold 1e-5: marginal)**.
   It equals ABAQUS's own value (path-trapezoid error); the model's exact dissipation is G_c to 1e-12.
3. Mixed mode (shear, δ_s = δ_n; LINEAR, EXPONENTIAL and TABULAR): ≤ 4.8e-8 σ_c: **pass**.
4. Viscous: backward Euler = ABAQUS to ≤ 4.8e-8 σ_c until inviscid D = 1: **pass**. μ → 0 recovers
   inviscid to 3.9e-11. **Partial:** after D = 1, ABAQUS completes D_v → 1 faster (rule unknown);
   difference ≤ 3.4e-5 σ_c in the traction-free tail, negligible at production μ/Δt ≈ 8e-4.
5. Gradients: AD vs FD ≤ 1e-6 inside branches; finite at δ = 0, δ_0, δ_f, α = 0 and compression;
   dG/dG_c = 1 within 1e-8: **pass**.
6. Assembly: R = t·A to 7.3e-13: **pass**.
- Quadrature: COH2D4 is nodal (Newton–Cotes) [D, probe]; it is now the default.
- Fast suite: 78 passed on Bell (2 slow deselected).

**Runtime.** One 2000-increment single-element history: ≈ 0.5 s after JIT (lax.scan, Bell login
CPU), against ≈ 60 s for the ABAQUS run (job 45857972: 10 runs in 9 min including extraction).
The two-block bar needed up to 17 staggered passes per step in softening.

**Gain.** The JAX interface reproduces ABAQUS COH2D4 at the material and element level to the
precision of the ABAQUS output (≤ 7e-8 σ_c) for every representation, mode mix and unloading path
we use, and it is exactly differentiable in (K0, σ_c, G_c, α). With S1, both ingredients of the
T-peel model now agree with ABAQUS independently, so any S3 mismatch must come from the coupling
(geometry, frame under NLGEOM, history scheme, loading) and not from the constitutive pieces.

**Cost / limitations.**
- Post-failure viscous tail differs from ABAQUS (above).
- `frame="midsurface"` is implemented but untested (S3).
- History is frozen per solve. The staggered scheme converges but may need many passes; S3
  chooses staggered vs monolithic.
- Manual citations in `INTERFACE.md` unverified.
- Found in passing: `tslid.laws.g_alpha` loses up to 1.6e-12 near α = 1e-4. Harmless for our α
  values; not changed (the ABAQUS-side code is frozen by the rules of this work).

**Open items.**
- S3 instructions (not yet written): T-peel mesh, loading, staggered vs monolithic history,
  COH2D4 frame under NLGEOM, P–Δ vs ABAQUS σ_c = 40, α = 0.
- D21 (bulk energy) and the S1 u_tip flag await the user.

## S3a — T-peel mesh import, couplings, elastic gate — 2026-11-15 — commits e31227c…(S3a/S3b commit)

**Built.**
- `src/tslid/fe/inp_import.py`: dependency-free parser for the builder's flat decks. It reads
  nodes, CPE4I with a counter-clockwise check, the COH2D4 chain as (bot, top) pairs, sets,
  surfaces, couplings, sections, materials, the cohesive law, BCs and `*Static`. Unknown keywords
  are kept in `Deck.unparsed`.
- `src/tslid/fe/tpeel.py`: `TPeelModel`, the reduced unknowns q = [U, θ_RP], kinematic couplings
  by elimination (finite or linearised rigid motion), and the chain-rule reduction
  K_q = JᵀK_UJ + G. This keeps tatva's colouring on the mesh pattern (34 colours) instead of the
  ~150+ a dense RP row would force.
- `interface.py`: `frame="corotational"` / `"corotational_linear"`, the COH2D4 kinematics found by
  the probes below.
- `kinematics.py`: closed-form 2×2 Hencky (the S1 series version is kept as `hencky_series`).
- Scripts: `build_s3a_elastic.py`, `s3a_elastic.py`, `s3a_compare.py`, `run_s3a*.sh`,
  `build_coh_probe.py`, `run_coh_probe.sh`, `extract_tpeel_coh.py`; the extractors now read LE
  under NLGEOM.
- Tests: `tests/fe/test_s3a_tpeel_model.py` (11): import counts and area vs the builder, rigid
  translation, FD tangent including the θ rows, linear proportionality and equilibrium,
  closed-form Hencky vs logm, and the 7 COH2D4 probes.
- Formulation: `docs/solver/TPEEL.md`. Results: `results/solver/S3_tpeel/`.

**Verification.**
- [D] Import (`sw_s40`): 27,414 nodes, 26,046 CPE4I, 582 COH2D4, bulk area 1111.0068 mm²
  (analytic), bond 18.0 → 76.2. The elastic decks are byte-identical in mesh to `sw_s40.inp`.
- [D] All five reference runs exist on Negishi (`phase1a_sigc_sweep/sw_s40, sw_s58, sw_s21`,
  `phase1b_alpha_sweep/as_s21_am3, as_s21_a7`) with .inp, .json, .odb, `_response.csv` and
  `_profiles.npz`.
- [D] NLGEOM = NO elastic T-peel vs ABAQUS (tight controls), all 4 frames: RF2 ≤ 3.3e-8, UR3
  ≤ 3.0e-8, mouth opening ≤ 2.9e-8, full U field ≤ 4.2e-8, cohesive tractions ≤ 5.0e-8
  (ODB single precision).
- [D] **The S3a gate fails under NLGEOM.** Elastic, 16 increments, P = 331…7026 N; the real
  T-peel peak is 562 N:
  - SVK at 331 / 675 N: RF2 2.2e-5 / 2.7e-5; UR3 5.2e-5 / 6.2e-5; mouth 1.3e-5 / 1.5e-5.
    Gate 1e-5.
  - Hencky gives the same sign and magnitude (RF2 2.6e-5, UR3 4.6e-5 at 331 N), so the bulk
    energy is not the cause.
  - At 7026 N (≈ 1 % fibre strain): RF2 3.7e-4 (SVK), 3.6e-4 (Hencky).
  - [H] The residual is the Q4IM-vs-CPE4I NLGEOM floor already seen in S1 (tip deflection 3.6e-5,
    accepted in D21). It has not been isolated further.
- [D] **COH2D4 kinematics with a geometric gap.** The first NLGEOM comparison used the S2
  jump-based separation. It missed by 1.2e-4 at 331 N, the upper adherend sat 0.026 mm away from
  ABAQUS's, and the cohesive shear tractions were off by 0.4–1.5 % of the peak traction.
  - Seven single-COH2D4 probes (job 45868344) show that ABAQUS uses the current gap vector in the
    deformed mid-surface frame minus the reference gap. With NLGEOM = NO it uses the linearised
    form, δ_s = [[u]]·t₀ + ω g.
  - Example: a 0.1 rad rigid rotation with the 0.4 mm gap gives S = 0 in ABAQUS, and −339 MPa of
    shear with the jump-based form.
  - All 7 probes are reproduced by the new frames, and the linear T-peel field offset disappears.
- [D] Test suite: 146 fast tests pass on Bell (`tests/test_laws.py`, `tests/fe`).

**Runtime.**
- [D] Production elastic increment (159,014 unknowns, 34 colours, Negishi EPYC 7763, 4 cores):
  about 7 s with 3 Newton iterations (tangent 3.2 s, SuperLU 4.0 s). The first increment takes
  18 s, including JIT. Setup takes 2.5–6 s.
- ABAQUS: 32 s (4 increments, default controls), 45 s (tight), 123 s (16 increments, tight).

**Gain.**
- We now know the T-peel mesh, couplings, BCs and reaction are reproduced exactly in the linear
  case.
- The nonlinear model is within 2–3e-5 of ABAQUS at T-peel loads, about 200× below the S3d
  whole-curve gate.
- We found that COH2D4 treats a bond with geometric thickness corotationally. Without that, the
  interface would see a spurious shear separation of φ × 0.4 mm, comparable to δ₀ = 0.0047 mm at
  σ_c = 40.

**Cost / limitations.**
- The 1e-5 NLGEOM gate is not met (2–5×). The cause is not isolated beyond "not the bulk
  energy, not the interface kinematics".
- The probes did not test whether ABAQUS integrates the interface over reference or current
  length under stretch. We use reference length; the bond-line stretch is ≤ 1e-3.
- The ABAQUS manual citations are still unverified, but the probes now carry the evidence.

**Open items.**
- User decision: accept a 2–3e-5 NLGEOM floor at T-peel loads (consistent with S1/D21), or ask
  for a deeper Q4IM-vs-CPE4I NLGEOM investigation before S3c.
- The S3d "frame toggle" is resolved by the probes (corotational is the ABAQUS behaviour).
  S3d will still report global vs corotational on `sw_s40`, as instructed.

## S3b — Monolithic damage + displacement (incremental potential) — 2026-11-15 — same commits as S3a

**Built.**
- `laws_jax.s_integral`, `envelope_integral`: closed-form ∫s and ∫T for the α law, with a series
  ratio for |α| < 0.5.
- `TabularLawParams.envelope_integral`: exact per-segment quadratic integral.
- `interface.psi_incremental` and `CohesiveInterface.energy_incremental`:
  Φ_n = a(1−D_v,n)K0/2 δ_m² + bΨ_env(δ_m) + K0/2⟨−δ_n⟩², with a = μ/(μ+Δt), b = Δt/(μ+Δt) and
  Ψ_env the unloading branch up to δ* = max(δ_max,n, δ₀), then F(δ_m) − F(δ*).
- `TPeelModel(history="monolithic" | "staggered")`.
- `src/tslid/fe/incremental.py`:
  - Newton with backtracking line search on ‖r‖₂ (it also catches non-finite trial residuals);
  - ABAQUS-like incrementation: dt0, dtmax, ×1.5 after 2 increments of ≤ 4 iterations, halve
    after 16 iterations, dtmin abort, linear extrapolation;
  - per-increment logs of iterations, passes, cutbacks, line-search halvings and timings.
- `benchmarks.single_cohesive(history="monolithic")` and `two_block_bar(history="monolithic")`.
- Scripts: `s3_run_tpeel.py`, `run_s3b_coarse.sh`, `s3ab_plot.py`.
- Tests: `tests/fe/test_s3b_monolithic.py` (46).

**Verification.**
- [D] The derivation is checked rather than assumed: ∇Φ_n equals the `update_damage` traction to
  ≤ 1.2e-14 σ_c on 3000 random states per case. Cases: α ∈ {−3, ±1e-3, 0, 0.3, 3, 7}, analytic
  and tabular, inviscid and viscous; compression, unloading and δ_max at 0, δ₀, δ_f and beyond.
- [D] S′ = s to 1e-12; F(∞) + σ_c δ₀/2 = G_c to 1e-14 (α law). The tabular F differs from G_c by
  3e-7–4e-6 N/mm, which is the table's own representation error.
- [D] Gradient safety: first and second derivatives in (δ_n, δ_s) and first derivatives in
  (K0, σ_c, G_c, α) are finite at δ_m = 0, δ₀, δ*, δ_f, in compression and at α = 0.
- [D] **Gate: all 20 S2 single-element runs** are reproduced by the monolithic scheme within the
  S2 tolerances. They are identical to the S2 scheme to ≤ 1e-12 σ_c.
- [D] **Gate: two-block test**:
  - R = tA, the law is reproduced to 1e-12 σ_c, D = 1 at the end, and the reactions equal the
    staggered ones to 1e-10;
  - one solve per step, with 2–4 Newton iterations;
  - total iterations 43 vs 183 staggered (α = 0) and 64 vs 160 (α = 3); staggered needed up to
    16/17 passes.
- [D] Coarse damaged T-peel (σ_c = 40, α = 0, D16 load settings, 7,306 unknowns), full step:
  - tslid.fe monolithic (COH2D4 kinematics): 1778 increments, 321 cutbacks, 2551 Newton
    iterations (at most 8 per increment), 602 line-search halvings; completed.
  - ABAQUS on the same deck: 1479 increments, 232 cutbacks, 2910 iterations.
  - P_peak −2.2e-5, Δ_peak equal to within one increment, whole-curve rms 2.9e-4 P_peak. The
    maximum, 0.59 %, comes from single-element snap events of the coarse bond, at δ = 1.39.
  - The `global`-frame run: P_peak −1.2e-4, rms 4.1e-4.
- Staggered on the coarse T-peel (Negishi job 45868159) was still running when this entry was
  written (more than 30 min, against 8.5 min for the monolithic run with the same frame). Its
  numbers go into a dated note below this entry.

**Runtime.**
- [D] Coarse damaged run: 670 s (tangent 190 s, solve 47 s, history 22 s, about 410 s line-search
  residuals, J products and Python overhead), against ABAQUS 256 s on the same deck. Negishi,
  4 cores. Not profiled or optimised yet; that is S3c.

**Gain.**
- One Newton solve per increment with the exact ABAQUS damage and viscous update; the staggered
  loop is gone (2.5–4× fewer iterations in the two-block test).
- The T-peel runs end to end with ABAQUS-like incrementation, and peak load agrees to 2e-5 on the
  coarse deck.
- The incremental potential is a true energy, so energy-based line search and (S5) implicit
  differentiation of the minimisation apply directly.

**Cost / limitations.**
- The coarse run is 2.6× slower than ABAQUS, before any S3c optimisation (condensation, solver
  choice, tolerance, overhead).
- The softening tangent is indefinite in propagation. Cutbacks there are of the same order as
  ABAQUS's.

**Open items.** S3c: profiling, IM condensation, linear-solver benchmark, tolerance study, and the
E4–E6 gate on the coarse deck.


*Note 2026-11-15 (staggered coarse run).* [D] Negishi job 45868159 (staggered, `global` frame, same
coarse deck and options) hit its 4 h SLURM time limit without finishing. The monolithic run with
the same frame took 515 s. The driver writes its CSV/JSON only at the end, so there are no
increment counts. The only staggered numbers are the partial Bell run in
`results/solver/S3_tpeel/README.md`: in 150 increments it reached t = 11.88 s, against 17.06 s for
the monolithic run. Conclusion: on the T-peel, staggered history is at least 28× slower than
monolithic. It is kept only for cross-checks. S3c will make the driver write records
incrementally, so a run that times out still leaves data.

*Note 2026-11-15 (S3c profile; corrects the S3b runtime paragraph).* [D] The S3b cost split
attributed the unaccounted ~410 s of the 670 s coarse run to "line-search residuals, J products
and Python overhead". The S3c full-run profile (Negishi job 45898677, same code and deck, 753 s
with cProfile) shows instead that **only 297 s are spent in converged increments; ~456 s (61 %) are
the 321 abandoned attempts before cutbacks**. Each attempt ran the full 16 iterations, often with
8 line-search halvings each: 9979 tangents in total, of which ~5650 belong to abandoned attempts.
The per-phase totals over the whole run were: jacfwd 278 s, SuperLU 123 s, gradient 84 s, numpy
lift/J and J^T K J reduction 179 s, eager history update 47 s. The S3b increment log had counted
converged increments only. The S3b "Cost / limitations" paragraph should be read with this
correction.

## S3c — Coarse T-peel with damage: profiling, performance, tolerance, energy gate — 2026-11-15 — commits a63fbc0…034fe59 (code), log in the following commit

Done in the user's order of 2026-11-15 (D23, `S3_INSTRUCTIONS.md` update). Each gain is measured
separately. Timed runs are pinned to 1 thread with `scripts/fe/pin_threads.sh`: Negishi allocates
CPUs by memory and sets no cpuset, and ABAQUS ran with `cpus=1`. Negishi AMD EPYC 7763. Rundir
`/scratch/negishi/basu56/tslid_runs/solver_S3c/`. Tables in `results/solver/S3_tpeel/s3c_*.csv`,
figure `s3c.png`.

**Built.**
- `s3_run_tpeel.py` writes records as the run goes: one CSV row per increment, flushed, and the
  JSON summary and profiles every N increments. It adds phase timers, cProfile, failed-attempt
  accounting, JIT cache sizes and schedule replay.
- `tpeel.py`: phase timers, `lift_jax` and a jitted q-energy (`value_and_grad`).
- `incremental.py`: `newton_energy` (Armijo / residual acceptance, full step first), the
  divergence check, `on_increment`, `schedule`.
- `assembly.py` (`ElementSystem`): element-local Hessians with the coupling inside the element,
  IM condensation with a closed-form 4 × 4 inverse, a jitted history update, the bulk energy.
- `linsolve.py`: SuperLU, SuperLU-MMD, PARDISO and PARDISO-sym, with the symbolic analysis reused
  per pattern. `pypardiso` is added to the `fe` extra and the lockfile (x86_64).
- `tpeel_output.py`: ABAQUS-compatible `_response.csv`, `_profiles.npz`, `.json`, `.sta`.
- Scripts: `run_s3c.sh`, `run_s3c_ladder.sh`, `run_s3c_tol.sh`, `run_s3c_prod.sh`,
  `run_s3c_bench.sh`, `s3c_bench.py`, `s3c_summary.py`, `s3c_plot.py`, `pin_threads.sh`.
- Tests: `tests/fe/test_s3c_assembly.py`, 9 tests.

**1. Profile (S3b code, coarse, job 45898677) [D].**
- Converged increments take only 297 s of the 753 s. **About 456 s are the 321 abandoned
  attempts before cutbacks**: 16 iterations each, often with 8 line-search halvings.
- Totals over the run: sparse jacfwd 278 s (9979 calls), SuperLU 123 s, gradient 84 s, numpy
  lift/J and JᵀKJ 179 s, eager history update 47 s.
- No recompilation: every jitted function has a cache size of 1, both here and in all S3c runs.
- Corrects the S3b attribution; see the dated note under S3b.

**2. Gain ladder (coarse, all configurations in one job, 1 thread; two copies, jobs
45900830/45900831; mean, A/B spread ≤ 12 % before item 3 and ≤ 1 % after) [D].**

| step | wall | converged / abandoned | iterations (all attempts) | gain |
|---|---|---|---|---|
| S3b path (residual line search, jacfwd, SuperLU) | 926 s | 391 / 534 s | 7687 | – |
| item 2: jitted q-energy, energy Newton | 657 s | 236 / 420 s | 6946 | −29 % |
| item 2b: divergence check (I₀ = 4) | 572 s | 251 / 321 s | 5792 | −13 % |
| item 3: element-local assembly | 179 s | 74 / 104 s | 5785 | 3.2× |
| item 4: IM condensation (7282 → 2802 unknowns) | 102 s | 41 / 61 s | 5786 | 1.74× |
| item 5: PARDISO symmetric, symbolic reuse | **91 s** | 37 / 54 s | 5784 | −11 % |
| (item 5 without condensation) | 110 s | | 5781 | |
| ABAQUS, same deck | 256 s | | 2910 | |

- Every step has the same converged response: in the compare_ld window (δ ≤ 1.2756 mm) the
  max |ΔP| vs ABAQUS is 0.52 % P_peak (a coarse snap event, also present in S3b) and the rms is
  3.0e-4.
- From item 2 on, the energy-Newton variants agree with each other to ≤ 3.5e-12 P_peak in the
  window. Their increment path differs from S3b's (1684–1691 vs 1778 increments), which moves
  snap events by ≤ 0.15 % P_peak.
- Outside the window, the end-of-ligament collapse (P 176 → 4 N over Δδ = 3e-4 mm) sits 5e-5 mm
  apart between the codes, which reads as 3.9 % at one grid point.

**3. Production micro-benchmarks (`sw_s40`, 158,954 free / 54,770 condensed, state at 0.06 T,
bench2 jobs 45900589/45900590) [D].**

| | 1 thread | 4 threads |
|---|---|---|
| tangent: jacfwd + JᵀKJ (S3b) | 1.77 s | 0.93 s |
| tangent: element | 0.217 s | 0.116 s |
| tangent: element, condensed | 0.218 s | 0.119 s |
| solve, free: SuperLU / SuperLU-MMD / PARDISO / PARDISO-sym | 1.22 / 0.34 / 0.19 / 0.144 s | 1.25 / 0.34 / 0.076 / 0.062 s |
| solve, condensed: same order | 0.53 / 0.17 / 0.12 / 0.088 s | 0.53 / 0.18 / 0.040 / 0.031 s |

- The element tangent equals jacfwd to 6.3e-14 (coarse 1.3e-14). The element residual equals
  the S3b one to 7.3e-13.
- Backward errors are ≤ 1.8e-13, with 0 perturbed pivots. The first PARDISO call includes the
  analysis: 0.19–0.43 s.
- Node-to-node variation is up to 1.8×: the same jacfwd tangent measured 3.12 s in another job.
  Hyperthread siblings are shared between jobs. Ratios within a job are reliable.
- The first condensation (batched `jnp.linalg.solve`) cost 0.16 s more per tangent than no
  condensation. The closed-form 4 × 4 inverse removed that overhead.
- UMFPACK and CHOLMOD were not benchmarked: SuiteSparse is not available on Negishi (no module,
  no conda), and installing it via micromamba was blocked by the session's permission settings.

**4. Condensation exactness [D].** The condensed step solves the full system to backward error
5e-16, the same as a direct full solve. Forward agreement with the full solve is 5e-11. It
cannot be 1e-12: cond(K) ≈ 3e8, and two SuperLU solves of the same full system already differ
by 8.6e-11. The test checks a backward error ≤ 1e-14 and a forward difference ≤ 1e-9.

**5. Tolerance (coarse, final configuration, job 45900832) [D].** Each rtol was replayed on the
increments of the rtol = 1e-10 run, with max |ΔP|/P_peak at identical increments in the window:

| rtol | 1e-9 | 1e-8 | 1e-7 | **1e-6** | 1e-5 | 1e-4 |
|---|---|---|---|---|---|---|
| max \|ΔP\|/P_peak | 3.0e-9 | 1.1e-8 | 3.2e-7 | **2.3e-6** | 2.3e-5 | 2.9e-3 |
| iterations | 2293 | 2217 | 2130 | 1896 | 1522 | 1160 |

- **The loosest tolerance meeting ≤ 1e-5 is rtol = 1e-6.** It is proposed as D24.
- With their own incrementation, runs at rtol ≥ 1e-8 differ from the reference by up to
  1.5e-3 P_peak in the window. That is not the tolerance: a different cutback pattern moves
  single-element snap events on this coarse bond.
- [I] This incrementation-path sensitivity of the coarse deck is relevant to S4, where
  increments are reduced.

**6. Coarse energy gate (final configuration, rtol 1e-10, job 45901165; `energy_checks.py`
unchanged) [D].**

| | tslid.fe | ABAQUS (same deck) | tolerance |
|---|---|---|---|
| E4 balance (window) | 4.3e-5 | 2.4e-4 | 1 % |
| E5 (ALLVD + ALLCD)/ALLDMD | 3.0e-4 | 5.1e-4 | 1 % |
| E6 dissipation slope / (G_c W) − 1 | −1.14e-3 | −1.19e-3 | 2 % |
| full-record balance | 0.023 | 0.024 | – |
| ℓ_cz | 2.691 ± 0.033 mm (5.4 el.) | 2.688 ± 0.037 mm | – |

- Pass. ALLDMD is the closed-form inviscid dissipation G_d(δ_max). An earlier trapezoid version
  gave E4 = 2.0e-3 (`elc2`, superseded); see `tpeel_output.py`.

**7. Production timing (measured, full `sw_s40` runs; jobs 45901170, 45902505, 45902506) [D].**

| run | increments / cutbacks | iterations (all) | wall | split |
|---|---|---|---|---|
| ABAQUS (`cpus=1`) | 839 / 9 | 1079 | 2321 s | – |
| tslid.fe, rtol 1e-10, 1 thread | 842 / 10 | 1827 | 754 s | Hessian 430, solve 209, (E, r) 53, outputs 51, abandoned 38 s |
| **tslid.fe, rtol 1e-6, 1 thread** | 838 / 8 | 1141 | **490 s = 0.21× ABAQUS** | Hessian 270, solve 124, (E, r) 38, outputs 49 s |
| tslid.fe, rtol 1e-6, 4 threads | 838 / 8 | 1141 | 335 s | Hessian 172, solve 37 s |

- Accuracy, preliminary for S3d, since the formal comparison script and toggles are S3d's:
  P_peak +5.7e-6, Δδ_peak −5e-5 mm.
- In the compare_ld window: max |ΔP| 8.4e-5 P_peak and rms 1.3e-5, for both tolerances.
  rtol 1e-6 vs 1e-10: 3.0e-6.
- E4 2.6e-5, E5 7.1e-5, E6 −1.3e-3, the same as ABAQUS's. ℓ_cz 26.9 elements in both.
- Setup 3.5 s, first call (JIT) 7 s.
- Projection: none needed; measured directly at 0.21× (1 thread) and 0.14× (4 threads) of
  ABAQUS's wall time. This is below the 2× threshold, so no joint decision on further
  optimization is needed.

**Gain.**
- The coarse damaged T-peel is 10× faster than at S3b and 2.8× faster than ABAQUS on 1 thread.
- The production reference case runs in 8 min on 1 thread (5.6 min on 4) instead of ABAQUS's
  39 min, with P within 1e-4 P_peak and identical energy diagnostics.
- Runs now leave ABAQUS-compatible outputs, so `energy_checks.py` and `compare_ld.py` run
  unchanged.
- The path is element-local and has a fixed pattern. That is the structure batched (vmap over
  parameters) and GPU solves in S4 need.

**Cost / limitations.**
- About 60 % of the coarse wall time is still abandoned attempts before cutbacks (54 of 91 s).
  Production has only 8 cutbacks.
- The increment strategy is ABAQUS-like by design (D22). Changing it is S4's question.
- Production time is now dominated by the element Hessian, 55 % on 1 thread: AD Hessians of the
  SVK + enhanced-strain element.
- The output writer costs ~58 ms per increment at production size, ~10 %.
- Timings vary up to 1.8× between Negishi nodes. Absolute numbers carry that uncertainty;
  ratios measured in one job do not.
- UMFPACK and CHOLMOD are not benchmarked.
- The divergence check is our rule, not a reproduction of ABAQUS's controls.

**Open items.**
- User review of D24 (configuration, rtol = 1e-6).
- S3d: `compare_solvers.py`, the formal `sw_s40` gates, the SVK/Hencky toggle, the location of
  the S3a traction difference along the bond (D23 item), and then the other four reference
  cases.

*Note 2026-11-15 (S3d; corrects the S3c entry, item 3).* The S3c entry attributes the
node-to-node timing variation to "hyperthread siblings shared between jobs". That is wrong:
Negishi `cpu` nodes have no SMT (`scontrol show node`: ThreadsPerCore = 1, 2 × 64 cores). [H] The
variation is more likely from other jobs sharing the L3 cache and memory bandwidth of the node,
and from clock-frequency differences. The conclusion stands: ratios measured within one job are
reliable, absolute numbers across jobs are indicative.

## S3d — Production reference case `sw_s40` vs ABAQUS — 2026-11-15 — commits b346dd2…cdf0817 (code), log in the following commit

D24 configuration, agreed at the S3c review: element assembly, IM condensation, PARDISO-sym with
analysis reuse, divergence check, rtol 1e-6. The production mesh is imported from the `sw_s40`
deck, D16 settings. Rundir `/scratch/negishi/basu56/tslid_runs/solver_S3d/`. Tables
`results/solver/S3_tpeel/s3d_compare.csv`, `s3d_timing.csv`, `s3d_traction_location.csv`; figure
`s3de.png`.

**Built.**
- `src/tslid/analysis/compare_solvers.py` (Python 3.6 compatible). It reuses
  `compare_ld.load`, `energy_checks.check` and `zone_fronts`, and reports:
  - whole-curve max |ΔP| on the union of both records' δ in the compare_ld window, and the rms;
  - ΔP_peak and Δδ_peak against the local increment;
  - the elastic slope, fitted on the rising branch;
  - ℓ_cz, the crack tip and σ_c front x(δ), and E4–E6 for both solvers;
  - the gates.
- `linsolve.make_solver`: automatic SuperLU-MMD fallback when pypardiso/MKL is unavailable, with
  a test.
- Driver defaults are now D24.
- `tpeel_output.py`: viscous-overstress energy split (see S3e).
- `scripts/fe/s3d_traction_location.py`, `thin_output_deck.py`, `run_s3d_timing.sh`,
  `s3d_timing_summary.py`, `s3de_compare.py`.
- Tests: 156 pass (fast suite), CI green.

**Verification, gates [D]** (`sw_s40`, compare_ld window 0–1.2743 mm):

| quantity | tslid.fe vs ABAQUS | gate |
|---|---|---|
| whole-curve max \|ΔP\|/P_peak | 8.6e-5 (at δ = 0.136; post-peak 5.6e-5) | ≤ 5e-3; target 1e-3 met |
| rms | 1.3e-5 | – |
| ΔP_peak | +5.7e-6 | ≤ 1e-3 |
| Δδ_peak | −4.4e-5 mm (increment 2e-3 mm) | within one increment |
| elastic slope | 3259.18 vs 3259.29 N/mm (−3.5e-5) | – |
| ℓ_cz | 2.6869 vs 2.6869 mm (0.00 el.) | ≤ 2 elements |
| crack tip / σ_c front | max 0.0016 / 0.0015 mm (0.02 el.) | – |
| E4 / E5 / E6 | 2.6e-5 / 7.2e-5 / −1.3e-3 (ABAQUS 1.2e-5 / 7.1e-5 / −1.3e-3) | pass both |

The target is met. The difference is close to the S3a elastic floor (2–3e-5 in RF2), i.e. near
the Q4IM-vs-CPE4I NLGEOM floor, not the 0.04 % mesh floor.

**Toggles (one run each) [D].**
- **SVK vs Hencky:** max |ΔP| 2.7e-5 P_peak (≤ 1e-4), so **SVK stays the default** (D21
  acceptance test passed). Hencky is 1.4× slower (498 vs 364 s, 4 threads, separate jobs).
- **Cohesive frame:**
  - `global` and `midsurface` (rotated without gap correction, the frame item 5 describes) are
    identical to 2e-9 N.
  - [I] With equal normal and shear stiffness the energy depends only on δ_m and the compressive
    part, and a rotation leaves |δ| unchanged.
  - Both are 1.2e-4 P_peak from the default corotational frame (with gap correction, S3a) and
    farther from ABAQUS: max 1.5e-4 vs 8.6e-5, rms 3.6e-5 vs 1.3e-5.
  - **The corotational frame stays the default.**

**D23 item: where the S3a traction difference sits [D].**
- Elastic NLGEOM run, frames at 331 and 675 N. The largest normal (2.6e-4 / 3.0e-4 of max S22)
  and shear (1.4e-4 / 1.6e-4) differences are **at the crack tip**, x = 18.0–18.1 mm, the first
  bond node, which is also the location of max S22.
- 35–46 % of Σ|Δt| lies within 2 mm of the tip; none within 2 mm of the bond end (x = 76.2).

**Timing, D25 protocol [D]** (job 45907082, node a017, AMD EPYC 7763; one job, runs in sequence,
each pinned to CPU 99 or 99–102 of the job's 16 dedicated cores):

| run | wall (whole command) | analysis | increments / cutbacks / iterations | output |
|---|---|---|---|---|
| ABAQUS, default deck (`cpus=1`) | 2292 s | 2273 s | 839 / 9 / 1079 | 722 MB |
| ABAQUS, field output thinned to tslid.fe's (`cpus=1`) | 2183 s (0.95×) | 2172 s | 839 / 9 / 1079 | 51 MB |
| tslid.fe, 1 thread | **532 s (0.23×; 0.24× of thinned)** | 489 s | 838 / 8 / 1141 | 6 MB |
| tslid.fe, 4 threads | **383 s (0.17×; 0.18× of thinned)** | 339 s | 838 / 8 / 1141 | 6 MB |

- tslid.fe setup is 4.8 / 7.3 s, first call (JIT) 9.0 / 6.4 s, output writing 41 / 37 s (1 / 4
  threads).
- The timed 1-thread run passes all gates with the same numbers as above.
- An exclusive node (128 cores) exceeds the tomar group cap of 64 hp_cpu and never starts. The
  job therefore had 16 dedicated cores on a shared node.
- [H] Other jobs on the node may share L3 and memory bandwidth (no SMT on Negishi).
- The cross-job numbers in S3c and S3e (e.g. 490 s) are indicative; these are the publishable
  ratios.
- Thinning ABAQUS's field output saves only 5 %: ABAQUS's cost is the solution, not the output.

**Gain.**
- On the production reference case, tslid.fe reproduces ABAQUS's P–Δ to 8.6e-5 P_peak and its
  energy diagnostics, ℓ_cz and crack-tip history.
- It does so in 0.23× ABAQUS's wall time on 1 core (0.17× on 4), measured in one job on one node.
- `compare_solvers.py` makes the cross-solver comparison a single command for S4a.

**Cost / limitations.**
- The pinned node was shared with other jobs (group cap).
- The ALLDMD/ALLCD split is ours (exact in the continuum); ABAQUS's split is not documented to us.
- Hencky costs 1.4× SVK.

**Open items.** See S3e.

## S3e — The four extreme cases — 2026-11-15 — same commits as S3d

D24 defaults (element assembly, IM condensation, PARDISO-sym, divergence check, rtol 1e-6), 4
threads. `compare_solvers.py` against the ABAQUS references (`phase1a_sigc_sweep`,
`phase1b_alpha_sweep`). Rundir `/scratch/negishi/basu56/tslid_runs/solver_S3e/`. Tables in
`results/solver/S3_tpeel/s3e_compare.csv`, figure `s3de.png`. Reported per case, not averaged.

**Verification [D]** (window = compare_ld window; ℓ_cz in elements of h_coh = 0.1 mm):

| case | regime | max \|ΔP\|/P_peak (rms) | ΔP_peak | Δδ_peak (mm) | Δℓ_cz | max \|Δx_tip\| | E4 / E5 / E6 ours (ABAQUS) | gates |
|---|---|---|---|---|---|---|---|---|
| `sw_s58` | short zone, ℓ_cz = 10.8 el. | 3.6e-4 (8.5e-5) | −1.8e-5 | −4e-5 | +0.01 | 0.22 el. | 3.8e-5 / 9.3e-5 / −3.6e-4 (1.9e-5 / 9.0e-5 / −3.6e-4) | pass, target met |
| `sw_s21` | long zone, ℓ_cz = 63 el. | 4.1e-5 (7.7e-6) | −2.5e-5 | −4e-5 | −0.01 | 0.01 el. | 2.1e-5 / 7.2e-5 / −3.2e-3 (4.7e-6 / 1.0e-4 / −3.2e-3) | pass, target met |
| `as_s21_am3` | α = −3 | 7.0e-5 (1.1e-5) | −1.5e-5 | −4e-5 | −0.00 | 0.01 el. | 2.3e-5 / 5.4e-5 / −1.9e-3 (2.4e-6 / 5.3e-5 / −1.9e-3) | pass, target met |
| `as_s21_a7` | α = 7, ℓ_cz = 144 el. > 2H | 2.8e-5 (7.9e-6) | −1.9e-5 | −4e-5 | −0.01 | 0.01 el. | 7.7e-3 / **3.4e-2** / −1.8e-2 (2.6e-6 / 4.8e-5 / −1.7e-2) | **E5 fails** (explained below); P gates pass, target met |

Elastic slopes agree to ≤ 4.4e-5 in all cases. The largest P difference is in the short-zone case,
where single-element snaps are largest (10.8 elements per ℓ_cz), and is still 14× under the
target.

**The `as_s21_a7` E5 failure.** `energy_checks.py` ends its E4/E5 window when the traction-free
tip passes L − 15 mm. With the α = 7 tail, the tip lags so far behind the σ_c front that the
end-of-ligament collapse (δ ≈ 1.3776 mm) begins inside that window.
- ABAQUS goes from P = 175 to 30 N in one frame. Its last window frame (694) is therefore before
  the collapse: E5 4.8e-5.
- tslid.fe resolves the collapse in several increments. Its last window frame (702) is the first
  collapse frame (P 170 → 106 N), in which the viscous overstress works over the unstable step.
- Over the following frame ABAQUS shows the same thing: ALLCD jumps from 0.013 to 9.96 N·mm, and
  its balance residual reaches −5.2e-2.
- Up to the frame before the collapse (701), ours gives E4 7e-6 and ALLCD/ALLDMD 6.4e-5.
- Over the compare_ld window used by the P gates (σ_c front at L − 15 mm, δ ≤ 1.2123):

  | | E4 | E5 |
  |---|---|---|
  | tslid.fe | 2.1e-5 | 4.7e-5 |
  | ABAQUS | 2.6e-6 | 4.7e-5 |

  The same holds in all five cases: E5 within 3 % of ABAQUS's value, E4 ≤ 3.8e-5.
- [I] The failure is a window-definition artifact at the end-of-ligament collapse, not a solver
  error. `energy_checks.py` is shared analysis code and is not changed without the user.

**Energy split corrected during S3e.**
- The S3c split, ALLDMD = G_d(δ_max) with a trapezoid ALLCD = ∫Y d(D_v − D), double-counted the
  energy still stored through a lagging D_v.
- On the first S3e runs it gave E4 1.9e-2 and a negative E5 (−1.6e-2) for `as_s21_a7`, and
  E5 = −4.8e-5 for `sw_s21`.
- It is replaced by ALLDMD = G_d(δ_max) − (D − D_v)·Y and ALLCD = the work of the viscous
  overstress (D − D_v)K0δ⁺ (`tpeel_output.py`; formulation in its docstring). With this,
  ALLSE_coh + ALLDMD + ALLCD = ∫t·dδ holds exactly, and ALLCD ≥ 0.
- All S3d/S3e runs were repeated with it. The superseded outputs are kept in `solver_S3d/v1`
  and `solver_S3e/v1`; P is identical.
- The S3c coarse energy gate used the old split. Its values were within the S3c tolerances, but
  only the S3d/S3e values use the final split.

**Runtime (indicative, separate jobs, 4 threads)** [D]:

| case | ours: increments / cutbacks / iterations, wall | ABAQUS: increments / cutbacks / iterations, wall |
|---|---|---|
| `sw_s58` | 875 / 25 / 1972, 653 s | 1009 / 87 / 1790, 3338 s |
| `sw_s21` | 834 / 8 / 1007, 520 s | 824 / 6 / 1004, 2154 s |
| `as_s21_am3` | 836 / 7 / 1238, 390 s | 845 / 11 / 1105, 2254 s |
| `as_s21_a7` | 827 / 6 / 969, 312 s | 812 / 2 / 1182, 2395 s |

- The same runs measured 315–805 s in the first (v1) set; node-to-node variation is up to 1.6×.
- PARDISO had 0 perturbed pivots in every run.

**Gain.** The five reference cases span ℓ_cz from 10.8 to 144 elements and α from −3 to 7. In all
of them our P–Δ matches ABAQUS to ≤ 3.6e-4 P_peak and P_peak to ≤ 2.5e-5. The crack tip agrees to
≤ 0.22 elements and ℓ_cz to ≤ 0.01 elements.

**Cost / limitations.**
- One energy-gate failure, explained above, which needs a user decision on the E4/E5 window.
- The energy split is ours (documented, exact in the continuum); ABAQUS's split is not known to
  us.

**Open items (user).**
- The E4/E5 window for long-tail laws. Proposal: evaluate E4/E5 over the compare_ld window (σ_c
  front at L − 15 mm) for both solvers, or end the window before the first frame of the
  end-of-ligament collapse.
- Then S4a: the 21 σ_c and α sweep runs against ABAQUS (D25).
