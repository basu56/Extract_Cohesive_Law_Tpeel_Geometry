# S5.0 / S5a — law parameters as inputs, direct sensitivities

Produced by `scripts/fe/s5a_analysis.py` from the Negishi runs below. The 2026-11-15 jobs were
45934469–45934476, at commit 2cd473f. Derivation: `docs/solver/ADJOINT.md`. Log: `docs/SOLVER_LOG.md`, S5.0/S5a.

Parameters θ = (ln K0, ln σ_c, ln G_c, α). Sensitivities are dP/dθ in N per unit of θ.

Window: 0 ≤ δ ≤ d_end of the ABAQUS reference, i.e. until the σ_c front reaches L − 15 mm.
- Coarse `c_s40` (σ_c 40, α 0; S3c deck): d_end = 1.2756 mm.
- Production `sw_s40`: d_end = 1.2744 mm.

All runs used one pinned thread in the deterministic environment (`scripts/fe/det_env.sh`):
MKL_CBWR=COMPATIBLE and single-threaded XLA.

| file | content |
|---|---|
| `s5_gate.csv` | S5.0 gate 2, five S3 reference cases. Old code (7e21d1a) vs new vs new again, in one job each: schedules, bitwise flags, max \|ΔP\|/P_peak (whole run; on the union of δ if the schedules differ), wall times |
| `s5a_determinism.csv` | coarse, rtol 1e-10. Repeat (b1/b2), PARDISO vs SuperLU, replay of b1's schedule (r0); compared on the common prefix of identical increments, which covers the window |
| `s5a_stepstudy.csv` | direct vs central finite differences with replay, h = 1e-3 … 1e-7, per parameter: curve-wise ‖S−S_FD‖₂/‖S_FD‖₂ and point-wise maximum in the window. Rows without errors: the ± runs left b1's schedule inside the window (cutback), so they are invalid |
| `s5a_tolerance.csv` | rtol 1e-6 vs 1e-10 sensitivities. On b1's schedule (tolerance alone), and on its own schedule (interpolated in δ; pre-peak separately) |
| `s5a_cost.csv` | forward vs forward + sensitivities, coarse (×2) and production, one job (45934476) |
| `s5a_sw_s40_S.npz` | production `sw_s40`, rtol 1e-6: per increment time, delta, P, dP (n, 4), dmouth_u/l, n_switch, n_fail, res_free, finite, wall, theta, names, d_end_window |
| `s5a.png/.pdf` | (a) step study; (b) coarse sensitivities and FD h = 1e-6 in the window; (c) production sensitivities with P |

Main numbers (details in SOLVER_LOG):

**S5.0 forward gate.**
- New vs new again: bit-identical in 5/5 cases.
- Old vs new: the first difference is 1 ulp (5e-16 P_peak) at increment 2. In the window, the
  maximum difference is 3.8e-12 to 6.5e-12.
- Two cases depart from the old schedule after the window (δ 1.38 and 1.39 mm).
- The ≤ 1e-14 gate is **not met** as stated.

**Determinism.**
- A repeated run is bit-identical, in P and in all four sensitivities.
- PARDISO vs SuperLU: P to 2e-12; S to 3e-10 – 1e-9.

**Step study.** The best curve-wise errors are below. h = 1e-3 is invalid for every parameter
(cutbacks in the ± runs); h = 1e-4 is also invalid for ln G_c.

| parameter | best curve-wise error | at h |
|---|---|---|
| ln K0 | 1.3e-6 | 1e-6 |
| ln σ_c | 1.02e-5 | 1e-7 (1.03e-5 at 1e-6) |
| ln G_c | 6.1e-7 | 1e-7 |
| α | 4.0e-6 | 1e-6 |

**Tolerance.**
- rtol 1e-6 vs 1e-10 on the same schedule: 3.6e-5 – 2.4e-4.
- On its own schedule: 2e-8 – 5e-8 before the peak, but 0.32–0.43 over the whole window. The
  element-failure spikes fall at different increments.

**Cost.** Sensitivities add 39–48 % on the coarse run and 94 % on production `sw_s40`
(541 → 1051 s on 1 thread).

## S5b / S5c (added 2026-11-15)

Scripts: `s5b_run.py` (`run_s5b.sh`), `s5b_analysis.py`, `s5c_fisher.py`, `s5c_width_study.py`.
Runs: `solver_S5b/`, `solver_S5c/`. Log: SOLVER_LOG S5b, S5c.

| file | content |
|---|---|
| `s5b_adjoint_vs_direct.csv` | per run (t10: α law rtol 1e-10; pw30: piecewise p = 31; cost) and window: J, relative difference adjoint vs direct for each direct p |
| `s5b_adjoint_vs_fd.csv` | adjoint dJ/dθ vs central FD of J from the S5a replay runs, per window, parameter and h (valid_replay: both ± runs keep b1's increments through the last bin) |
| `s5b_scaling.csv` | piecewise law: direct wall time for p = 4, 8, 16, 31 and adjoint per J, one forward run |
| `s5b_cost.csv` | coarse and production: forward, adjoint (rhs / solve / back), storage, peak RSS (measured before the refinement solves were added) |
| `s5c_fisher.csv` | Fisher matrices (i) pre-peak, (ii) full filtered, (iii) full raw; θ3, θ4; s = 1e-3, 1e-4 P_peak: eigenvalues, condition, CRB, weak eigenvector; schedules A–D |
| `s5c_stability.csv` | B, C, D vs A: relative eigenvalue changes, eigenvector angles, δ where the schedules split |
| `s5c_width_study.csv` | eigenvalue and weak-direction stability vs bin width (1–16 passages) |
| `s5c_directions.csv` | trade-off slopes d ln σ_c/dα: sweeps iso-peak (both solvers), D18 pairs (raw and corrected), S5a ∂P_peak/∂θ, Fisher weak directions; angles to the sweep direction |
| `s5c_rayleigh.csv` | vᵀFv/λ_min of each candidate direction |
| `s5c_fdcheck.csv` | sweep FD slopes vs S5a, bin-averaged (S5c item 4) |
| `s5c.png/.pdf` | (a) raw and bin-averaged sensitivities with P; (b) Fisher spectra, four schedules; (c) trade-off directions |

The noise levels s are assumed, not measured.
