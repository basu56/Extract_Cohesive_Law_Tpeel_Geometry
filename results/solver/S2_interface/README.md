# S2 — cohesive law and interface: tslid.fe vs ABAQUS COH2D4

Units: mm, N, MPa, N/mm, s. One cohesive element, 1 mm long, width 1, K0 = 8500 MPa/mm.
ABAQUS 2026, Negishi: the 14 Phase 0 runs (job 45820978, `phase0_single_element/`) plus 10 new S2 runs
(job 45857972, `scripts/fe/run_s2_abaqus.sh`, decks from `scripts/fe/build_se_deck.py`, raw data in
`/scratch/negishi/basu56/tslid_runs/solver_S2_interface/`). All ODBs were re-extracted per integration point
(`scripts/fe/extract_se_odb.py`). Comparison: `scripts/fe/s2_compare.py` (Bell, uv env).

| File | Content |
|---|---|
| `s2_single_element.csv` | per run: traction error vs ABAQUS (normal, shear), damage error, unloading error, dissipation and work vs G_c and vs ABAQUS, viscous error split before/after inviscid D = 1 |
| `s2_viscous.csv` | μ sweep on one path: μ → 0 limit, effect of μ/Δt |
| `s2_probe.csv` | COH2D4 quadrature probe |
| `abaqus_se_reference.npz` | ABAQUS histories (IP 1, float32) + metadata; read by `tests/fe/test_s2_single_element.py` |
| `s2_interface.png/pdf` | (a) mode I, (b) shear and mixed mode, ABAQUS (grey) vs tslid.fe; (c) error per run |

## Results [D]

**Test 1 — material point vs `tslid.laws`** (36 parameter sets × 4004 separations): traction and
damage agree to ≤ 9.8e-16 σ_c (threshold 1e-12). Side finding: `tslid.laws.g_alpha` loses up to
1.6e-12 to cancellation near α = 1e-4 (not at any α we use). `laws_jax` uses a series ratio for
|α| < 0.5 and is exact to ≤ 3 ulp against a 50-digit reference.

**Test 2 — 14 Phase 0 runs** (mode I, LINEAR / EXPONENTIAL / TABULAR, α = −2 … 7, two cyclic):
- traction vs ABAQUS ≤ 6.8e-8 σ_c, all runs (thresholds 1e-6 for LINEAR/EXPONENTIAL, 2e-5 for TABULAR).
  TABULAR is matched to the same level because `TabularLawParams` interpolates the same table.
- secant unloading vs ABAQUS ≤ 1.7e-8 σ_c (threshold 1e-6).
- dissipation integrated over the 2000-increment path: equal to ABAQUS ALLDMD to ≤ 4e-8 G_c. Versus G_c it
  is ≤ 7.8e-6 for monotonic runs and **1.04e-5 for the cyclic EXPONENTIAL run, just above the 1e-5
  threshold**. ABAQUS has the identical value (it is the trapezoid error of the path integration).
  The model's exact dissipation is G_c to 1e-12 (test 5).

**Test 3 — mixed mode** (new decks: pure shear and δ_s = δ_n, for LINEAR, EXPONENTIAL and TABULAR):
normal and shear tractions vs ABAQUS ≤ 4.8e-8 σ_c. This confirms QUADS onset at K0 δ_m = σ_c and
evolution in δ_m only.

**Test 4 — viscous regularisation** (μ = 1e-2 and 1e-4 s at T = 1 s; μ = 1e-4 s at the T-peel rate
1 mm/min, T = 4.99 s):
- Until the inviscid damage reaches 1: traction vs ABAQUS ≤ 4.8e-8 σ_c. ABAQUS integrates
  Ḋ_v = (D − D_v)/μ by backward Euler, exactly as implemented.
- After D = 1 (δ > δ_f, a traction-free tail): ABAQUS drives D_v to 1 faster than backward Euler and
  sets it to exactly 1. This happens at once when μ/Δt ≤ 0.2, and after about 85 increments when
  μ/Δt = 20. Residual difference ≤ 3.4e-5 σ_c (μ/Δt = 20), 1.4e-5 (0.2), 2.7e-6 (0.04).
  **Not reproduced; the rule is unknown without the manual [H].** Negligible at production
  settings (μ/Δt ≈ 8e-4).
- Total work vs ABAQUS ALLWK ≤ 6.1e-6 G_c. In these runs ABAQUS reports ALLVD = 0, and the
  regularisation energy is outside ALLDMD (0.054 N·mm/mm² at μ = 1e-2). It is not ALLVD; our
  `energy_checks.py` already counts ALLCD for it.
- μ → 0: μ = 1e-12 s reproduces the inviscid run to 3.9e-11 σ_c (`s2_viscous.csv`).

**Test 5 — gradients** (`tests/fe/test_s2_gradients.py`): AD vs central FD (h = 1e-6 |θ_i|) of the
traction wrt (K0, σ_c, G_c, α):
- ≤ 1e-6 relative at α ∈ {0, 3, −1.5, 5e-4} and at points 1e-3 inside each branch.
- Finite AD at δ = 0, δ_0, δ_f and 1.2 δ_f, and in compression. At the kinks AD equals a one-sided
  FD (the law is not differentiable there in θ).
- Dissipated energy (Gauss–Legendre over [0, δ_f]) = G_c to 1e-12, with dG/dG_c = 1 within 1e-8
  and the other derivatives ≤ 1e-8.

**Test 6 — assembly** (`two_block_bar`: two Q4IM blocks with one cohesive segment, opened through
failure): reaction = traction × area to 7.3e-13 σ_c A, and traction = law(opening) to 1.1e-15 σ_c.
Staggered history needed up to 17 solve/update passes per load step in softening.

**Quadrature** (`s2_probe`): one top node opened by 1e-4 mm gives S22 = 0 and 0.8500000 MPa at the two
integration points. That is the nodal prediction; Gauss would give 0.180 and 0.670.
**COH2D4 uses nodal (Newton–Cotes) integration.** `CohesiveInterface` now defaults to it.
