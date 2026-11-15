# Cohesive interface in `tslid.fe` (S2)

Units: mm, N, MPa, N/mm, s. Code: `src/tslid/fe/laws_jax.py`, `interface.py`, `benchmarks.py`
(`single_cohesive`, `se_case`, `two_block_bar`). Tests: `tests/fe/test_s2_*.py`.
Results: `results/solver/S2_interface/`.

## What the ABAQUS decks specify (re-read from the code, 2026-11-14) [D]

From `build_tpeel_inp.py` → `AlphaLaw.abaqus_cards` (`laws.py`), and the Phase 0 single-element script:
- `*Elastic, type=TRACTION` with K_nn = K_ss = K_tt = K0 (8500 MPa/mm); `*Cohesive Section …
  thickness=SPECIFIED` with data line `1.0, W`. The constitutive thickness is 1, so nominal strain
  equals separation. ✔
- `*Damage Initiation, criterion=QUADS`, with the three strengths all equal to σ_c. ✔
- `*Damage Evolution, type=DISPLACEMENT`: LINEAR (α = 0, data δ_f − δ_0), EXPONENTIAL (δ_f − δ_0, α)
  or TABULAR ((D, δ_m − δ_0) table, 1000 rows clustered with exponent 2). ✔
- No `*Damage Evolution … mixed mode behavior` / mode-mix keyword, so evolution depends on δ_m
  only. ✔
- `*Section Controls, viscosity=μ` (T-peel default 1e-3 s in the builder; production 1e-4 s, D16;
  Phase 0 single-element runs 0). ✔
- Not in the deck: no `*Damage Stabilization`, and no element deletion in the single-element runs
  (`element deletion=NO`). In the T-peel decks deletion is left at the ABAQUS default.

## ABAQUS definitions used

> **Citation status: from memory, NOT verified against the manual text** (the manual is not
> installed on Bell or Negishi, and help.3ds.com needs a login). User: please confirm.
> *Abaqus Analysis User's Guide*, "Defining the constitutive response of cohesive elements using a
> traction-separation description" (numbered §32.5.6 in 6.x-era manuals), and *Abaqus Theory Guide*,
> cohesive elements. Every formula below was then **checked numerically** against ABAQUS 2026
> single-element runs (tests 2–4).

- Effective separation δ_m = √(⟨δ_n⟩² + δ_s²) in 2D (compressive normal separation excluded).
- QUADS: (⟨t_n⟩/t_n⁰)² + (t_s/t_s⁰)² = 1. With equal strengths σ_c and equal stiffnesses K0, this
  means K0 δ_m = σ_c at onset, i.e. δ_m⁰ = σ_c/K0 for every mode mix.
- Stiffness degradation: t_n = (1 − D) K0 δ_n for δ_n ≥ 0, t_n = K0 δ_n in compression;
  t_s = (1 − D) K0 δ_s.
- D = D(δ_m,max), where δ_m,max is the largest δ_m in the loading history (no healing; secant
  unloading).
- LINEAR: D = δ_f(δ_max − δ_0) / [δ_max(δ_f − δ_0)].
  EXPONENTIAL: D = 1 − (δ_0/δ_max){1 − [1 − e^{−α x}]/[1 − e^{−α}]}, x = (δ_max − δ_0)/(δ_f − δ_0).
  Both are the secant D = 1 − σ(δ_max)/(K0 δ_max) of the α family (Phase 0 [D]).
- Viscous regularisation: Ḋ_v = (D − D_v)/μ; D_v replaces D in the stiffness. ABAQUS integrates
  this with backward Euler: D_v,n+1 = (Δt D_n+1 + μ D_v,n)/(μ + Δt).

## Our formulation

- **Separations** from matched node pairs on the lower and upper bond faces: [[u]] = u_top − u_bot,
  interpolated linearly along each segment. δ_n = [[u]]·n and δ_s = [[u]]·t in the reference segment
  frame (`frame="global"`, S2). `frame="midsurface"` uses the deformed mid-surface tangent, as
  COH2D4 does under NLGEOM [H]; it is implemented but **not yet tested (open item for S3)**.
- **Energy with frozen damage**:
  ψ = (1 − D) K0/2 (⟨δ_n⟩₊² + δ_s²) + K0/2 ⟨−δ_n⟩₊², integrated over the reference segment length × W.
  The traction from `jax.grad(ψ)` is exactly the ABAQUS secant law above [D, test 1 and test 6].
- **History**: `update_history(u, hist, law, mu, dt)` updates δ_m,max, D and D_v (backward Euler).
  It is called after a solve. Within a solve the damage is frozen; the staggered vs monolithic
  choice is S3's.
- **Law representations**: `AlphaLawParams` (analytic; matches LINEAR/EXPONENTIAL) and
  `TabularLawParams` (linear interpolation of the same (D, u) table that ABAQUS receives; matches
  TABULAR).
- **Gradient safety** (`laws_jax.py`): s(x; α) and g(α) are ratios of power series for |α| < 0.5,
  so the 0/0 at α = 0 cancels exactly and there is no cancellation error. Branches use the
  double-`where` pattern. √ of δ_m has a guarded gradient at 0. Tests at δ = 0, δ_0, δ_f, α = 0 and
  compression are in `test_s2_gradients.py`.

## Quadrature of COH2D4

Decided by an elastic probe, not by recollection: a single COH2D4 with only one top node opened
(δ varies linearly along the element). With 2-point Gauss, the integration points sit at
x = (1 ∓ 1/√3)/2 and carry S22 = K0 δ(x). With nodal (Newton–Cotes) integration they carry 0 and
K0 δ_probe. **Result [D]** (Negishi job 45857972, `s2_probe`): S22 = 0 and 0.8500000 MPa (δ_probe =
1e-4 mm), against a Gauss prediction of 0.180 / 0.670. **COH2D4 integrates at the nodes
(2-point Newton–Cotes).** `CohesiveInterface` defaults to `quadrature="nodal"`; `"gauss"` is kept
for comparison.

## Viscous regularisation: measured behaviour [D]

The backward-Euler ODE above matches ABAQUS to ≤ 4.8e-8 σ_c until the inviscid D reaches 1. After
that, ABAQUS sets D_v → 1 faster (immediately for μ/Δt ≤ 0.2). We do not reproduce this; the
difference is ≤ 3.4e-5 σ_c in the traction-free tail (details in the S2 results README). In these
runs ABAQUS reports the regularisation energy outside ALLVD, consistent with `energy_checks.py`
using ALLCD.

## Open items

- Deformed-frame separations (COH2D4 under NLGEOM): S3.
- Manual citations above: user to confirm.
