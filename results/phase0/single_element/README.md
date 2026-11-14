# Phase 0 — single cohesive-element verification (2026-11-14)

Run: Negishi SLURM 45820978, Abaqus 2026, repo @ 55858b6 (code identical to 43a07fd for
these scripts). Raw: `/scratch/negishi/basu56/tslid_runs/phase0_single_element/`.
Script: `scripts/phase0/run_single_element.sh`; comparison:
`src/tslid/analysis/compare_single_element.py`. Failed first attempt 45820961 (runner
path bug, no analysis ran) kept in `failed_45820961/`.

One COH2D4, 1 mm², K = 8500, σ_c = 20, G_c = 0.4 (and G_c = 0.1, σ_c = 29, φ_e = 0.495),
no viscosity, opened to 1.2 δ_f in 2000 increments; two cyclic runs unload at
x = 0.4 to zero and reload.

**Result [D]: all 14 runs pass.**

| check | worst value | case |
|---|---|---|
| max traction error / σ_c, EXPONENTIAL & LINEAR | 1.0e-7 | — |
| max traction error / σ_c, TABULAR (1000 rows) | 1.6e-5 | α = 7 |
| dissipated energy vs G_c (E1 in ABAQUS) | 1.0e-5 | exp α = 3 cyclic |
| global balance ALLWK − ALLSE − ALLDMD − ALLVD (E4) | 0 (printed precision) | — |
| secant unloading deviation / σ_c | 2.0e-6 | tab α = 3 cyclic |

Conclusions:
- ABAQUS `softening=EXPONENTIAL` **is** the project's s(x; α) family (confirms D8's hypothesis).
- ABAQUS accepts **α < 0** (α = −2, concave) with EXPONENTIAL; it reproduces the law exactly.
- TABULAR with 1000 rows reproduces every shape, giving the independent representation
  for the inverse-crime rule.
- ALLDMD at full failure = G_c (the elastic energy at onset is included in ALLDMD by
  the end), so E6 can use ALLDMD + ALLVD directly.
