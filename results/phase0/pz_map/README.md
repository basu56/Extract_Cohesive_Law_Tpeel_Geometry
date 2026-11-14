# Phase 0 pilot — cohesive-zone length map, α = 0 (2026-11-14)

Run: Negishi SLURM 45821190 (≈ 70 min; licence queue limited concurrency to ~12 jobs),
Abaqus 2026, decks from repo @ 55858b6; analysis `src/tslid/analysis/energy_checks.py`
@ current HEAD. Raw: `/scratch/negishi/basu56/tslid_runs/phase0_pz_map/`.
Setup: D11 geometry, K = 8500, linear softening, G_c ∈ {0.1, 0.25, 0.4}, σ_c 5 → E2 cap,
h_coh = 0.1 mm, viscosity 1e-3 s, Δ = 3 mm at 1 mm/min. 21/21 runs completed.

ℓ_cz (D14): from the δ = δ_0 point (traction σ_c) to the δ = δ_f point (traction-free),
on the nodal opening profile, averaged while the traction-free tip is in [a0 + 5, L − 15] mm.

**Results [D]**
- Energy: E6 (dissipation per crack advance vs G_c·W) within 1 % for all 21 runs. E4
  (global balance, analysis window) ≤ 1.3e-3 for 19 runs; it fails for σ_c = 5 at G_c = 0.4
  (2.4e-2) and is marginal for σ_c = 5 at G_c = 0.25 (9.2e-3). These have ℓ_cz/H = 3.2 and
  2.7, so the zone reaches the end region. Both are outside the D12 bound.
- **Collapse:** ℓ_cz/H depends on ℓ_ch/H only. Pairs with equal ℓ_ch/H at G_c = 0.1 and 0.4
  agree to ≤ 0.4 % (ℓ_ch/H 3.11, 5.52, 12.42, 49.69 → ℓ_cz/H 0.425, 0.66, 1.053, 1.94).
- **ℓ_cz ≪ ℓ_ch:** ℓ_cz/ℓ_ch = 0.016–0.14. The local exponent d ln ℓ_cz / d ln ℓ_ch falls
  from 1.15 (ℓ_ch/H ≈ 2) to 0.37 (ℓ_ch/H ≈ 100): close to LEFM-like at small ℓ_ch,
  strongly confined by the adherends at large ℓ_ch [interpretation].
- Mesh: ≥ 11 cohesive elements per ℓ_cz in every run.
- σ_c lower bound from ℓ_cz ≤ c·H (α = 0; c not yet agreed):

| c | ℓ_ch/H limit | σ_c,min at G_c 0.1 / 0.25 / 0.4 (MPa) |
|---|---|---|
| 0.5 | 3.8 | 18.1 / 28.6 / 36.2 |
| 1.0 | 11.3 | 10.5 / 16.6 / 21.0 |
| 1.5 | 26.7 | 6.8 / 10.8 / 13.6 |

Upper bound (E2, φ_e ≤ 0.5): 29.2 / 46.1 / 58.3 MPa.

Files: `pz_map_summary.csv` (all metrics), `pz_map_lcz_vs_lch.{png,pdf}`.
