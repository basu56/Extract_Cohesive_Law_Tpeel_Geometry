# σ_c at fixed G_c = 0.40 N/mm, α = 0 — first look (D15), from the pilot runs

Source: pilot runs `pz_g0p4_s*` (SLURM 45821190, h_coh 0.1, n_inc 400 over Δ = 3 mm,
viscosity 1e-3). In bounds (c = 1): σ_c = 25, 30, 40, 50, 57; σ_c = 5–20 drawn grey as context.
Reference σ_c = 40. Post-peak window 0.254–1.344 mm (ends when the traction-free tip reaches L − 15).
Script: `src/tslid/analysis/compare_ld.py`.

| σ_c | P_peak (N) | ΔP_peak | Δδ_peak | Δ initial slope | max post-peak \|ΔP\|/P_pk | rms |
|---|---|---|---|---|---|---|
| 25 | 528.9 | −5.75 % | +7.6 % | 0 | 0.12 % | 0.04 % |
| 30 | 542.7 | −3.29 % | +3.8 % | 0 | 0.07 % | 0.02 % |
| 40 | 561.2 | ref | ref | ref | ref | ref |
| 50 | 575.2 | +2.50 % | −3.8 % | 0 | 0.05 % | 0.01 % |
| 57 | 580.5 | +3.45 % | −7.6 % | 0 | 0.06 % | 0.02 % |

[D] Same G_c, different σ_c: identical elastic slope, P_peak changes −5.8 % … +3.5 %, and
the post-peak (propagation) response is identical to ≤ 0.12 %.
**Caveats:** (1) whether 0.12 % exceeds the numerical error floor is not yet known
(convergence study `scripts/phase0/run_convergence.sh`). (2) δ_peak is quantized by the
output increment (0.0075 mm ≈ 4 % of δ_peak), so the Δδ_peak values are resolution-limited.

---

# Dense σ_c sweep, production settings (D15, D16)

Negishi SLURM 45831057, commit 3cd52f3, `scripts/phase1/run_sigc_sweep.sh`.
G_c 0.40 N/mm, α = 0, σ_c = 21 … 58 MPa (the c = 1 bounds, D12); h_coh 0.1, viscosity 1e-4,
n_inc 800 (δ frames 0.002 mm ≈ 1 % of δ_peak). Reference σ_c = 40. All 9 runs completed; all pass
E4 (≤ 1.9e-5), E5 (≤ 1.0e-4), E6 (≤ 0.32 %). Files: `gc040_sigc_sweep.png/pdf`,
`gc040_sigc_sweep_summary.csv`, `sweep_energy_summary.csv`.
Raw: `/scratch/negishi/basu56/tslid_runs/phase1a_sigc_sweep/`.

| σ_c | ℓ_cz/H | P_peak (N) | ΔP_peak | Δδ_peak | post-peak max | post-peak rms |
|---|---|---|---|---|---|---|
| 21 | 1.00 | 514.7 | −8.43 % | +12.4 % | 0.21 % | 0.07 % |
| 25 | 0.83 | 528.7 | −5.93 % | +8.2 % | 0.14 % | 0.05 % |
| 30 | 0.66 | 542.5 | −3.49 % | +4.1 % | 0.08 % | 0.02 % |
| 35 | 0.53 | 553.3 | −1.56 % | +2.1 % | 0.04 % | 0.01 % |
| 40 | 0.43 | 562.1 | ref | ref | — | — |
| 45 | 0.34 | 569.3 | +1.29 % | −2.1 % | 0.04 % | 0.01 % |
| 50 | 0.27 | 575.3 | +2.35 % | −4.1 % | 0.06 % | 0.01 % |
| 55 | 0.20 | 580.8 | +3.32 % | −5.2 % | 0.08 % | 0.01 % |
| 58 | 0.17 | 583.2 | +3.76 % | −5.2 % | 0.06 % | 0.02 % |

Post-peak window δ = 0.262–1.338 mm. Initial slope identical (≤ 0.08 %). Floor (D16): ≤ 0.04 %.

Findings [D]:
1. Same G_c, different σ_c: the peak changes by −8.4 … +3.8 % (12 % across the admissible
   range), monotonic and concave in σ_c; 200× the error floor.
2. Propagation branch: identical to ≤ 0.21 % of P_peak. A small residual σ_c effect exists right
   after the peak — signed with σ_c (lower σ_c → higher P), mean +0.16 % for σ_c 21 in δ 0.26–0.4,
   +0.06 % in 0.4–0.8, +0.01 % in 0.8–1.34 (≈ floor). σ_c ≥ 30: at the floor everywhere.
3. Pre-peak: lower σ_c gives an earlier departure from linearity (panel b, up to −12 % at δ ≈ 0.18).
[I] The P–Δ information about σ_c is confined to the peak region (δ ≲ 0.4 mm); steady peeling
carries G_c only. This is the non-uniqueness of D15 in its first controlled form.
