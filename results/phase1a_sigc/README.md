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
