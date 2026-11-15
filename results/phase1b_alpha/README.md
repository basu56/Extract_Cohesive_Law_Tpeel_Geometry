# Shape (α) sweep at fixed G_c = 0.40 N/mm and fixed σ_c (D17)

Source: Negishi SLURM 45840508 (14/14 completed), `scripts/phase1/run_alpha_sweep.sh`, rundir
`/scratch/negishi/basu56/tslid_runs/phase1b_alpha_sweep/`. Production settings D16 (h_coh 0.1,
viscosity 1e-4, n_inc 800, Δ 1.6). Reference: α = 0 at the same σ_c. Error floor ≤ 0.04 % of P_peak.
Files: `gc040_s40_alpha.*`, `gc040_s21_alpha.*`, `alpha_energy_summary.csv`.
Comparison window (compare_ld.py, changed 2026-11-14): from 1.2 δ_peak to the δ at which the
**σ_c front** reaches L − 15 mm. (It used to be the traction-free tip; for long zones that let the
end-of-ligament collapse into the window: 29 % artefact at σ_c 21, α 7. The σ_c-sweep numbers are unchanged.)

## σ_c = 40 MPa

| α | ℓ_cz/H | P_peak (N) | ΔP_peak | max post-peak \|ΔP\|/P_pk | E6 |
|---|---|---|---|---|---|
| −3 | 0.31 | 576.6 | +2.58 % | 0.07 % | −0.05 % |
| −1.5 | 0.35 | 571.1 | +1.61 % | 0.06 % | −0.08 % |
| 0 | 0.42 | 562.1 | ref | ref | −0.13 % |
| 1.5 | 0.54 | 550.1 | −2.14 % | 0.04 % | −0.21 % |
| 3 | 0.69 | 538.7 | −4.16 % | 0.05 % | −0.32 % |
| 5 | 0.93 | 529.9 | −5.73 % | 0.08 % | −0.52 % |
| 7 | **1.16** | 527.2 | −6.20 % | 0.25 % | −0.67 % |

## σ_c = 21 MPa

| α | ℓ_cz/H | P_peak (N) | ΔP_peak | max post-peak \|ΔP\|/P_pk | E6 |
|---|---|---|---|---|---|
| −3 | 0.77 | 539.0 | +4.72 % | 0.09 % | −0.19 % |
| −1.5 | 0.86 | 529.3 | +2.83 % | 0.06 % | −0.24 % |
| 0 | 1.00 | 514.7 | ref | ref | −0.32 % |
| 1.5 | **1.21** | 497.2 | −3.40 % | 0.07 % | −0.46 % |
| 3 | **1.48** | 482.2 | −6.31 % | 0.13 % | −0.75 % |
| 5 | **1.89** | 471.8 | −8.33 % | 1.45 %* | −1.2 % |
| 7 | **2.28** | 468.7 | −8.93 % | 2.09 %* | −1.7 % |

**Bold** = outside the c = 1 bound (ℓ_cz > H, D12), run on purpose (D17).
\* These are not in the propagation branch: the peak-region difference reaches δ ≈ 0.45–0.48 mm
for these long zones, past the window start (0.269 mm). Beyond δ ≈ 0.48 mm, |ΔP| < 0.1 %.
Energy checks: E4 ≤ 3.5e-5 and E5 ≤ 5.8e-4 for all runs. E6 is within the 2 % tolerance but grows with ℓ_cz
(−1.7 % at ℓ_cz = 2.3 H). [H] The steady window starts at a0 + 5 mm, before a 14 mm zone is fully developed.

## Findings
[D] As with σ_c, α changes **only the region around the peak**. The elastic slope is identical
(≤ 0.08 %), and the propagation branch is identical to ≈ the floor.
[D] **The shape effect on P_peak is as large as the σ_c effect, not smaller.** Over α −3 … 7, P_peak spans
8.8 % (σ_c 40) and 13.6 % (σ_c 21). Over σ_c 21 … 58 at α = 0 it spans 12.2 %.
Higher α (a convex law with a long tail) lowers the peak. The effect is larger at σ_c = 21 (longer ℓ_cz),
consistent with H2.
Claude Web predicted a shape effect smaller than the σ_c effect. That is not borne out for P_peak.

## σ_c–α trade-off: matched-peak pairs (exploratory)
Laws with the same G_c and different (σ_c, α) can give the same P_peak. Rundir `phase1c_tradeoff/`
(symlinks to the two sweeps). Files: `pair_p529.*`, `pair_p540_summary.csv`.
Whole-curve max |ΔP|/P_pk over 0 ≤ δ ≤ 1.22 mm:

| Reference | Other law | ΔP_peak | Δδ_peak | Whole-curve max \|ΔP\| (at δ) | rms |
|---|---|---|---|---|---|
| σ_c 25, α 0 | σ_c 21, α −1.5 | +0.10 % | +1.0 % | 1.69 % (0.146 mm) | 0.43 % |
| σ_c 25, α 0 | σ_c 40, α 5 | +0.21 % | −6.7 % | 5.12 % (0.154 mm) | 1.16 % |
| σ_c 25, α 0 | σ_c 40, α 7 | −0.29 % | −6.7 % | 5.00 % (0.154 mm) | 1.13 % |
| σ_c 30, α 0 | σ_c 21, α −3 | −0.64 % | +3.0 % | 3.59 % (0.154 mm) | 0.89 % |
| σ_c 30, α 0 | σ_c 40, α 3 | −0.70 % | −2.0 % | 2.50 % (0.152 mm) | 0.54 % |

[D] P_peak alone cannot separate σ_c from α. The pairs remain distinguishable only through the
**nonlinear rising branch just before the peak** (δ ≈ 0.1–0.22 mm), by 1.7–5 % of P_peak. After the peak
they agree to ≤ 0.11 %.
[I] The information on (σ_c, α) is concentrated in a narrow window around the peak (about δ 0.1–0.3 mm).
The closest pair (σ_c 25, α 0 vs σ_c 21, α −1.5) differs by 1.7 % at most. Whether that survives
experimental noise sets practical identifiability (E8–E10, RESEARCH_PLAN §7).
The figure legend in `pair_p529.png` shows σ_c only. The lines are σ_c 25 α 0 (thick), σ_c 40 α 5 and α 7
(dark), and σ_c 21 α −1.5 (light).
