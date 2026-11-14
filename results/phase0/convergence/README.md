# Phase 0 — numerical error floor of P(δ) (convergence study)

Negishi SLURM 45824413, commit 9e338c8, runner `scripts/phase0/run_convergence.sh`.
G_c = 0.40 N/mm, α = 0, Δ = 1.6 mm. Base: h_coh 0.1 mm, n_inc 400, viscosity 1e-3 s.
One numerical setting changed at a time. All 11 runs completed.
Raw output: `/scratch/negishi/basu56/tslid_runs/phase0_convergence/`.

## Energy checks (`conv_energy_summary.csv`)
All pass. E4 (balance, now including ALLCD) ≤ 5.4e-5; E6 (dissipation) within 0.31 %.
E5 (viscous/damage) = 5.8e-4 – 8.9e-4 at viscosity 1e-3, 7.2e-5 at 1e-4, 7.2e-6 at 1e-5.
ℓ_cz/H = 1.00 (σ_c 21), 0.425 (40), 0.171 (58); independent of mesh.
Elements per ℓ_cz at base mesh: 63 / 27 / 11.

## P(δ) differences vs base (`conv_s{40,58,21}_summary.csv`; % of P_peak,ref)

| run | ΔP_peak | post-peak max | post-peak rms |
|---|---|---|---|
| s40 h 0.05 / 0.025 | +0.01 / +0.01 | 0.02 / 0.02 | 0.01 |
| s58 h 0.05 / 0.025 | +0.02 / +0.03 | 0.04 / 0.03 | 0.01 |
| s21 h 0.05 | 0.00 | 0.04 | 0.00 |
| s40 n_inc 1600 | +0.02 (δ_peak +0.8 %) | 0.02 | 0.01 |
| s40 visc 1e-4 / 1e-5 | −0.04 / −0.04 | 0.08 / 0.09 | 0.03 |

Error floor [D]: mesh and increment ≤ 0.04 %; viscosity 1e-3 adds a bias of ≈ 0.1 %
just after the peak, decaying to ≈ 0.02 % (visible in `conv_s40.png`, panel b, lower pair).
1e-4 and 1e-5 agree to < 0.01 %, so 1e-4 is converged.

Production settings (Phase 1): h_coh 0.1 mm, viscosity 1e-4 s, n_inc 800.

Note: the figures label every curve "σ_c = 40/58/21" (plot written for σ_c sweeps); the
curves are the runs in the table, in the order base, h025, h050, n1600, v1e-4, v1e-5.
