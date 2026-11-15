# S4a — σ_c and α sweeps at G_c = 0.4: tslid.fe vs ABAQUS

Units: mm, N. Details: `docs/SOLVER_LOG.md` (S4a). Decisions: D25, D26.
Script: `scripts/fe/s4a_compare.py`, which uses `src/tslid/analysis/compare_solvers.py`.
Runs: tslid.fe with the D24 defaults, 4 threads (`scripts/fe/run_s3c_prod.sh`), Negishi
`tslid_runs/solver_S4a/`. ABAQUS references: `phase1a_sigc_sweep/`, `phase1b_alpha_sweep/`.
There are 23 runs of 21 laws: `as_s40_a0` and `as_s21_a0` repeat `sw_s40` and `sw_s21`.

| file | content |
|---|---|
| `s4a_compare.csv` | one row per run: σ_c, α, increments, cutbacks, iterations, wall, all `compare_solvers.py` fields. The load gates use the compare_ld window. E4–E6 are computed in the D26 window (σ_c front at L − 15 mm) for both solvers; the default-window values are in `*_tipwin_*` |
| `s4a_pairs.csv` | D18 matched-peak pairs: per solver, the pre-peak pair difference (P_other − P_ref)/P_peak over 0 ≤ δ ≤ the later of the two peaks (max, its δ, rms); and the difference between the solvers' pair differences |
| `s4a_duplicates.csv` | same law run twice, per solver |
| `s4a.png/pdf` | (a) max \|ΔP\| vs ABAQUS per run; (b) pair differences, tslid.fe lines vs ABAQUS markers; (c) tslid.fe − ABAQUS pair difference |

**Results [D].**
- **All 23 runs pass every load gate and the 0.1 % target:**
  - max \|ΔP\| 2.8e-5 – 5.5e-4 P_peak (largest `as_s40_am3`, then `sw_s58` 3.6e-4);
  - \|ΔP_peak\| ≤ 8.3e-5; Δδ_peak −4e-5 mm; \|Δℓ_cz\| ≤ 0.04 elements.
- **Energy checks (D26 window):**
  - E4 and E5 pass in all 23, for both solvers. E5 ours / ABAQUS agree to ≤ 7 %.
  - E6 ours and ABAQUS agree to ≤ 1e-4 absolute in every run.
  - **E6 fails for `as_s21_a7` in both solvers, identically (−0.0212; tolerance 0.02).** In the
    default window it is −0.0176 / −0.0173.
  - E6 grows with α in both solvers, e.g. σ_c 21: −0.0029 (α −1.5) … −0.0212 (α 7). That is a
    property of the case and window, not of the solver.
- **D18 matched-peak pairs:**

  | pair | pair signal max \|ΔP\|/P_peak (ABAQUS / tslid.fe) | solvers' pair differences differ by | relative |
  |---|---|---|---|
  | σ_c 21, α −1.5 vs σ_c 25, α 0 | 1.693 % / 1.693 % | 8.1e-6 P_peak | 0.05 % |
  | σ_c 40, α 5 vs σ_c 25, α 0 | 5.119 % / 5.118 % | 3.1e-5 | 0.06 % |
  | σ_c 40, α 7 vs σ_c 25, α 0 | 5.001 % / 5.001 % | 2.9e-5 | 0.06 % |
  | σ_c 21, α −3 vs σ_c 30, α 0 | 3.587 % / 3.587 % | 1.0e-5 | 0.03 % |
  | σ_c 40, α 3 vs σ_c 30, α 0 | 2.497 % / 2.496 % | 1.05e-4 | 0.42 % |

  The location of each pair's maximum also agrees (δ = 0.146–0.154 mm in both). The largest
  discrepancy (1.05e-4) is a single-element snap at δ ≈ 0.135 mm.
- **Duplicate laws:**
  - ABAQUS is bit-identical (0.0).
  - tslid.fe differs by 1.8e-10 / 1.9e-10 P_peak with the same increments. [H] This comes from
    reduction order in multithreaded PARDISO/XLA.
