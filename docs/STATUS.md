# Project status (handoff page)

_Last updated: 2026-11-14 (afternoon) by Claude Code. Overwrite this page each session._

## Where we are

**Phase 0 — infrastructure and verification.**

Agreed: D1–D7, D9 (G_c 0.1–0.4 N/mm, K = 8500), D10 (energy constraints + tolerances),
D11 (elastic adherends, H 6.324, a0 18, t 0.4, clamp/load), D12 (σ_c lower bound from
ℓ_pz ≤ c·H, c = 1 proposed), D13 (Negishi only).

Done:
- `src/tslid/laws.py` (α-family, E1–E3), 11 unit tests pass.
- Generator `--law alpha`; rejects laws violating E1–E3.
- **Single-element verification passed (14/14)** — `results/phase0/single_element/`.
  ABAQUS EXPONENTIAL == s(x; α), α < 0 accepted, TABULAR within 1.6e-5 σ_c, dissipation = G_c.

Done (pilot): **ℓ_cz map, α = 0, 21/21 runs** — `results/phase0/pz_map/`. ℓ_cz/H is a
single function of ℓ_ch/H; ℓ_cz ≈ 0.02–0.14 ℓ_ch. c = 1 → σ_c,min 10.5 / 16.6 / 21.0 MPa
at G_c 0.1 / 0.25 / 0.4. Energy checks pass, except E4 at σ_c = 5 (ℓ_cz > 2.7H, out of bounds).

## Warning — zone lengths

SDEG-based lengths (pilot preview, campaign03 Stage F) **underestimate** the zone by up to
~25 %. Use ℓ_cz from the opening profile (D14, agreed). ℓ_pz is undefined until the user
settles it.

Pilot preview [D, α = 0, 17 runs]: ℓ_cz/H collapses onto one curve in ℓ_ch/H:
ℓ_ch/H 1.5 / 3.1 / 5.5 / 12.4 / 50 → ℓ_cz/H 0.18 / 0.42 / 0.66 / 1.05 / 1.94.

## Running

- **Convergence study** (numerical error floor), Negishi SLURM 45824413,
  `/scratch/negishi/basu56/tslid_runs/phase0_convergence/`: G_c 0.4, α 0, Δ 1.6 mm;
  σ_c 40 / 58 / 21 × {base h 0.1, h 0.05, h 0.025 (40, 58), n_inc 1600, visc 1e-4, 1e-5 (40)}.

## Next actions

1. Error floor ε_num from the convergence study → choose production settings.
2. Dense σ_c sweep at G_c 0.4, α 0 within [21, 58.3] MPa (D15), with the production
   settings and finer increments near the peak; compare peak and post-peak differences with ε_num.
3. Then α at fixed (G_c, σ_c) — one parameter at a time (D15).
4. User: ℓ_pz definition; α range.

First result (D15, pilot runs): `results/phase1a_sigc/` — at G_c 0.4, σ_c 25–57 changes P_peak
by −5.8 … +3.5 %, while the post-peak P–Δ is identical to ≤ 0.12 % (error floor pending).

## Open questions for the user

- c in ℓ_cz ≤ c·H: c = 0.5 / 1 / 1.5 gives σ_c,min (G_c 0.4) = 36 / 21 / 14 MPa.
- α range for Phase 1 (now that α < 0 works): e.g. −3 … 7?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7).
- Figures follow `FIGURE_DESIGN.md` (IJF guide) conventions unless told otherwise.
