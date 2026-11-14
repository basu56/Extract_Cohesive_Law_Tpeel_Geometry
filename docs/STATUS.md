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

Running:
- **Pilot process-zone map**, Negishi SLURM 45821190, `/scratch/negishi/basu56/tslid_runs/phase0_pz_map/`:
  α = 0, G_c ∈ {0.1, 0.25, 0.4}, σ_c ∈ {5…57} up to the E2 cap (21 runs). Output:
  ℓ_pz/H(G_c, σ_c) → σ_c,min (D12); E4–E6 energy checks per run; elements per ℓ_pz.

## Warning — zone lengths

ℓ_pz values from the SDEG 0.99/0 fronts (pilot preview, campaign03 Stage F) **underestimate**
the softening zone by up to ~25 %, with a bias that depends on σ_c and α. Opening-based
definitions are proposed (D14, awaiting user). See the session log, continuation 3.

## Next actions

1. Analyse the pilot: ℓ_pz/H map, σ_c,min(G_c) for c = 1, energy checks, mesh resolution.
2. Repeat for α ≠ 0 at a few points (ℓ_pz depends on shape at fixed G_c, σ_c).
3. Convergence at the reference law (h_coh, increment, viscosity) → ε_num.
4. Phase 1A sweep design over (G_c, σ_c, α) within the bounds.

## Open questions for the user

- c in ℓ_pz ≤ c·H (proposed 1) — decide after seeing the map.
- α range for Phase 1 (now that α < 0 works): e.g. −3 … 7?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7).
- Figures follow `FIGURE_DESIGN.md` (IJF guide) conventions unless told otherwise.
