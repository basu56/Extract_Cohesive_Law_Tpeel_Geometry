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

## Next actions

1. User: choose c (ℓ_cz ≤ c·H) and the α range; define ℓ_pz (evening).
2. ℓ_cz map for α ≠ 0 at a few ℓ_ch/H values (shape shifts the curve).
2b. Shorten the prescribed Δ (crack completes at 0.7–1.4 mm; 3 mm wastes ~half the run).
3. Convergence at the reference law (h_coh, increment, viscosity) → ε_num.
4. Phase 1A sweep design over (G_c, σ_c, α) within the bounds.

## Open questions for the user

- c in ℓ_cz ≤ c·H: c = 0.5 / 1 / 1.5 gives σ_c,min (G_c 0.4) = 36 / 21 / 14 MPa.
- α range for Phase 1 (now that α < 0 works): e.g. −3 … 7?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7).
- Figures follow `FIGURE_DESIGN.md` (IJF guide) conventions unless told otherwise.
