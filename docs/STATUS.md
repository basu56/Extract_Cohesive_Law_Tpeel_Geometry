# Project status (handoff page)

_Last updated: 2026-11-14 (evening) by Claude Code. Overwrite this page each session._

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

## Done (evening): convergence study — `results/phase0/convergence/`, D16
Error floor [D]: mesh and increment change P by ≤ 0.04 % of P_peak; viscosity 1e-3 biases P by
≈ 0.1 % just after the peak; 1e-4 ≡ 1e-5 (< 0.01 %). All energy checks pass (E4 ≤ 5.4e-5).
Production: h_coh 0.1, viscosity 1e-4, n_inc 800, Δ 1.6 (D16).

First result (D15, pilot runs, viscosity 1e-3): `results/phase1a_sigc/` — at G_c 0.4, σ_c 25–57
changes P_peak by −5.8 … +3.5 % (≫ floor), post-peak P–Δ differs by ≤ 0.12 % (≈ floor).

## Done: dense σ_c sweep (D15) — `results/phase1a_sigc/` (second section of README)
G_c 0.4, α 0, σ_c 21–58, production settings, 9/9 runs, all energy checks pass. P_peak −8.4 … +3.8 %
vs σ_c 40; post-peak P–Δ identical to ≤ 0.21 % (σ_c ≥ 30: ≤ 0.08 %, ≈ floor); the residual decays
away from the peak. σ_c information is confined to δ ≲ 0.4 mm.

## Running

Nothing.

## Next actions

1. α at fixed (G_c 0.4, σ_c 40): needs the user's α range (proposal −3 … 7).
2. Possibly repeat the σ_c sweep at G_c 0.1 and 0.25 (same design) to show the result holds across G_c.
3. User: ℓ_pz definition.

Git: the Negishi clone now tracks `main` (its old `master` branch is stale; leave it).

## Open questions for the user

- α range for Phase 1 (now that α < 0 works): e.g. −3 … 7?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7).
- Figures follow `FIGURE_DESIGN.md` (IJF guide) conventions unless told otherwise.
