# Project status (handoff page)

_Last updated: 2026-11-14 by Claude Code. Overwrite this page each session._

## Where we are

**Phase 0 — infrastructure and verification. Nothing has been run yet.**

- Repo initialized with CLAUDE.md, RESEARCH_PLAN, DECISIONS (from Claude Web session
  2026-11-14) plus the context protocol, this page and a review of the existing
  ABAQUS model (`docs/EXISTING_MODEL.md`).
- Forward-model infrastructure already exists in `basu56/TPeel_cohesive_zone`
  (campaign03 generator + ODB extractor + Negishi runner), verified on Abaqus 2026.
  Not yet copied or imported into this repo (waiting on D7).

## Next actions (in order)

1. **User:** confirm or reject D3 (α-family), D4/D6 (context workflow), D7 (reuse generator).
2. Bring the forward-model code into this repo (`src/tslid/`), once D7 is settled.
3. Implement `laws.py`: α-family, G_c(α) closed form, Abaqus EXPONENTIAL and TABULAR
   cards; unit tests for the energy integral.
4. Single cohesive-element test (Negishi): EXPONENTIAL vs TABULAR vs the analytic
   σ(δ), for α ∈ {−2, 0, 1, 3, 7}; check whether α < 0 is accepted.
5. Structural reference runs on the calibrated geometry: mesh (h_coh), increment and
   viscosity convergence of P–Δ → numerical error floor ε_num.
6. Phase 1A sweep design over (ℓ_ch/H, α), respecting σ < √(2 G K).

## Open questions for the user

- Reference law for Phase 1: G_c = 0.40 N/mm (SG24 tip) or another value? Fixed K
  = 8500 MPa/mm (physical E_a/t) or a stiffer penalty?
- Geometry for synthetic studies: a0 = 17 or 18 mm? Elastic adherends (simpler,
  campaign01 says plasticity is irrelevant)?
- Noise model and number of P–Δ samples (RESEARCH_PLAN §7).
- Plotting style: reuse `TPeel_cohesive_zone/scripts/plotstyle.py` / `example_plot/template_plot.py`?
- Should Claude Web sessions get the repo through the GitHub integration of a Claude
  Project, or will you paste the files?
