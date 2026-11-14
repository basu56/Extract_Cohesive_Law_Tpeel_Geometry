# Project status (handoff page)

_Last updated: 2026-11-14 by Claude Code. Overwrite this page each session._

## Where we are

**Phase 0 — infrastructure and verification. No runs yet.**

Agreed so far: synthetic only (D1), P–Δ as the only observable (D2), α shape family (D3),
context workflow (D4/D6), build on the campaign03 generator without reproducing the
calibration (D7), G_c ∈ [0.1, 0.4] N/mm from experimental evidence (D9), and energy
constraints as a guiding principle (D10, checks in RESEARCH_PLAN §4b).

## Next actions (in order)

1. Copy `build_tpeel_inp.py` + `extract_tpeel_odb.py` from `TPeel_cohesive_zone`
   (`742e1ae`) into `src/tslid/`, recording the source commit; add the α law
   (`softening=EXPONENTIAL`, and TABULAR as an alternative).
2. `src/tslid/laws.py` + unit tests: G_c closed form vs quadrature (E1), φ_e (E2),
   admissibility (E3).
3. Single cohesive-element test on Negishi: ABAQUS EXPONENTIAL / TABULAR vs analytic
   σ(δ) for α ∈ {0, 1, 3, 7} (and α < 0 if accepted); check dissipated energy = G_c.
4. `energy_checks.py`: E4–E7 from the extracted histories, applied to every run.
5. Structural convergence at the reference law (G_c = 0.40): h_coh, increment size,
   viscosity → numerical error floor ε_num. Note: δ_f is only 0.005–0.08 mm and the
   process zone may be short at high σ_c; check that h_coh = 0.1 mm resolves it
   (≥ 3–5 elements).
6. Phase 1A sweep design over (G_c, σ_c, α) within D9 bounds.

## Open questions for the user

- **σ_c bounds**: adhesive strength data to set the lower/upper σ_c (only the energy
  cap √(2G_cK) is fixed so far).
- **E_a** for K = E_a/t (3400 MPa is assumed in campaign03; what is the source?).
- D11: elastic adherends and a0 = 18 mm for Phase 1?
- Proposed tolerances in §4b (E2 φ_max = 0.5, E4 1 %, E5 1 %, E6 2 %): acceptable?
- Noise model and P–Δ sampling (RESEARCH_PLAN §7); plotting style file.
