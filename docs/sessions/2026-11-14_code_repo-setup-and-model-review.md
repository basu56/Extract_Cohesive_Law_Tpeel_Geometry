# 2026-11-14 — Claude Code — repo setup and review of the existing model

## Context / goal
Start the project "AI-based prediction of fracture constitutive laws in adhesives".
Review `add_project_context.sh` (from the Claude Web session), set up the shared repo,
review the existing T-peel ABAQUS model, and define how Code and Web stay in sync.

## What was done
- Cloned the repo over SSH to `/scratch/bell/basu56/Extract_Cohesive_Law_Tpeel_Geometry`
  and wrote the four files from the script's heredocs unchanged, except:
  - CLAUDE.md: filled the cluster TBDs from TPeel_cohesive_zone Stage F records
    (Negishi, `abaqus/2026`, account `tomar`, partition `cpu`, SSH-only GitHub);
    added a pointer to the context protocol.
- Added `docs/CONTEXT_PROTOCOL.md`, `docs/STATUS.md`, `docs/EXISTING_MODEL.md`,
  `docs/sessions/` (this log and a reconstructed log of the Web session), and D6–D8 in DECISIONS.
- Reviewed `basu56/TPeel_cohesive_zone` (branch `campaign03-bridging-czm` @ `742e1ae`).
  The user referred to its folder as "TPeel_cohesive_zone".

## Results
- [D] Two model generations exist. The legacy CAE decks are superseded. The campaign03
  generator `build_tpeel_inp.py` is parametric and verified (Bell 2025 ≡ Negishi 2026).
- [D] Stage F (constant G_tip = 0.40, σ_tip 20→100 MPa): plateau and R-curve unchanged,
  peak +2 %→+13 %, ℓ_pz/H 0.68→0.16. This is prior evidence on our own geometry for H1/H3.
- [D] Bilinear admissibility σ < √(2GK) caps σ ≈ 81 MPa at K = 8500, G = 0.4.
- [H] The α-family equals ABAQUS's built-in exponential softening; needs a
  single-element check, including whether α < 0 is allowed.

## Decisions
- D6 (proposed): context-sync workflow (CONTEXT_PROTOCOL.md).
- D7 (proposed): reuse the campaign03 generator as the forward model.
- D8 (proposed): truth with ABAQUS EXPONENTIAL, inversion with a different representation.

## Not done
- No ABAQUS runs. The forward-model code has not been copied into this repo yet.

## Next steps
See `docs/STATUS.md`.

---

## Continuation — user decisions and energy constraints

**User decisions:** D3 (α-family) agreed. D4/D6 agreed: Claude Code keeps notes in the repo
and Claude Web reviews it periodically. D7 agreed with a change: reuse the generator but do
not reproduce the experimental calibration; use experimental evidence only to bound
parameters. G_c ∈ 0.1–0.4 kJ/m², reference ≈ 0.40 N/mm, not exact.
The user stressed that **energy constraints must guide the whole project**: fracture energy
is constrained by evidence, and generated and inferred laws must stay physically reasonable.

**Done:**
- DECISIONS: D3/D4/D6/D7 updated; new D9 (parameter bounds), D10 (energy principle),
  D11 (elastic adherends, a0 = 18; proposed).
- RESEARCH_PLAN §4a parameter-bounds table with admissibility arithmetic; §4b energy
  constraints E1–E10 at law, simulation and inference level. CLAUDE.md rule 8.

**Results [D, arithmetic]:** with K = 8500, σ_c < √(2G_cK) = 41–82 MPa over the G_c range.
ℓ_ch/H spans ≈ 0.35–50. Energy partition φ_e = σ_c²/(2KG_c) reaches 0.94 at
G_c = 0.1, σ_c = 40, i.e. almost elastic-brittle, which motivates the E2 limit.

**Interpretation [I]:** E6 (dALLDMD/da = G_c W) and E8 (G_c × unit-area shape) make energy
both a dataset quality check and a structural prior for the inversion. E9 frames
identifiability as energy vs energy partition vs shape at fixed energy.

**Open:** σ_c bounds, source of E_a, D11, tolerances (see STATUS.md).

---

## Continuation 2 — first simulations (Negishi)

**User decisions:** K = 8500 agreed; D11 agreed; tolerances agreed; σ_c lower bound to be
set from the physical process-zone size (ℓ_c should not be much larger than H) → D12;
Bell stopped, compute on Negishi → D13.

**Done:**
- Imported generator/extractor/runner verbatim (a91db9e), then added `--law alpha`, D11
  defaults, laws.py + tests, single-element tools, energy_checks.py, pilot runner.
- Negishi bare-repo mirror + clone set up (see CLAUDE.md cluster section).
- Single-element verification: SLURM 45820961 **failed** (runner path / module bug, no
  analysis ran); fixed; 45820978 **passed 14/14** [D] — see `results/phase0/single_element/`.
- Pilot ℓ_pz map submitted: SLURM 45821190.

**Note:** `FIGURE_DESIGN.md` (user's IJF figure guide, added by the user) was swept into
commit 55858b6 by `git add -A`. From now on files are staged by name.

---

## Continuation 3 — zone-length definitions (user flagged as very important)

**Problem found [D]:** the "ℓ_pz" reported for the pilot preview came from campaign03's
extractor: the distance between the SDEG > 0 front and the SDEG ≥ 0.99 front, element-wise.
SDEG is stiffness degradation, not traction loss. With K = 8500, D = 0.99 is reached at
δ = 0.60 δ_f for (G_c 0.4, σ_c 10, α 0), where 41 % of σ_c still acts and 16 % of G_c is
not yet dissipated. For α = 7, 15 % of G_c remains. The bias therefore depends on σ_c and α.
Measured on 6 pilot runs, using the opening-based tip (δ = δ_f) vs SDEG 0.99:
ℓ/H 1.44 → 1.93 (G_c 0.4, σ_c 10), 1.30 → 1.60 (0.25, 10), 0.99 → 1.11 (0.25, 15),
0.41 → 0.42 (0.4, 40).
**The SDEG-based ℓ_pz values in earlier notes must not be used.** Definitions proposed
to the user (D14, pending): see STATUS.md.

## Continuation 4 (late evening) — convergence + dense σ_c sweep
- Convergence study (SLURM 45824413, 11/11): error floor ≤ 0.04 % of P_peak; production settings D16.
- Dense σ_c sweep at G_c 0.4, α 0 (SLURM 45831057, 9/9): P_peak −8.4 … +3.8 % about σ_c 40, initial
  slope identical, post-peak P–Δ identical to ≤ 0.21 % → `results/phase1a_sigc/README.md`.
- User asked why runs are slow. Answer: adherend mesh follows h_coh (26k CPE4I) plus full-field output
  every increment; not ℓ_cz ≪ H (≥ 11 el per ℓ_cz, mesh converged). Speed-ups proposed, not applied.
- User is checking Claude Web access to the repo before the next step.
